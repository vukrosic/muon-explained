"""Rare operations: does Muon learn the rare ones better or worse?

Task: "a op b = ?" modulo N (default 97), ops + - * /. Every op has the same number of training pairs, but ops are drawn into
batches with different frequencies (common ... rare). Half of each op's pairs are held out: test accuracy on them
shows whether the model learned the rule, not just memorized.

Muon matrices get U S^p V^T (exact SVD), scaled to the total size of exact Muon (every singular value 1).
p = 0: exact Muon, every direction pushed equally. p = 0.5: weak directions pushed less. p = 1: pushed as the gradient says.
Usage: python3 arith.py P ORDER SEED     ORDER: "+-*/" means + is the most common, / the rarest
"""
import json, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

N = int(os.environ.get("N", 97))                          # numbers 0..N-1, answers modulo N
FREQ = [0.55, 0.30, 0.12, 0.03]                           # share of each batch, most common op first
STEPS, B, EVERY = int(os.environ.get("STEPS", 3000)), 512, 50
MUON_LR, ADAM_LR, BETA, DECAY = 0.02, 0.01, 0.95, 0.5
WD = float(os.environ.get("WD", 0.05))                    # Muon weight decay, multiplied by lr
HERE = os.path.dirname(os.path.abspath(__file__))
OPS = {"+": lambda a, b: (a + b) % N, "-": lambda a, b: (a - b) % N, "*": lambda a, b: (a * b) % N,
       "/": lambda a, b: (a * pow(b, N - 2, N)) % N}


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(N + 4, 64)                   # tokens: 23 numbers + 4 op symbols
        s.inp = nn.Linear(192, 256, bias=False)
        s.h1 = nn.Linear(256, 256, bias=False)
        s.h2 = nn.Linear(256, 256, bias=False)
        s.out = nn.Linear(256, N)

    def forward(s, x):
        h = F.relu(s.inp(s.emb(x).flatten(1)))
        h = F.relu(s.h1(h))
        h = F.relu(s.h2(h))
        return s.out(h)


def data(order, gen):
    """Per op (in rarity order): train and test tensors of rows (a, op token, b, answer)."""
    tr, te = [], []
    for k, op in enumerate(order):
        rows = torch.tensor([(a, N + "+-*/".index(op), b, OPS[op](a, b)) for a in range(N) for b in range(N)
                             if not (op == "/" and b == 0)])
        rows = rows[torch.randperm(len(rows), generator=gen)]
        tr.append(rows[: len(rows) // 2]); te.append(rows[len(rows) // 2:])
    return tr, te


def shaped(G, p):
    if not torch.isfinite(G).all() or G.norm() < 1e-20:     # dead layer: no gradient, no update
        return torch.zeros_like(G)
    G = G.double()
    try:
        U, s, Vh = torch.linalg.svd(G, full_matrices=False)
    except torch._C._LinAlgError:                           # LAPACK rarely fails to converge: retry with a tiny jitter
        U, s, Vh = torch.linalg.svd(G + 1e-9 * G.norm() * torch.randn_like(G), full_matrices=False)
    U, s, Vh = U.float(), s.float(), Vh.float()
    O = (U * s.clamp_min(1e-30) ** p) @ Vh
    return O * (min(G.shape) ** 0.5 / (O.norm() + 1e-12))  # same total size as exact Muon


def acc(m, rows):
    with torch.no_grad():
        return (m(rows[:, :3]).argmax(1) == rows[:, 3]).float().mean().item()


def main(p, order, seed):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    tr, te = data(order, gen)
    m = Net()
    muon = [m.inp.weight, m.h1.weight, m.h2.weight]
    adam = torch.optim.AdamW([m.emb.weight, m.out.weight, m.out.bias], lr=ADAM_LR, weight_decay=0)
    mom = [torch.zeros_like(w) for w in muon]
    freq = torch.tensor(FREQ)
    log = []
    t0 = time.perf_counter()
    for step in range(STEPS + 1):
        if step % EVERY == 0:
            log.append(dict(step=step, train=[round(acc(m, r), 4) for r in tr], test=[round(acc(m, r), 4) for r in te]))
        if step == STEPS:
            break
        sched = min(1.0, (1 - step / STEPS) / DECAY)
        k = torch.multinomial(freq, B, replacement=True, generator=gen)          # which op each example uses
        x = torch.cat([tr[i][torch.randint(len(tr[i]), (int((k == i).sum()),), generator=gen)] for i in range(4)])
        loss = F.cross_entropy(m(x[:, :3]), x[:, 3])
        m.zero_grad(); loss.backward()
        for gr in adam.param_groups:
            gr["lr"] = ADAM_LR * sched
        adam.step()
        with torch.no_grad():
            for w, buf in zip(muon, mom):
                buf.mul_(BETA).add_(w.grad)
                w.mul_(1 - WD * MUON_LR * sched)
                w.add_(shaped(w.grad + BETA * buf, p), alpha=-MUON_LR * sched)
    secs = time.perf_counter() - t0
    end = log[-1]
    print(f"p={p:g} order={order} seed={seed}  " + "  ".join(
        f"{op}({f:.0%}) train {a:.2f} test {b:.2f}" for op, f, a, b in zip(order, FREQ, end["train"], end["test"]))
        + f"   N={N} WD={WD:g} [{secs:.0f} s]", flush=True)
    if os.environ.get("SAVE"):                                  # keep the model and its split for cut.py; no log line
        torch.save(dict(model=m.state_dict(), tr=tr, te=te, p=p, order=order, seed=seed), os.environ["SAVE"])
        return
    with open(os.path.join(HERE, "arith.jsonl"), "a") as f:
        f.write(json.dumps(dict(p=p, order=order, seed=seed, N=N, wd=WD, freq=FREQ, steps=STEPS, log=log)) + "\n")


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
