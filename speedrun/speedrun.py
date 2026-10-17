"""Laptop speedrun: reach the target validation loss in the least training time.

Fixed: data, model, batch size 512, seed 0, 8 CPU threads, and how validation loss is measured.
Free: everything in the OPTIMIZER section (which optimizer, rounds, learning rates, momentum,
schedule, number of steps). Rules and records: README.md.
Usage: python3 speedrun.py "short description of the change"
"""
import json, os, sys, time, datetime
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from train import Net, batch, train, val, CTX, PYTHON   # same data and model as the experiments

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = 1.68
SEED = 0


# ---------------------------------------------------------------- OPTIMIZER (edit freely) ----
STEPS = 1000
MUON_LR, ADAM_LR, BETA = 0.01, 3e-3, 0.95
ROUNDS = 3
ABC = (3.4445, -4.775, 2.0315)


def flatten(G):
    """Muon: divide by the size, then ROUNDS rounds of the quintic curve."""
    X = G / (G.norm() + 1e-7)
    a, b, c = ABC
    for _ in range(ROUNDS):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X


def optimize(m, g):
    muon_ps = [m.inp.weight, m.h1.weight, m.h2.weight]
    adam = torch.optim.AdamW([m.emb.weight, m.out.weight, m.out.bias], lr=ADAM_LR, weight_decay=0)
    mom = [torch.zeros_like(p) for p in muon_ps]
    for step in range(STEPS):
        sched = min(1.0, 2 * (1 - step / STEPS))         # flat, then linear decay over the last half
        x, y = batch(train, g)
        loss = F.cross_entropy(m(x), y)
        m.zero_grad(); loss.backward()
        for gr in adam.param_groups:
            gr["lr"] = ADAM_LR * sched
        adam.step()
        with torch.no_grad():
            for p, buf in zip(muon_ps, mom):
                buf.mul_(BETA).add_(p.grad)
                p.add_(flatten(p.grad + BETA * buf), alpha=-MUON_LR * sched)   # Nesterov momentum
# --------------------------------------------------------------------- end of OPTIMIZER ----


def full_val(m):
    """Mean next-character loss over every position of the validation text (fixed, do not edit)."""
    idx, tot = torch.arange(len(val) - CTX), 0.0
    with torch.no_grad():
        for i in range(0, len(idx), 16384):
            ix = idx[i:i + 16384]
            tot += F.cross_entropy(m(val[ix[:, None] + torch.arange(CTX)]), val[ix + CTX], reduction="sum").item()
    return tot / len(idx)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    m = Net()
    g = torch.Generator().manual_seed(1000 + SEED)
    t0 = time.perf_counter()
    optimize(m, g)
    secs = time.perf_counter() - t0
    v = full_val(m)
    hit = v <= TARGET
    print(f"train {secs:.1f} s   val {v:.4f}   target {TARGET}: {'REACHED' if hit else 'missed'}")
    rec = dict(desc=sys.argv[1] if len(sys.argv) > 1 else "", seconds=round(secs, 2), val=round(v, 5),
               reached=hit, steps=STEPS, seed=SEED, python=PYTHON, torch=torch.__version__,
               when=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(HERE, "runs.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
