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
DECAY = 0.5                                              # share of steps at the end over which lr falls to 0
ABC = (3.4445, -4.775, 2.0315)
COEFS = None                                             # optional list of (a, b, c), one per round
SPEC = 0                                                 # 0: divide by total size; k > 0: by the largest strength, k power steps
SAFE = 1.05                                              # with SPEC, divide by SAFE x the estimate (an underestimate would blow up)


def flatten(G, v=None):
    """Muon: divide by the size, then ROUNDS rounds of the quintic curve.
    v: this matrix's power-iteration vector, kept between steps (used when SPEC > 0)."""
    if SPEC:
        for _ in range(SPEC):
            u = G @ v
            v.copy_(G.T @ u); v.div_(v.norm() + 1e-12)
        X = G / (SAFE * (G @ v).norm() + 1e-7)
    else:
        X = G / (G.norm() + 1e-7)
    for a, b, c in COEFS or [ABC] * ROUNDS:
        A = X @ X.T
        X = a * X + ((b * A + c * A @ A) if c else b * A) @ X   # c = 0: cubic round, one multiply fewer
    return X


def optimize(m, g):
    muon_ps = [m.inp.weight, m.h1.weight, m.h2.weight]
    adam = torch.optim.AdamW([m.emb.weight, m.out.weight, m.out.bias], lr=ADAM_LR, weight_decay=0)
    mom = [torch.zeros_like(p) for p in muon_ps]
    vecs = [torch.ones(p.shape[1]) / p.shape[1] ** 0.5 for p in muon_ps]
    for step in range(STEPS):
        sched = min(1.0, (1 - step / STEPS) / DECAY)     # flat, then linear decay over the last DECAY of steps
        x, y = batch(train, g)
        loss = F.cross_entropy(m(x), y)
        m.zero_grad(); loss.backward()
        for gr in adam.param_groups:
            gr["lr"] = ADAM_LR * sched
        adam.step()
        with torch.no_grad():
            for p, buf, v in zip(muon_ps, mom, vecs):
                buf.mul_(BETA).add_(p.grad)
                p.add_(flatten(p.grad + BETA * buf, v), alpha=-MUON_LR * sched)   # Nesterov momentum
# --------------------------------------------------------------------- end of OPTIMIZER ----


def full_val(m):
    """Mean next-character loss over every position of the validation text (fixed, do not edit)."""
    idx, tot = torch.arange(len(val) - CTX), 0.0
    with torch.no_grad():
        for i in range(0, len(idx), 16384):
            ix = idx[i:i + 16384]
            tot += F.cross_entropy(m(val[ix[:, None] + torch.arange(CTX)]), val[ix + CTX], reduction="sum").item()
    return tot / len(idx)


def main(desc, settings=""):
    torch.manual_seed(SEED)
    m = Net()
    g = torch.Generator().manual_seed(1000 + SEED)
    t0 = time.perf_counter()
    optimize(m, g)
    secs = time.perf_counter() - t0
    v = full_val(m)
    hit = v <= TARGET
    print(f"{desc:<40} train {secs:5.1f} s   val {v:.4f}   {'REACHED' if hit else 'missed'}", flush=True)
    rec = dict(desc=desc, settings=settings, seconds=round(secs, 2), val=round(v, 5), reached=hit, seed=SEED,
               python=PYTHON, torch=torch.__version__, when=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(HERE, "runs.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return secs, v


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
