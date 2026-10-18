"""Laptop speedrun: the lowest validation loss after a fixed number of training tokens.

Fixed: data, model, batch size 512, 1000 steps (512,000 predictions), seed 0, how validation loss is measured,
and a compute cap: the whole training run may use at most as many matrix-multiply FLOPs as the baseline
(Muon, 5 rounds). Matrix decompositions (SVD, eigh, QR, ...) are not allowed.
Free: everything in the OPTIMIZER section. Rules and records: README.md.
The score does not depend on how fast or busy the computer is: same settings, same loss and FLOPs on every run.
Usage: python3 speedrun.py "short description of the change"
"""
import json, os, sys, time, datetime
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from train import Net, batch, train, val, CTX, PYTHON   # same data and model as the experiments

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0
STEPS = 1000                                             # fixed: 1000 steps x 512 examples
BASELINE_GFLOP = 14646.5                                   # compute cap, measured from the baseline below
BANNED = {"svd", "_svd_helper", "symeig", "eig", "qr", "geqrf", "cholesky", "cholesky_inverse", "lu_solve",
          "triangular_solve", "inverse", "pinverse", "matrix_exp", "lstsq", "matrix_power"}


# ---------------------------------------------------------------- OPTIMIZER (edit freely) ----
MUON_LR, ADAM_LR, BETA = 0.02, 0.036, 0.95
ROUNDS = 2
WD, ADAM_BETAS = 0.05, (0.9, 0.999)                         # Muon weight decay (decoupled), AdamW betas
DECAY = 0.85                                             # share of steps at the end over which lr falls to 0
ABC = (3.4445, -4.775, 2.0315)
COEFS = None                                             # optional list of (a, b, c), one per round
ROW_BETA = 0                                             # > 0: per-neuron normalization (NorMuon), running average of each row's size
SPEC = 0                                                 # 0: divide by total size; k > 0: by the largest strength, k power steps
SAFE = 1.05                                              # with SPEC, divide by SAFE x the estimate (an underestimate would blow up)
EARLY_N, EARLY_MULT = 0, 1.0                             # Muon lr times EARLY_MULT for the first EARLY_N steps
MATCH = 0                                                # > 0: rescale the result to its size after MATCH rounds (shape of ROUNDS rounds, step size of MATCH)


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
    for k, (a, b, c) in enumerate(COEFS or [ABC] * ROUNDS):
        if MATCH and k == MATCH:
            size = X.norm()
        A = X @ X.T
        X = a * X + ((b * A + c * A @ A) if c else b * A) @ X   # c = 0: cubic round, one multiply fewer
    if MATCH and MATCH < ROUNDS:
        X = X * (size / (X.norm() + 1e-12))
    return X


def optimize(m, g):
    muon_ps = [m.inp.weight, m.h1.weight, m.h2.weight]
    adam = torch.optim.AdamW([m.emb.weight, m.out.weight, m.out.bias], lr=ADAM_LR, betas=ADAM_BETAS, weight_decay=0)
    mom = [torch.zeros_like(p) for p in muon_ps]
    vecs = [torch.ones(p.shape[1]) / p.shape[1] ** 0.5 for p in muon_ps]
    rows = [torch.zeros(p.shape[0], 1) for p in muon_ps]
    for step in range(STEPS):
        sched = min(1.0, (1 - step / STEPS) / DECAY)     # flat, then linear decay over the last DECAY of steps
        x, y = batch(train, g)
        loss = F.cross_entropy(m(x), y)
        m.zero_grad(); loss.backward()
        for gr in adam.param_groups:
            gr["lr"] = ADAM_LR * sched
        adam.step()
        with torch.no_grad():
            for p, buf, v, r in zip(muon_ps, mom, vecs, rows):
                buf.mul_(BETA).add_(p.grad)
                p.mul_(1 - WD * MUON_LR * sched)
                O = flatten(p.grad + BETA * buf, v)                            # Nesterov momentum
                if ROW_BETA:                                                    # each neuron (row) divided by its running size
                    r.mul_(ROW_BETA).add_(O.pow(2).mean(1, keepdim=True), alpha=1 - ROW_BETA)
                    size = O.norm()
                    O = O / (r / (1 - ROW_BETA ** (step + 1))).sqrt().add(1e-8)
                    O = O * (size / (O.norm() + 1e-12))                         # same total size as before
                p.add_(O, alpha=-MUON_LR * sched * (EARLY_MULT if step < EARLY_N else 1.0))
# --------------------------------------------------------------------- end of OPTIMIZER ----


def full_val(m):
    """Mean next-character loss over every position of the validation text (fixed, do not edit)."""
    idx, tot = torch.arange(len(val) - CTX), 0.0
    with torch.no_grad():
        for i in range(0, len(idx), 16384):
            ix = idx[i:i + 16384]
            tot += F.cross_entropy(m(val[ix[:, None] + torch.arange(CTX)]), val[ix + CTX], reduction="sum").item()
    return tot / len(idx)


class Guard(TorchDispatchMode):
    """Refuses matrix decompositions; every other op runs as usual."""
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name.split("::")[-1]       # e.g. "mm", "_linalg_svd", "linalg_vector_norm"
        if name.lstrip("_") in BANNED or ("linalg" in name and "norm" not in name):
            raise RuntimeError(f"not allowed in the speedrun: {func.__name__}")
        return func(*args, **(kwargs or {}))


def main(desc, settings=""):
    torch.manual_seed(SEED)
    m = Net()
    g = torch.Generator().manual_seed(1000 + SEED)
    fc = FlopCounterMode(display=False)
    t0 = time.perf_counter()
    with fc, Guard():
        optimize(m, g)
    secs = time.perf_counter() - t0
    gflop = fc.get_total_flops() / 1e9
    v = full_val(m)
    ok = gflop <= BASELINE_GFLOP * 1.0001
    print(f"{desc:<48} val {v:.4f}   {gflop:7.0f} GFLOP ({gflop / BASELINE_GFLOP:.0%} of cap){'' if ok else '  OVER CAP'}"
          f"   [{secs:.1f} s, not scored]", flush=True)
    rec = dict(desc=desc, settings=settings, val=round(v, 5), gflop=round(gflop, 1), within_cap=ok, seconds=round(secs, 2),
               seed=SEED, python=PYTHON, torch=torch.__version__, when=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(HERE, "runs.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return v, gflop


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
