"""Delete the weakest (or strongest) singular directions of the trained Muon matrices; which operations break?
Usage: python3 cut.py models/p0_A_s0.pt  -> lines in cut.jsonl"""
import json, os, sys
import torch
from arith import Net, acc, FREQ

HERE = os.path.dirname(os.path.abspath(__file__))
d = torch.load(sys.argv[1])
m = Net(); m.load_state_dict(d["model"])
W0 = {k: m.state_dict()[k].clone() for k in ("inp.weight", "h1.weight", "h2.weight")}
SVD = {k: torch.linalg.svd(W.double(), full_matrices=False) for k, W in W0.items()}
for k, (U, s, Vh) in SVD.items():
    q = [round(s[int(i * (len(s) - 1))].item(), 2) for i in (0, .1, .5, .9, 1)]
    print(f"{k:>10} strengths: largest {q[0]}, 10% {q[1]}, middle {q[2]}, 90% {q[3]}, smallest {q[4]}")


def cut(frac, weakest):
    """Zero a share of each matrix's directions: the weakest ones or the strongest ones."""
    sd = m.state_dict()
    for k, (U, s, Vh) in SVD.items():
        keep = torch.ones_like(s)
        n = int(round(frac * len(s)))
        if n:
            keep[-n:] = 0 if weakest else 1
            if not weakest:
                keep[:n] = 0
        sd[k].copy_(((U * (s * keep)) @ Vh).float())


out = open(os.path.join(HERE, "cut.jsonl"), "a")
print(f"{'cut':>16} " + " ".join(f"{op}({f:.0%}) tr/te".rjust(17) for op, f in zip(d["order"], FREQ)))
for weakest, fracs in ((True, (0, .1, .2, .25, .3, .35, .4, .45, .5, .6, .75, .9)), (False, (0, .01, .02, .05, .1))):
    for frac in fracs:
        cut(frac, weakest)
        tr = [acc(m, r) for r in d["tr"]]; te = [acc(m, r) for r in d["te"]]
        name = f"{'weakest' if weakest else 'strongest'} {frac:.0%}"
        print(f"{name:>16} " + " ".join(f"{a:7.2f} /{b:5.2f}".rjust(17) for a, b in zip(tr, te)))
        out.write(json.dumps(dict(model=os.path.basename(sys.argv[1]), order=d["order"], weakest=weakest, frac=frac,
                                  train=[round(a, 4) for a in tr], test=[round(b, 4) for b in te])) + "\n")
