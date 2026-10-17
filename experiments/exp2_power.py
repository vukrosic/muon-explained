"""Experiment 2: does under-pushing the weak directions of an update cost learning?

Exact SVD of each Muon update G = U·S·Vᵀ, then every strength s -> s^p, rescaled so the update's total
size (Frobenius norm) equals that of U·Vᵀ. p = 0 is exact flattening (all strengths 1); p = 1 keeps the
update's own shape (weak directions barely pushed). Only p changes. The learning rate is a check:
if the order of p is the same at every learning rate, p decides the result, not the learning rate.
Usage: python3 exp2_power.py <Muon lr> [p values, e.g. 0,0.25] [seed, default 0]
"""
import json, os, sys, time
import torch
import train as T

P = [0, 0.25, 0.5, 1]


def power(p):
    def f(G):
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        w = S ** p
        w = w * (len(S) ** 0.5) / w.norm()          # same total size as U·Vᵀ for every p
        return U @ torch.diag(w) @ Vh
    return f


if __name__ == "__main__":
    lr = float(sys.argv[1])
    ps = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else P
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    path = os.path.join(T.HERE, "exp2_results.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for p in ps:
        t0 = time.time()
        r = T.run(power(p), seed, lr)
        res[f"p={p:g} lr={lr:g}" + (f" seed={seed}" if seed else "")] = dict(p=p, lr=lr, seed=seed, val=r["val"], curve=r["curve"],
                                         time=time.time() - t0, python=T.PYTHON)
        json.dump(res, open(path, "w"), indent=0)
        print(f"p = {p:<4g}  lr = {lr:<6g} seed {seed}  val {r['val']:.4f}  ({time.time() - t0:.0f} s)", flush=True)
