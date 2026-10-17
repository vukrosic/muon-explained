"""Chart for experiment 2 -> ../images/exp2_power.png"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
res = list(json.load(open(os.path.join(HERE, "exp2_results.json"))).values())
P = [0, 0.25, 0.5, 1]
LRS = sorted({r["lr"] for r in res})
cmap = plt.get_cmap("viridis")
fig, ax = plt.subplots(1, 2, figsize=(15, 6), dpi=130)

a = ax[0]
for i, lr in enumerate(LRS):
    pts = sorted((r["p"], r["val"]) for r in res if r["lr"] == lr and r.get("seed", 0) == 0)
    a.plot(*zip(*pts), "o-", color=cmap(i / (len(LRS) - 1)), lw=2, ms=7, label=f"lr {lr:g}")
a.set_xlabel("p   (0 = every direction pushed equally, 1 = weak directions barely pushed)")
a.set_ylabel("final validation loss (lower = better)")
a.set_title("Seed 0, every learning rate tried\np = 1 is worst at every learning rate")
a.legend(fontsize=10)

b = ax[1]
p_line = np.linspace(0, 1, 101)
best = {}
for p in P:
    s0 = min((r for r in res if r["p"] == p and r.get("seed", 0) == 0), key=lambda r: r["val"])
    s1 = [r for r in res if r["p"] == p and r.get("seed", 0) == 1 and r["lr"] == s0["lr"]][0]
    best[p] = (s0["val"], s1["val"], s0["lr"])
for p, (v0, v1, lr) in best.items():
    b.plot([p, p], [v0, v1], color="#888888", lw=1.5)
    b.plot(p, v0, "o", color="#3a6489", ms=9)
    b.plot(p, v1, "s", color="#b5532f", ms=9)
    b.plot(p, (v0 + v1) / 2, "_", color="black", ms=26, mew=2.5)
    b.text(p + 0.03, (v0 + v1) / 2, f"mean {(v0 + v1) / 2:.4f}\n(lr {lr:g})", va="center", fontsize=10)
pred0 = best[0][0]
b.plot(p_line, pred0 + 0.004 * p_line + 0.085 * p_line**3, ":", color="#888888", label="my prediction before running")
b.plot([], [], "o", color="#3a6489", label="seed 0"); b.plot([], [], "s", color="#b5532f", label="seed 1")
b.plot([], [], "_", color="black", ms=14, mew=2.5, label="mean of the 2 seeds")
b.set_xlim(-0.08, 1.25)
b.set_xlabel("p"); b.set_ylabel("final validation loss at each p's best learning rate")
b.set_title("Best learning rate per p, 2 seeds\np = 0, 0.25, 0.5 tie (gaps smaller than seed noise); p = 1 is worse")
b.legend(loc="upper left", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "images", "exp2_power.png"))
for p, (v0, v1, lr) in best.items():
    print(p, lr, v0, v1, round((v0 + v1) / 2, 4))
