"""Charts for rare_push.jsonl -> ../images/rare_push1.png, rare_push2.png"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "..", "images")
rows = [json.loads(l) for l in open(os.path.join(HERE, "rare_push.jsonl"))]
P, SEEDS = [0.0, 0.25, 0.5], [0, 1, 2]
val = {(r["p"], r["seed"]): r["val"] for r in rows}
REC3 = {0: 1.6297, 1: 1.6201, 2: 1.6307}                       # record 3, 2 Newton-Schulz rounds, runs.jsonl
marks = {0: "o", 1: "s", 2: "^"}
plt.rcParams.update({"font.size": 14})

fig, a = plt.subplots(figsize=(8, 8), dpi=140)
for s in SEEDS:
    a.plot(P, [val[p, s] for p in P], marks[s], color="#aaaaaa", ms=9, label=f"seed {s}")
mean = [np.mean([val[p, s] for s in SEEDS]) for p in P]
a.plot(P, mean, "-", color="black", lw=3, label="mean of 3 seeds")
r3 = np.mean(list(REC3.values()))
a.axhline(r3, ls="--", color="#b5532f", lw=2, label=f"record 3, 2 rounds (mean {r3:.4f})")
for p, m in zip(P, mean):
    a.text(p + (0.03 if p == 0 else 0), m + (0.0006 if p == 0 else -0.0016), f"{m:.4f}", ha="left" if p == 0 else "center", va="bottom" if p == 0 else "top", fontsize=13, fontweight="bold")
a.annotate("Muon: every\nsingular value\nset to 1", xy=(0, mean[0]), xytext=(0.05, 1.6385),
           arrowprops=dict(arrowstyle="->"))
a.set_xticks(P)
a.set_xlabel("p   (singular values S become S^p)")
a.set_ylabel("final validation loss (lower = better)")
a.set_title("1. Loss vs p, speedrun record 3 settings")
a.set_ylim(1.611, 1.646)
a.legend(loc="upper right", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "rare_push1.png"))

fig, a = plt.subplots(figsize=(8, 8), dpi=140)
w = 0.25
for i, p in enumerate([0.25, 0.5]):
    gaps = [val[p, s] - val[0.0, s] for s in SEEDS]
    x = np.arange(3) + (i - 0.5) * w
    a.bar(x, gaps, w, color=["#3a6489", "#b5532f"][i], label=f"p = {p:g}")
    for xi, g in zip(x, gaps):
        a.text(xi, g - 0.0004, f"{g:+.4f}", ha="center", va="top", fontsize=11)
a.axhline(0, color="black", lw=1)
a.set_xticks(range(3)); a.set_xticklabels([f"seed {s}" for s in SEEDS])
a.set_ylim(-0.0175, 0.002)
a.set_ylabel("loss change vs p = 0 (below 0 = better than Muon)")
a.set_title("2. Better than equal singular values on every seed")
a.legend(loc="lower right", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "rare_push2.png"))
print([round(m, 4) for m in mean], r3)
