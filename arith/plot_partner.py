"""Rare operation held-out accuracy vs its partner's (x for /, - for +), every run with 3 common ops + 1 rare op."""
import json, os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
load = lambda f: [json.loads(l) for l in open(os.path.join(HERE, f))]
rows = load("freq.jsonl") + load("partner.jsonl") + load("adam_long.jsonl")
style = {("muon", 0): ("Muon (p = 0)", "C0"), ("muon", 0.5): ("Muon p = 0.5", "C1"), ("adam", 0): ("AdamW", "C2")}
fig, ax = plt.subplots(figsize=(6.2, 5.2))
seen = set()
for r in rows:
    name, c = style[(r["opt"], r["p"])]
    te = r["log"][-1]["test"]
    gone = r["freq"][2] == 0
    lab = f"{name}{', partner removed' if gone else ''}"
    ax.scatter(te[2], te[3], c=c, marker="x" if gone else "o", s=40 if gone else 30, alpha=.8,
               label=None if lab in seen else lab)
    seen.add(lab)
ax.plot([0, 1], [0, 1], c="gray", lw=0.8, ls=":")
ax.set_xlabel("partner operation (× or −), held-out accuracy")
ax.set_ylabel("rare operation (÷ or +, 2–25% of batch), held-out accuracy")
ax.set_title("A rare operation is learned exactly as well as its partner\n"
             f"{len(rows)} runs, both orders, seed 0", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "..", "images", "arith_partner.png"), dpi=130)
import numpy as np
x = np.array([r["log"][-1]["test"][2] for r in rows]); y = np.array([r["log"][-1]["test"][3] for r in rows])
print("corr", np.corrcoef(x, y)[0, 1].round(3), "n", len(rows), "mean |rare - partner|", np.abs(x - y).mean().round(3))
