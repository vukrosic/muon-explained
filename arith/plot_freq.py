"""Rare operation held-out accuracy vs how often it appears, with and without its partner (x for /, - for +)."""
import json, os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "freq.jsonl"))]
part = [json.loads(l) for l in open(os.path.join(HERE, "partner.jsonl"))]
runs = [("muon", 0, "Muon (p = 0)", "C0"), ("muon", 0.5, "Muon p = 0.5", "C1"), ("adam", 0, "AdamW", "C2")]
orders = [("+-*/", "÷ rare, partner ×"), ("/*-+", "+ rare, partner −")]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (order, title) in zip(axs, orders):
    for opt, p, name, c in runs:
        pts = sorted((r["freq"][3] * 100, r["log"][-1]["test"][3], r["log"][-1]["test"][2]) for r in rows
                     if r["opt"] == opt and r["p"] == p and r["order"] == order)
        ax.plot([x for x, _, _ in pts], [y for _, y, _ in pts], "o-", c=c, label=f"{name}, partner in data")
        cut = sorted((r["freq"][3] * 100, r["log"][-1]["test"][3]) for r in part
                     if r["opt"] == opt and r["p"] == p and r["order"] == order)
        ax.plot([x for x, _ in cut], [y for _, y in cut], "x--", c=c, ms=9, label=f"{name}, partner removed")
    ax.axhline(1 / 97, c="gray", lw=0.8, ls=":")
    ax.set_title(f"{title}\n3000 steps, weight decay 1, seed 0", fontsize=10)
    ax.set_xlabel("% of batch that is the rare operation")
    ax.set_xticks([2, 4, 8, 12, 16, 25])
axs[0].set_ylabel("rare operation, held-out accuracy")
axs[1].legend(fontsize=7.5, loc="center right")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "..", "images", "arith_freq.png"), dpi=130)
