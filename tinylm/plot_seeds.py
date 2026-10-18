"""Plot AdamW phase-2 learning curves per seed from adam_seeds.jsonl -> adam_seeds.png"""
import json, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src = sys.argv[1] if len(sys.argv) > 1 else "adam_seeds.jsonl"
runs = sorted((json.loads(l) for l in open(src)), key=lambda r: r["seed"])
first = lambda r, k, th: next((e["step"] for e in r["log"] if e[k] >= th), None)
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
cm = plt.get_cmap("tab20")
for i, r in enumerate(runs):
    st = [e["step"] for e in r["log"]]
    c, lab = cm(i), f"seed {r['seed']} ({first(r, 'R', 0.5)})"
    ax[0].plot(st, [e["R"] for e in r["log"]], color=c, label=lab)
    ax[1].plot(st[1:], [e["loss"] for e in r["log"][1:]], color=c)
    ax[2].plot(st, [sum(e["Rpos"]) / 6 for e in r["log"]], color=c)
ax[0].set(title="Reverse: whole answer right", xlabel="phase-2 step", ylabel="exact match")
ax[1].set(title="Training loss (all tasks)", xlabel="phase-2 step", yscale="log")
ax[2].set(title="Reverse: share of letters right", xlabel="phase-2 step", ylim=(0, 1.02))
ax[2].axhline(0.1, color="grey", ls=":", lw=1); ax[2].text(50, 0.12, "chance 0.1", color="grey")
ax[0].legend(title="seed (first step R ≥ 0.5)", fontsize=8)
for a in ax: a.grid(alpha=0.3)
fig.suptitle("AdamW learns a new task (reverse) after copy + shift — 12 seeds, same settings")
fig.tight_layout(); fig.savefig(src.replace(".jsonl", ".png"), dpi=110)
for r in runs:
    print(r["seed"], "R>=0.5 at", first(r, "R", 0.5), " R>=0.99 at", first(r, "R", 0.99),
          " letters>=0.3 at", next((e["step"] for e in r["log"] if sum(e["Rpos"]) / 6 >= 0.3), None))
