"""Chart for the 'can a trained model learn a new op?' section: critical.jsonl (Adam, x enters at step d)."""
import json, os, collections, statistics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = collections.defaultdict(list)
for l in open(os.path.join(HERE, "critical.jsonl")):
    r = json.loads(l); L = r["log"]; d = int(r["add_at"].split(":")[1])
    R[d].append((L[-1]["test"][2], min(L, key=lambda x: abs(x["step"] - d))["erank"]))
ds = sorted(R)
fig, ax = plt.subplots(figsize=(8, 5))
for d in ds:
    ax.scatter([d] * len(R[d]), [t for t, _ in R[d]], color="#1f77b4", alpha=0.6, s=28)
ax.plot(ds, [statistics.median(t for t, _ in R[d]) for d in ds], color="#1f77b4", lw=2, label="× held-out at the end (dots: 8 runs, line: median)")
ax.axhline(0.03, color="#d62728", ls="--", lw=1.5, label="model trained 9000 steps on + and − first (8 runs, all 0.02–0.03)")
ax.set_xlabel("step at which × enters training (+ and − train from step 0)")
ax.set_ylabel("× held-out accuracy after 9000 steps")
ax.set_ylim(0, 1.02)
ax2 = ax.twinx()
ax2.plot(ds, [statistics.median(e for _, e in R[d]) for d in ds], color="#2ca02c", lw=1.5, ls=":", label="layer h1 effective rank when × enters")
ax2.set_ylabel("h1 effective rank", color="#2ca02c"); ax2.set_ylim(0, 220)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, frameon=False)
ax.set_title("The later × arrives, the less often it is learned (Adam)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "images", "arith_critical.png"), dpi=130)
