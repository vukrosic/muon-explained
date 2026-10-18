"""Chart for the loss-spike section: spike_probe.json (one failing fresh Adam run, every step) and spike.jsonl / kick.jsonl
(12 fresh runs with and without gradient clipping)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
P = [r for r in json.load(open(os.path.join(HERE, "spike_probe.json"))) if 1400 <= r["step"] <= 1600]
runs = {}
for f in ("kick.jsonl", "spike.jsonl"):
    for l in open(os.path.join(HERE, f)):
        d = json.loads(l)
        if d["model"].startswith("lpfresh") and not d.get("kick") and d["steps2"] == 9000 and d["warm"] == 500 and d.get("b2", 0.999) == 0.999:
            runs.setdefault((d["model"], d["bseed"]), {})[d.get("clip", 0)] = d["log"][-1]["test"][2]
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
a.plot([r["step"] for r in P], [r["loss"] for r in P], color="#444", lw=1.2, label="batch loss (left axis, log scale)")
a.set_yscale("log"); a.set_xlabel("training step"); a.set_ylabel("batch loss")
a2 = a.twinx()
D = [r for r in P if "dead_x" in r]
a2.plot([r["step"] for r in D], [r["dead_x"][1] for r in D], color="#d62728", lw=2, label="h1 units off on every × input")
a2.plot([r["step"] for r in D], [r["dead_plus"][1] for r in D], color="#ff7f0e", lw=2, ls="--", label="h1 units off on every + input")
a2.set_ylim(0, 1); a2.set_ylabel("share of h1 units")
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
a.set_title("One Adam run: loss spikes to 17 at step 1488,\nthen two thirds of h1 switch off for good", fontsize=10)
pairs = sorted(runs.values(), key=lambda v: v[0])
for i, v in enumerate(pairs):
    b.plot([0, 1], [v[0], v[1]], color="#999", lw=0.8)
b.scatter([0] * len(pairs), [v[0] for v in pairs], color="#d62728", zorder=3, label="no clipping")
b.scatter([1] * len(pairs), [v[1] for v in pairs], color="#2ca02c", zorder=3, label="gradient norm clipped at 1")
b.set_xticks([0, 1], ["no clipping", "clip at 1"]); b.set_xlim(-0.4, 1.4); b.set_ylim(0, 1.02)
b.set_ylabel("× held-out accuracy after 9000 steps")
b.set_title("Same 12 runs (6 inits × 2 batch orders)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "images", "arith_spike.png"), dpi=130)
