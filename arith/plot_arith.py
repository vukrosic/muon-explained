"""Chart for arith.jsonl (weight decay 0.05 runs) -> ../images/arith_rare.png"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
runs = [r for r in (json.loads(l) for l in open(os.path.join(HERE, "arith.jsonl"))) if r["wd"] == 0.05]
COL = {0.0: "#3a6489", 0.5: "#b5532f", 1.0: "#999999"}
NAME = {0.0: "p = 0 (Muon, equal push)", 0.5: "p = 0.5 (weak directions pushed less)", 1.0: "p = 1 (pushed as the gradient says)"}
plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(1, 2, figsize=(15, 6.5), dpi=130)
seen = set()
for r in runs:
    steps = [e["step"] for e in r["log"]]
    lab = NAME[r["p"]] if r["p"] not in seen else None; seen.add(r["p"])
    ls = "-" if r["seed"] == 0 else "--"
    ax[0].plot(steps, [e["train"][2] for e in r["log"]], ls, color=COL[r["p"]], lw=2, label=lab)
    ax[1].plot(steps, [(e["test"][0] + e["test"][1]) / 2 for e in r["log"]], ls, color=COL[r["p"]], lw=2)
ax[0].set_title("Rare operation (12% of examples)\nits training pairs answered right")
ax[0].set_xlabel("training step"); ax[0].set_ylabel("accuracy on its training pairs")
ax[0].legend(loc="center right", fontsize=11)
ax[1].set_title("The 2 common operations (55%, 30%)\nunseen pairs answered right")
ax[1].set_xlabel("training step"); ax[1].set_ylabel("accuracy on held-out pairs")
for a in ax:
    a.set_ylim(-0.03, 1.03)
fig.text(0.5, 0.005, "a op b mod 97, ops + - x /. Each line: one run (2 rarity orders x seeds; solid seed 0, dashed seed 1).",
         ha="center", fontsize=11)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(HERE, "..", "images", "arith_rare.png"))
