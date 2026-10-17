"""Chart for cut.jsonl -> ../images/arith_cut.png"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "cut.jsonl"))]
plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(1, 2, figsize=(15, 6.5), dpi=130)
for a, weakest in zip(ax, (True, False)):
    for order, ls in (("+-*/", "-"), ("/*-+", "--")):
        r = [x for x in rows if x["order"] == order and x["weakest"] == weakest]
        f = [x["frac"] * 100 for x in r]
        a.plot(f, [x["train"][0] for x in r], ls, color="#3a6489", lw=2.5, label=f"common op, 55% ({order[0]})")
        a.plot(f, [x["train"][1] for x in r], ls, color="#7fa3c4", lw=2.5, label=f"common op, 30% ({order[1]})")
        a.plot(f, [x["train"][2] for x in r], ls, color="#b5532f", lw=3, label=f"rare op, 12% ({order[2]})")
    a.set_ylim(-0.03, 1.03); a.set_ylabel("accuracy on its training pairs")
ax[0].set_title("Delete the WEAKEST directions\nthe rare op breaks first, from the middle directions")
ax[0].set_xlabel("% of weakest directions deleted (in all 3 Muon matrices)")
ax[0].axvspan(0, 30, color="#dddddd", alpha=0.5); ax[0].text(15, 0.5, "bottom 30%:\nnothing\nbreaks", ha="center")
ax[1].set_title("Delete the STRONGEST directions\nthe common ops break a bit sooner")
ax[1].set_xlabel("% of strongest directions deleted")
ax[1].legend(fontsize=10, loc="upper right")
fig.text(0.5, 0.005, "Muon (p = 0), a op b mod 97. Solid: + - x / at 55/30/12/3%.  Dashed: order reversed.", ha="center", fontsize=11)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(os.path.join(HERE, "..", "images", "arith_cut.png"))
