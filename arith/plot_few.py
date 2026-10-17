"""Left: rare op held-out vs number of distinct training pairs, partner known (Adam 9000 steps, Muon 3000).
Right: lone rare op (no partner) held-out vs its share of the batch."""
import json, os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
load = lambda f: [json.loads(l) for l in open(os.path.join(HERE, f))]
few, lone = load("few.jsonl"), load("lone.jsonl")
names = {"/*-+": "+ (partner −)", "+-*/": "÷ (partner ×)"}
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3))
for order, ls in (("/*-+", "-"), ("+-*/", "--")):
    for opt, steps, c in (("adam", 9000, "C2"), ("muon", 3000, "C0")):
        pts = sorted((d["rare_pairs"], d["log"][-1]["test"][3]) for d in few
                     if d["order"] == order and d["opt"] == opt and d["steps"] == steps and d["freq"][2] > 0)
        a.plot(*zip(*pts), "o" + ls, c=c, label=f"{names[order]}, {'AdamW' if opt == 'adam' else 'Muon'} {steps} steps")
    pts = sorted((d["rare_pairs"], d["log"][-1]["test"][3]) for d in few if d["order"] == order and d["freq"][2] == 0)
    a.plot(*zip(*pts), "x" + ls, c="C3", ms=8, label=f"{names[order]}, partner removed, AdamW 9000")
    for opt, steps, c in (("adam", 3000, "C8"), ("adam", 9000, "C2"), ("muon", 3000, "C9"), ("muon", 9000, "C0")):
        pts = sorted((d["freq"][3] * 100, d["log"][-1]["test"][3]) for d in lone + few
                     if d["order"] == order and d["opt"] == opt and d["steps"] == steps and d["freq"][2] == 0
                     and d.get("rare_pairs", 0) in (0, 4704))
        b.plot(*zip(*pts), "o" + ls, c=c, label=f"{names[order][:1]} {'AdamW' if opt == 'adam' else 'Muon'} {steps}")
a.set_xscale("log"); a.axvline(4704, c="gray", lw=.8, ls=":")
a.set_xlabel("distinct training pairs of the rare op (4704 = all)"); a.set_ylabel("rare op, held-out accuracy")
a.set_title("Partner known: rare op (4% of batch) needs 700–1500 distinct pairs\n"
            "fewer are memorized (train 100%) but not generalized", fontsize=9.5)
a.legend(fontsize=7)
b.set_xlabel("% of batch that is the lone op (no partner, all 4704 pairs)")
b.set_title("No partner: the op must be 16–33% of the batch\n(with a partner, 2–4% was enough)", fontsize=9.5)
b.set_xticks([4, 8, 16, 25, 33]); b.legend(fontsize=7, ncol=2)
for ax in (a, b):
    ax.axhline(1 / 97, c="gray", lw=.8, ls=":")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "..", "images", "arith_few.png"), dpi=130)
