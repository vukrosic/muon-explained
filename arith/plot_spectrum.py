"""Rare op's gradient in the batch gradient's spectrum (layer h1, Muon p = 0, rare op at 4%).
Left: share of its energy in the strongest 10% of directions. Right: descent it gets from exact Muon (p = 0)
divided by descent from plain gradient direction (p = 1)."""
import json, os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
runs = [json.loads(l) for l in open(os.path.join(HERE, "spectrum.jsonl"))]
names = {"/*-+": "+", "+-*/": "÷"}
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3))
for d in runs:
    rows = [r for r in d["rows"] if r["layer"] == "h1"]
    st = [r["step"] for r in rows]
    partner = d["freq"][2] > 0
    lab = f"rare {names[d['order']]}, " + ("partner in data" if partner else "no partner")
    ls, c = ("-" if partner else "--"), ("C0" if d["order"] == "+-*/" else "C3")
    a.plot(st, [r["top10_3"] for r in rows], "o" + ls, c=c, label=lab)
    b.plot(st, [r["rel_p0"][3] / r["rel_p1"][3] for r in rows], "o" + ls, c=c, label=lab)
common = [r["top10_0"] for r in runs[0]["rows"] if r["layer"] == "h1"]
a.plot(st, common, "s:", c="gray", label="common op (for comparison)")
a.set(xlabel="step", ylabel="share of gradient energy", title="Rare op's gradient: share in the strongest 10% of directions", ylim=(0, 1))
b.axhline(1, c="gray", lw=0.8)
b.set(xlabel="step", ylabel="exact Muon ÷ plain direction", title="Descent on the rare op: exact Muon vs plain gradient direction", yscale="log")
for ax in (a, b): ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "images", "arith_spectrum.png"), dpi=130)
