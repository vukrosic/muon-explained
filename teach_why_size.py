"""images/step0_why.png: what the size of G is, why it is always at least the biggest strength, and its price."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_, B_, C_ = 3.4445, -4.775, 2.0315
f = lambda s: A_ * s + B_ * s**3 + C_ * s**5
INK, MUTE, BIG, WEAK = "#222222", "#888888", "#3a6489", "#b5532f"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14})

t = np.radians(30)
G = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) @ np.diag([1.0, 0.2])
entries2 = (G.ravel() ** 2)
strengths2 = np.linalg.svd(G, compute_uv=False) ** 2
assert np.isclose(entries2.sum(), strengths2.sum())

fig, ax = plt.subplots(1, 3, figsize=(19, 6), dpi=130)

# (a) two ways to add up to the same 1.040
a = ax[0]
for x, parts, labels, cols in [
    (0, entries2, ["0.866²", "0.100²", "0.500²", "0.173²"], ["#c9c2b8", "#b8b0a5", "#d8d1c7", "#a9a196"]),
    (1, strengths2, ["1.00²  (strong)", "0.20²  (weak)"], [BIG, WEAK]),
]:
    bottom = 0
    for p, lab, c in zip(parts, labels, cols):
        a.bar(x, p, bottom=bottom, color=c, width=0.6, edgecolor="white")
        if p > 0.05:
            a.text(x, bottom + p / 2, f"{lab}\n= {p:.3f}", ha="center", va="center", fontsize=12,
                   color="white" if c in (BIG, WEAK) else INK)
        bottom += p
    a.text(x, bottom + 0.03, f"total {bottom:.3f}", ha="center", va="bottom", fontweight="bold")
a.text(0.5, 1.16, "thin slices: 0.100² = 0.010 and 0.173² = 0.030 on the left,\nthe weak 0.20² = 0.040 (red) on the right", ha="center", fontsize=11, color=MUTE)
a.set_xticks([0, 1], ["the 4 entries of G,\nsquared", "the 2 strengths of G,\nsquared"])
a.set_ylim(0, 1.3); a.set_yticks([])
for s in ["top", "right", "left"]: a.spines[s].set_visible(False)
a.set_title("Same total: size² = sum of entries² = sum of strengths²")

# (b) biggest strength after dividing by the size
b = ax[1]
cases = [("our 2×2\n(1, 0.2)", np.array([1.0, 0.2])),
         ("8×8 from section 5\n(1, 0.6, … 0.01)", np.array([1, .6, .3, .15, .08, .04, .02, .01])),
         ("768×768,\nall 768 strengths 1", np.ones(768))]
vals = [1 / np.sqrt((s ** 2).sum()) for _, s in cases]
bars = b.bar([c[0] for c in cases], vals, color=BIG, width=0.6)
for bar, v, (_, s) in zip(bars, vals, cases):
    b.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}\n(size {np.sqrt((s**2).sum()):.3g})",
           ha="center", va="bottom", fontweight="bold")
b.axhline(1, ls="--", color=MUTE, lw=1)
b.text(2.45, 1.02, "1", color=MUTE, ha="right")
b.set_ylim(0, 1.3); b.set_yticks([])
for s in ["top", "right", "left"]: b.spines[s].set_visible(False)
b.set_title("Biggest strength after ÷ size: never above 1,\nbut far below 1 when many strengths are big")

# (c) the 768 case: the quintic lifts 0.036 in 3 rounds
c = ax[2]
x = vals[2]; path = [x]
for _ in range(5):
    x = f(x); path.append(x)
c.plot(range(6), path, "o-", color=BIG, lw=2.5, ms=9)
for k, v in enumerate(path):
    below = k == 4
    c.text(k + 0.12, v - 0.09 if below else v + 0.06, f"{v:.3f}", ha="left", fontsize=12, fontweight="bold", color=BIG)
c.axhline(1, ls="--", color=MUTE, lw=1)
c.set_xticks(range(6)); c.set_xlim(-0.3, 5.6); c.set_xlabel("round"); c.set_ylabel("strength")
c.set_ylim(0, 1.35)
c.set_title("The price is paid back: the quintic lifts\n0.036 to about 1 in 3 rounds")

fig.suptitle("Step 0, the reasons. The size uses only the entries (cheap), yet it equals √(s₁² + s₂² + …), so it is never smaller than the biggest strength.",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig("images/step0_why.png"); plt.close(fig)
print("biggest after ÷ size:", [round(v, 3) for v in vals], "768 path:", [round(v, 3) for v in path])
