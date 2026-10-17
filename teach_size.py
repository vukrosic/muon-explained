"""images/step0_size.png: what 'divide G by its size' means and why Muon does it."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_, B_, C_ = 3.4445, -4.775, 2.0315
f = lambda s: A_ * s + B_ * s**3 + C_ * s**5
INK, MUTE, GOOD, BAD = "#222222", "#888888", "#3a6489", "#b5532f"
plt.rcParams.update({"font.size": 14})

t = np.radians(30)
U = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
G = U @ np.diag([1.0, 0.2])
size = np.linalg.norm(G)  # Frobenius norm: sqrt of the sum of every entry squared
X = G / size

fig, ax = plt.subplots(1, 2, figsize=(16, 6.6), dpi=130, gridspec_kw={"width_ratios": [1.15, 1]})

a = ax[0]; a.axis("off"); a.set_xlim(0, 10); a.set_ylim(0, 10)


def grid(x0, y0, M, title, col):
    a.text(x0 + 1.5, y0 + 3.3, title, ha="center", fontsize=15, fontweight="bold", color=col)
    for r in range(2):
        for c in range(2):
            a.add_patch(plt.Rectangle((x0 + 1.5 * c, y0 + 1.5 * (1 - r)), 1.5, 1.5, fill=False, lw=1.5, color=col))
            a.text(x0 + 1.5 * c + 0.75, y0 + 1.5 * (1 - r) + 0.75, f"{M[r, c]:.3f}", ha="center", va="center", fontsize=15)


grid(0.3, 5.6, G, "G (the update)", INK)
grid(6.4, 5.6, X, "X = G ÷ 1.020", GOOD)
a.annotate("", xy=(6.2, 7.1), xytext=(3.6, 7.1), arrowprops=dict(arrowstyle="-|>", lw=2.5, color=INK))
a.text(4.9, 7.45, "÷ 1.020", ha="center", fontsize=15)
sq = G.ravel() ** 2
a.text(0.3, 4.4, "Size of G: square every entry, add them, take the square root", fontsize=14)
a.text(0.3, 3.6, "  0.866² + 0.100² + 0.500² + 0.173²", fontsize=14, family="monospace")
a.text(0.3, 2.9, f"= {sq[0]:.3f} + {sq[1]:.3f} + {sq[2]:.3f} + {sq[3]:.3f} = {sq.sum():.3f}", fontsize=14, family="monospace")
a.text(0.3, 2.2, f"√{sq.sum():.3f} = {size:.3f}", fontsize=14, family="monospace")
a.text(0.3, 1.0, f"Every entry ÷ {size:.3f}, so every strength ÷ {size:.3f}:", fontsize=14)
a.text(0.3, 0.3, f"1.00 → {1 / size:.3f}      0.20 → {0.2 / size:.3f}", fontsize=15, fontweight="bold", color=GOOD)

b = ax[1]
x = np.linspace(0, 1.6, 400)
b.axvspan(0, 1.0, color="#dfe8f1")
b.plot(x, f(x), color=INK, lw=2.5)
b.axhline(1, ls="--", color=MUTE, lw=1)
b.text(0.5, 4.4, "after ÷ size,\nevery strength is in here", ha="center", color=GOOD, fontsize=13, fontweight="bold")
for s in [1.3, 1.5]:
    b.plot(s, f(s), "o", color=BAD, ms=9)
    b.text(s - 0.03, f(s) + 0.15, f"f({s}) = {f(s):.2f}", color=BAD, ha="right", fontsize=13, fontweight="bold")
b.text(1.58, 0.25, f"f(2) = {f(2):.0f}\nf(3) = {f(3):.0f}", color=BAD, ha="right", fontsize=13, fontweight="bold")
b.set_xlim(0, 1.6); b.set_ylim(0, 5)
b.set_xlabel("strength going in"); b.set_ylabel("strength coming out")
b.set_title("Why: the curve only pulls toward 1 for small inputs")
fig.suptitle("Step 0. Before the rounds, Muon shrinks G so that its biggest strength is at most 1.\n"
             "A real gradient can be any size (strengths 0.001 or 40); without this, f blows the numbers up.", fontsize=15)
fig.tight_layout(); fig.savefig("images/step0_size.png"); plt.close(fig)
print(np.round(X, 3).tolist(), round(size, 4), f(1.3), f(1.5), f(2), f(3))
