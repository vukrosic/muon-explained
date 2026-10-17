"""Plot the Muon Newton–Schulz polynomials: one round and five rounds."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

quintic = lambda s: 3.4445 * s - 4.775 * s**3 + 2.0315 * s**5
cubic = lambda s: 1.5 * s - 0.5 * s**3


def rounds(f, s, k):
    for _ in range(k):
        s = f(s)
    return s


s = np.linspace(0, 1.2, 600)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)
for ax, k in zip(axes, [1, 5]):
    ax.plot(s, rounds(quintic, s, k), color="#b5532f", lw=2.5, label="quintic  3.4445s − 4.775s³ + 2.0315s⁵")
    ax.plot(s, rounds(cubic, s, k), color="#3a6489", lw=2.5, label="cubic  1.5s − 0.5s³")
    ax.plot(s, s, ":", color="gray", lw=1.2, label="no change (y = s)")
    ax.axhline(1, ls="--", color="black", lw=1, label="goal (y = 1)")
    ax.set_xlim(0, 1.2); ax.set_ylim(0, 1.3)
    ax.set_xlabel("strength before"); ax.set_ylabel(f"strength after {k} round{'s' if k > 1 else ''}")
    ax.set_title(f"{k} round{'s' if k > 1 else ''}"); ax.grid(alpha=0.3)
axes[0].legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig("images/poly_plot.png")
print("saved images/poly_plot.png")
