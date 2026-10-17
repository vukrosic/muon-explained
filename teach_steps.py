"""images/move0.png … move4.png: a plain update vs Muon's update, step by step.

move0: the two updates as shapes. Same total size √(s₁² + s₂²) = 1, different split.
move1–4: where the weights are after each step, measured along the blue (strong) and red (weak) direction.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIG, WEAK, INK, MUTE = "#3a6489", "#b5532f", "#222222", "#888888"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15})

plain = np.array([1.0, 0.05])
plain = plain / np.linalg.norm(plain)          # strengths 0.999, 0.050: total size 1
muon = np.array([1.0, 1.0]) / np.sqrt(2)       # strengths 0.707, 0.707: total size 1
UPD = [(plain, "Plain update"), (muon, "Muon's update")]

# ---------- move0: the two updates ----------
th = np.linspace(0, 2 * np.pi, 300)
fig, ax = plt.subplots(1, 2, figsize=(13, 6), dpi=130)
for a, (s, name) in zip(ax, UPD):
    a.fill(s[0] * np.cos(th), s[1] * np.sin(th), color="#e9dfd0")
    a.plot(s[0] * np.cos(th), s[1] * np.sin(th), color=INK, lw=2)
    a.annotate("", xy=(s[0], 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BIG, lw=3))
    a.annotate("", xy=(0, s[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=WEAK, lw=3))
    a.text(s[0] / 2, -0.1, f"blue {s[0]:.2f}", color=BIG, ha="center", va="top", fontweight="bold", fontsize=14)
    a.text(0.05, s[1] + 0.06, f"red {s[1]:.2f}", color=WEAK, ha="left", va="bottom", fontweight="bold", fontsize=14)
    share = s ** 2 / (s ** 2).sum()
    a.text(0, -0.95, f"total size √({s[0]:.2f}² + {s[1]:.2f}²) = {np.linalg.norm(s):.2f}\n"
                     f"blue gets {share[0]:.1%} of the step, red gets {share[1]:.1%}",
           ha="center", va="center", fontsize=13)
    a.set_xlim(-1.15, 1.15); a.set_ylim(-1.15, 1.0); a.set_aspect("equal"); a.axis("off")
    a.set_title(name)
fig.suptitle("The two updates. Both have the same total size, 1. The plain one puts almost all of it on blue.", fontsize=15)
fig.tight_layout(); fig.savefig("images/move0.png"); plt.close(fig)

# ---------- move1–4: the weights after each step ----------
for k in range(1, 5):
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2), dpi=130)
    for a, (s, name) in zip(ax, UPD):
        pos = np.array([[0.0, 0.0]] + [s * i for i in range(1, k + 1)])
        for i, (p0, p1) in enumerate(zip(pos[:-1], pos[1:])):
            new = i == k - 1
            a.annotate("", xy=p1, xytext=p0,
                       arrowprops=dict(arrowstyle="-|>", color=INK if new else "#aaaaaa", lw=2.5 if new else 1.5))
        a.plot(pos[:-1, 0], pos[:-1, 1], "o", color="#aaaaaa", ms=6)
        a.plot(*pos[-1], "o", color=INK, ms=9, zorder=5)
        end = pos[-1]
        a.plot([0, end[0]], [-0.15, -0.15], color=BIG, lw=6, solid_capstyle="butt")
        a.plot([-0.15, -0.15], [0, end[1]], color=WEAK, lw=6, solid_capstyle="butt")
        a.text(end[0] / 2, -0.3, f"{end[0]:.2f}", color=BIG, ha="center", va="top", fontweight="bold", fontsize=15)
        a.text(-0.3, max(end[1] / 2, 0.08), f"{end[1]:.2f}", color=WEAK, ha="right", va="center", fontweight="bold", fontsize=15)
        a.set_xlim(-0.9, 4.3); a.set_ylim(-0.75, 3.2); a.set_aspect("equal")
        a.set_xticks(range(5)); a.set_yticks(range(4)); a.tick_params(labelsize=11, colors=MUTE)
        a.grid(color="#eeeeee"); a.set_axisbelow(True)
        for sp in a.spines.values(): sp.set_color("#dddddd")
        a.set_xlabel("distance moved along blue (strong)", color=BIG)
        a.set_ylabel("distance moved along red (weak)", color=WEAK)
        a.set_title(f"{name}: blue {end[0]:.2f}, red {end[1]:.2f}")
    p, m = plain * k, muon * k
    note = {1: "Each step moves the weights by the update. Dark arrow = this step, gray = earlier steps.",
            2: "The plain update adds another 0.05 to red. Muon's adds another 0.71.",
            3: f"Red so far: {p[1]:.2f} with the plain update, {m[1]:.2f} with Muon's.",
            4: f"After 4 steps red has moved {m[1] / p[1]:.0f}× further with Muon. Blue has moved {p[0] / m[0]:.1f}× less."}[k]
    fig.suptitle(f"Step {k}. {note}", fontsize=15)
    fig.tight_layout(); fig.savefig(f"images/move{k}.png"); plt.close(fig)
print("plain", np.round(plain, 3), "muon", np.round(muon, 3), "after 4 red ratio", round(muon[1] / plain[1], 1))
