"""images/step1b_directions.png: where the strength arrows point, and how many there are."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIG, WEAK, INK, MUTE = "#3a6489", "#b5532f", "#222222", "#888888"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14})


def rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


th = np.linspace(0, 2 * np.pi, 400)
circle = np.stack([np.cos(th), np.sin(th)])

mats = [
    (np.diag([1.0, 0.2]), "G = [[1, 0], [0, 0.2]]\nstretch only, no turn"),
    (rot(30) @ np.diag([1.0, 0.2]), "our G: the same stretch,\nthen turned 30°"),
    (np.array([[0.9, 0.6], [0.1, 0.4]]), "G = [[0.9, 0.6], [0.1, 0.4]]\nany other matrix"),
]

fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), dpi=130)
for a, (G, title) in zip(ax, mats):
    U, S, Vt = np.linalg.svd(G)
    pts = G @ circle
    a.fill(pts[0], pts[1], color="#e9dfd0", zorder=1)
    a.plot(pts[0], pts[1], color=INK, lw=2, zorder=2)
    # a few other radii: every one is between the shortest and the longest
    r = np.hypot(pts[0], pts[1])
    for k in range(0, 400, 25):
        a.plot([0, pts[0, k]], [0, pts[1, k]], color="#c9c2b8", lw=1, zorder=1.5)
    for i, col in enumerate([BIG, WEAK]):
        u = U[:, i] * np.sign(U[np.argmax(np.abs(U[:, i])), i])
        v = u * S[i]
        a.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=3), zorder=3)
        a.text(*(v + u * 0.22), f"{S[i]:.2f}", color=col, fontsize=15, fontweight="bold", ha="center", va="center")
    ang = np.degrees(np.arctan2(U[1, 0], U[0, 0])) % 180
    a.text(0, -0.88, f"longest radius {r.max():.2f}, shortest {r.min():.2f}\nlong arrow at {ang:.0f}°, arrows at 90° to each other",
           ha="center", va="center", color=MUTE, fontsize=12)
    a.set_xlim(-1.35, 1.35); a.set_ylim(-1.05, 0.85); a.set_aspect("equal")
    a.axhline(0, color="#dddddd", lw=1, zorder=0); a.axvline(0, color="#dddddd", lw=1, zorder=0)
    a.set_xticks([]); a.set_yticks([]); a.set_title(title)

fig.suptitle("The arrows are not the x and y axes. They are the longest and the shortest radius of the ellipse, wherever the matrix put them.\n"
             "Gray lines = other directions, all in between. A 2×2 matrix has 2 strengths, 3×3 has 3, a 768×768 weight matrix has 768.",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.86]); fig.savefig("images/step1b_directions.png"); plt.close(fig)
for G, _ in mats:
    print(np.round(np.linalg.svd(G, compute_uv=False), 3))
