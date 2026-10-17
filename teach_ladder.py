"""Section 2 ladder: why X·Xᵀ·X cubes the strengths, and how the curve's three numbers are built up.

images/chain.png            the 9 moves of X·Xᵀ·X on a circle; the middle turns undo each other
images/curve1.png … curve3.png  the curve f(s) built one term at a time
images/xx_fail.png          X, X·X and X·Xᵀ·X side by side: only X·Xᵀ·X keeps X's arrows
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIG, WEAK, INK, MUTE, FILL = "#3a6489", "#b5532f", "#222222", "#888888", "#e9dfd0"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 13})
th = np.linspace(0, 2 * np.pi, 300)
circle = np.stack([np.cos(th), np.sin(th)])


def rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


# ---------- chain: X·Xᵀ·X as 9 moves ----------
# Same strengths as our G after step 0 (0.981, 0.196); a −50° first turn is added so every turn is visible.
V, U, S = rot(50), rot(30), np.diag([0.981, 0.196])
moves = [  # (matrix of this move, title, note)
    (V.T, "1. Vᵀ: turn −50°", "X starts"),
    (S, "2. S: stretch", ""),
    (U, "3. U: turn +30°", "X done"),
    (U.T, "4. Uᵀ: turn −30°", "undoes move 3"),
    (S, "5. S: stretch", "same arrows as move 2"),
    (V, "6. V: turn +50°", "Xᵀ done"),
    (V.T, "7. Vᵀ: turn −50°", "undoes move 6"),
    (S, "8. S: stretch", "same arrows again"),
    (U, "9. U: turn +30°", "X done: X·Xᵀ·X"),
]
fig, axs = plt.subplots(2, 5, figsize=(19, 9.8), dpi=130)
M = np.eye(2)
panels = [(M.copy(), "start: a circle", "")]
for m, title, note in moves:
    M = m @ M
    panels.append((M.copy(), title, note))
for a, (M, title, note) in zip(axs.ravel(), panels):
    pts = M @ circle
    a.fill(pts[0], pts[1], color=FILL, zorder=1); a.plot(pts[0], pts[1], color=INK, lw=1.8, zorder=2)
    for i, col in [(0, BIG), (1, WEAK)]:
        v = M @ V[:, i]
        a.annotate("", xy=v, xytext=(0, 0), zorder=3, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
    s_big, s_weak = (np.linalg.norm(M @ V[:, i]) for i in (0, 1))
    a.text(0, -1.22, f"blue {s_big:.3f}", ha="right", color=BIG, fontsize=11.5, fontweight="bold")
    a.text(0.08, -1.22, f"red {s_weak:.4f}", ha="left", color=WEAK, fontsize=11.5, fontweight="bold")
    a.set_xlim(-1.2, 1.2); a.set_ylim(-1.35, 1.2); a.set_aspect("equal")
    a.axhline(0, color="#dddddd", lw=1, zorder=0); a.axvline(0, color="#dddddd", lw=1, zorder=0)
    a.set_xticks([]); a.set_yticks([])
    a.set_title(title + ("\n" + note if note else "\n "), fontsize=12.5)
    if note.startswith("undoes"):
        for sp in a.spines.values(): sp.set_color("#2e8b57"); sp.set_linewidth(2.5)
fig.suptitle("X·Xᵀ·X, one move at a time (applied left to right, top row first). Green frame = a turn that undoes the turn just before it.\n"
             "Left over: turn, stretch, stretch, stretch, turn. Same turns as X, each strength multiplied by itself 3 times: "
             "0.981 → 0.944, 0.196 → 0.0075.", fontsize=13.5)
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.84, wspace=0.1, hspace=0.32); fig.savefig("images/chain.png"); plt.close(fig)
assert np.allclose(panels[-1][0], U @ S ** 3 @ V.T)

# ---------- X·X turns the arrows, X·Xᵀ·X keeps them ----------
X = U @ S @ rot(-40).T  # first turn +40° here: X·X then misses X's arrows by 23°, easy to see
cases = [(X, "X", "our matrix"),
         (X @ X, "X·X", "the arrows point somewhere new"),
         (X @ X.T @ X, "X·Xᵀ·X", "same arrows as X, strengths cubed")]
fig, axs = plt.subplots(1, 3, figsize=(17, 6.6), dpi=130)
for a, (M, name, sub) in zip(axs, cases):
    pts = M @ circle
    a.fill(pts[0], pts[1], color=FILL, zorder=1); a.plot(pts[0], pts[1], color=INK, lw=1.8, zorder=2)
    for i in (0, 1):  # X's own arrow directions, as dashed guide lines
        d = U[:, i] * 1.15
        a.plot([-d[0], d[0]], [-d[1], d[1]], ls="--", color=MUTE, lw=1.2, zorder=0)
    u, sv, _ = np.linalg.svd(M)
    for i, col in [(0, BIG), (1, WEAK)]:
        v = u[:, i] * sv[i]
        if v @ U[:, i] < 0: v = -v
        a.annotate("", xy=v, xytext=(0, 0), zorder=3, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.8))
    a.text(0, -1.2, f"blue {sv[0]:.3f}", ha="right", color=BIG, fontsize=13, fontweight="bold")
    a.text(0.08, -1.2, f"red {sv[1]:.4f}", ha="left", color=WEAK, fontsize=13, fontweight="bold")
    a.set_xlim(-1.2, 1.2); a.set_ylim(-1.3, 1.2); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"{name}\n{sub}", fontsize=15)
fig.suptitle("Dashed lines = the directions of X's arrows. X·X leaves them. X·Xᵀ·X stays on them.", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("images/xx_fail.png"); plt.close(fig)

# ---------- the curve, one term at a time ----------
a_, b_, c_ = 3.4445, -4.775, 2.0315
terms = [lambda s: a_ * s, lambda s: b_ * s ** 3, lambda s: c_ * s ** 5]
labels = ["3.4445·s", "− 4.775·s³", "+ 2.0315·s⁵"]
s = np.linspace(0, 1.0, 400)  # after step 0 every strength is 1 or less
pts = [(0.196, WEAK, "red"), (0.981, BIG, "blue")]
notes = [
    "Term 1 alone: every strength is multiplied by 3.4445. The weak red one grows to 0.675, good.\n"
    "But the strong blue one becomes 3.379, far above 1.",
    "Add −4.775·s³. For small s, s³ is tiny (0.196³ = 0.0075), so red barely changes.\n"
    "Near 1, s³ is almost 1, so blue is pulled down hard: to −1.129, too far.",
    "Add +2.0315·s⁵. s⁵ is even tinier for small s, so red stays at 0.640.\n"
    "Near 1 it lifts blue back up to 0.717. Now both land near 1.",
]
for k in range(1, 4):
    f_now = lambda x: sum(t(x) for t in terms[:k])
    f_prev = lambda x: sum(t(x) for t in terms[:k - 1])
    fig, ax = plt.subplots(figsize=(11, 7), dpi=130)
    ax.axhspan(0.7, 1.2, color="#e6f0e6", zorder=0)
    ax.axhline(1, ls="--", color=MUTE, lw=1)
    ax.axhline(0, color="#cccccc", lw=1)
    if k > 1:
        ax.plot(s, f_prev(s), color=MUTE, lw=1.8, ls="--", label="before this term")
    ax.plot(s, f_now(s), color=INK, lw=3, label="f(s) = " + " ".join(labels[:k]))
    for x, col, name in pts:
        y = f_now(x)
        if k > 1:
            ax.annotate("", xy=(x, y), xytext=(x, f_prev(x)), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
        ax.plot(x, y, "o", color=col, ms=11, zorder=5)
        ax.text(x + 0.04, y, f"{name}: {x:.3f} → {y:.3f}", color=col, fontweight="bold", fontsize=14, va="center")
    ax.text(1.41, 1.17, "green band: near 1, where we want every strength", ha="right", va="top", fontsize=11, color="#2e6b2e")
    ax.set_xlim(0, 1.42); ax.set_ylim(-1.6, 3.9)
    ax.set_xlabel("s = one strength going in"); ax.set_ylabel("f(s) = that strength coming out")
    ax.legend(loc="upper left", fontsize=12)
    ax.set_title(notes[k - 1], fontsize=13)
    fig.suptitle(f"Building the curve, term {k} of 3", fontsize=16, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"images/curve{k}.png"); plt.close(fig)

for x in (0.196, 0.981):
    print(x, [round(t(x), 4) for t in terms], round(sum(t(x) for t in terms), 4))
