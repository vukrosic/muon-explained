"""Section 2 ladder: why X·Xᵀ·X cubes the strengths, and how the curve's three numbers are built up.

images/idea1.png … idea5.png, idea3a.png, idea3b.png  why s alone fails, why s³ and then s⁵ are added
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

# ---------- the idea: why s, s³ and s⁵ ----------
xs = np.linspace(0, 1.0, 400)
RED, BLUE = 0.196, 0.981


def idea_axes(title, step, ylim=(0, 1.65)):
    fig, ax = plt.subplots(figsize=(11, 7.2), dpi=130)
    ax.axhline(1, ls="--", color=MUTE, lw=1)
    ax.set_xlim(0, 1.05); ax.set_ylim(*ylim)
    ax.set_xlabel("s = one strength going in"); ax.set_ylabel("strength coming out")
    ax.set_title(title, fontsize=13)
    fig.suptitle(f"The idea, step {step} of 5", fontsize=16, fontweight="bold")
    return fig, ax


def mark(ax, f, dx=0.03, fmt="{:.3f}"):
    for x, col, name in [(RED, WEAK, "red"), (BLUE, BIG, "blue")]:
        y = f(x)
        ax.plot(x, y, "o", color=col, ms=10, zorder=5)
        ax.text(x + dx if x < 0.5 else x - dx, y, f"{name}: {x} → " + fmt.format(y), color=col, fontweight="bold",
                fontsize=13, va="center", ha="left" if x < 0.5 else "right")


# step 1: the goal
fig, ax = idea_axes("How to read these plots: pick a strength on the bottom axis, go up to the line, read the new strength on the left.\n"
                    "Gray line = X as it is (nothing changes). Green line = the goal: every strength comes out as 1.", 1)
ax.plot(xs, xs, ":", color=MUTE, lw=2, label="X now: strength out = strength in")
ax.plot(xs, np.ones_like(xs), color="#2e8b57", lw=4, label="goal: every strength becomes 1")
for x, col, name, k in [(RED, WEAK, "red", 1 / RED), (BLUE, BIG, "blue", 1 / BLUE)]:
    ax.plot(x, x, "o", color=col, ms=10, zorder=5)
    ax.annotate("", xy=(x, 0.985), xytext=(x, x + 0.02), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
    ax.text(x + 0.02 if x < 0.5 else x - 0.02, (x + 1) / 2 if x < 0.5 else 1.12,
            f"{name}: {x} → 1\nneeds ×{k:.2f}", color=col, fontweight="bold", fontsize=13,
            ha="left" if x < 0.5 else "right", va="center")
ax.legend(loc="upper left", fontsize=12)
fig.tight_layout(); fig.savefig("images/idea1.png"); plt.close(fig)

# step 2: the three building blocks
fig, ax = idea_axes("s³ means s·s·s and s⁵ means s·s·s·s·s. A small strength almost vanishes: 0.196³ = 0.0075.\n"
                    "A strength near 1 barely changes: 0.981³ = 0.944. So s³ and s⁵ are big only where s is big.", 2,
                    ylim=(0, 1.45))
for p, ls, lab in [(1, "-", "s"), (3, "--", "s³ = s·s·s"), (5, ":", "s⁵ = s·s·s·s·s")]:
    ax.plot(xs, xs ** p, color=INK, lw=3, ls=ls, label=lab)
ax.axvline(RED, color=WEAK, lw=1, alpha=0.5); ax.axvline(BLUE, color=BIG, lw=1, alpha=0.5)
ax.text(RED + 0.015, 1.4, f"at red, s = 0.196:\n s  = 0.196\n s³ = {RED**3:.4f}\n s⁵ = {RED**5:.4f}",
        color=WEAK, fontweight="bold", fontsize=13, va="top", family="monospace")
ax.text(BLUE - 0.015, 1.4, f"at blue, s = 0.981:\n s  = 0.981\n s³ = {BLUE**3:.3f}\n s⁵ = {BLUE**5:.3f}",
        color=BIG, fontweight="bold", fontsize=13, va="top", ha="right", family="monospace")
ax.legend(loc="center left", fontsize=12)
fig.tight_layout(); fig.savefig("images/idea2.png"); plt.close(fig)

# step 3a: multiplying X by one number a scales both arrows by a
Ux = rot(30)
fig, axs = plt.subplots(1, 3, figsize=(17, 6.8), dpi=130)
for ax, a, name, sub in [(axs[0], 1.0, "X", "our matrix: arrows 0.981 and 0.196"),
                         (axs[1], 1 / BLUE, "1.02 · X", "a = 1.02 = 1 ÷ 0.981, chosen for blue"),
                         (axs[2], 1 / RED, "5.10 · X", "a = 5.10 = 1 ÷ 0.196, chosen for red")]:
    sv = a * np.array([BLUE, RED])
    M = Ux @ np.diag(sv)
    pts = M @ circle
    lim = max(1.35, 1.12 * sv[0])
    ax.fill(pts[0], pts[1], color=FILL, zorder=1); ax.plot(pts[0], pts[1], color=INK, lw=2, zorder=2)
    ax.plot(np.cos(th), np.sin(th), color="#2e8b57", lw=2, ls="--", zorder=3)
    for i, col, nm in [(0, BIG, "blue"), (1, WEAK, "red")]:
        v = Ux[:, i] * sv[i]
        ax.annotate("", xy=v, xytext=(0, 0), zorder=4, arrowprops=dict(arrowstyle="-|>", color=col, lw=3))
    ok = lambda x: "  ✓" if abs(x - 1) < 0.01 else ""
    ax.text(0, -0.84 * lim, f"blue {sv[0]:.2f}{ok(sv[0])}", ha="right", color=BIG, fontsize=14, fontweight="bold")
    ax.text(0.04 * lim, -0.84 * lim, f"red {sv[1]:.2f}{ok(sv[1])}", ha="left", color=WEAK, fontsize=14, fontweight="bold")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.axhline(0, color="#dddddd", lw=1, zorder=0); ax.axvline(0, color="#dddddd", lw=1, zorder=0)
    ax.set_title(f"{name}\n{sub}", fontsize=14)
axs[2].text(0, 0.93 * 5.6, "(zoomed out: this shape is 5× bigger)", ha="center", va="top", fontsize=11, color=MUTE)
fig.suptitle("Green dashed circle = the goal: both arrows should end on it (length 1).\n"
             "Multiplying X by a number a makes the whole shape a times bigger, so BOTH arrows grow by a. "
             "Red needs 5.10, blue needs 1.02.", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig("images/idea3a.png"); plt.close(fig)

# step 3b: one number a, applied to many strengths, gives dots on a straight line
A3 = 1 / RED
fig, axs = plt.subplots(1, 3, figsize=(19, 6.6), dpi=130)
few = [(0.1, INK, ""), (RED, WEAK, "red "), (0.5, INK, ""), (BLUE, BIG, "blue ")]
many = np.round(np.arange(0, 1.0001, 0.05), 2)
for k, ax in enumerate(axs):
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 5.6); ax.axhline(1, ls="--", color=MUTE, lw=1)
    ax.set_xlabel("s = strength going in"); ax.set_ylabel("strength coming out = 5.10 × s")
    if k == 0:
        for x, col, nm in few:
            ax.plot(x, A3 * x, "o", color=col, ms=11, zorder=5)
            right = x > 0.7
            ax.text(x - 0.03 if right else x + 0.03, A3 * x, f"{nm}{x} → {A3 * x:.2f}", color=col, fontsize=13,
                    fontweight="bold", va="center", ha="right" if right else "left",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
        ax.set_title("1. Four strengths, each multiplied by 5.10.\nEach one is a dot: across = in, up = out.", fontsize=13)
    elif k == 1:
        ax.plot(many, A3 * many, "o", color=INK, ms=7, zorder=4)
        for x, col, nm in few[1::2]:
            ax.plot(x, A3 * x, "o", color=col, ms=11, zorder=5)
        ax.set_title("2. Every strength from 0 to 1 in steps of 0.05.\nEach step right adds 5.10 × 0.05 = 0.255 up.", fontsize=13)
        for x in (0.6, 0.65):
            ax.annotate("", xy=(x + 0.05, A3 * x), xytext=(x, A3 * x), arrowprops=dict(arrowstyle="-", color=MUTE, lw=1.5))
        ax.annotate("", xy=(0.65, A3 * 0.65), xytext=(0.65, A3 * 0.6), arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.5))
        ax.text(0.665, A3 * 0.625, "+0.255", fontsize=11, color=MUTE, va="center")
    else:
        ax.plot(xs, A3 * xs, color=INK, lw=3, zorder=3)
        ax.plot(many, A3 * many, "o", color=INK, ms=5, zorder=4, alpha=0.5)
        for x, col, nm in few[1::2]:
            ax.plot(x, A3 * x, "o", color=col, ms=11, zorder=5)
            ax.text(x + 0.03 if x < 0.5 else x - 0.03, A3 * x, f"{nm}→ {A3 * x:.2f}", color=col, fontsize=13,
                    fontweight="bold", va="center", ha="left" if x < 0.5 else "right")
        ax.set_title("3. Same step up for every step right:\nthe dots form a straight line. It shows\nwhat 5.10 does to ANY strength.", fontsize=13)
fig.suptitle("Where the straight line comes from: multiply many strengths by the same number 5.10 and mark each result as a dot",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig("images/idea3b.png"); plt.close(fig)

# step 3: s alone is a straight line
fig, ax = idea_axes("With s alone, the only option is a·s: multiply every strength by the same number a.\n"
                    "Red needs ×5.10, blue needs ×1.02. No single a does both: the line cannot be flat.", 3,
                    ylim=(0, 5.4))
for a, ls, lab in [(1 / RED, "-", "5.10·s: red reaches 1, blue goes to 5.01"),
                   (1 / BLUE, "--", "1.02·s: blue reaches 1, red stays at 0.20")]:
    ax.plot(xs, a * xs, color=INK, lw=3, ls=ls, label=lab)
    for x, col in [(RED, WEAK), (BLUE, BIG)]:
        ax.plot(x, a * x, "o", color=col, ms=10, zorder=5)
ax.plot(xs, np.ones_like(xs), color="#2e8b57", lw=3, alpha=0.6, label="goal")
ax.legend(loc="upper left", fontsize=12)
fig.tight_layout(); fig.savefig("images/idea3.png"); plt.close(fig)

# step 4: add s³ — two cubics, each with a flaw
cub_steep = lambda x: 3.4445 * x - 2.7445 * x ** 3
cub_flat = lambda x: 1.5 * x - 0.5 * x ** 3
fig, ax = idea_axes("Add b·s³ with b negative. It takes almost nothing from red and a lot from blue, so the line bends down.\n"
                    "But with two numbers Muon must choose: a gentle start (red only reaches 0.290)\n"
                    "or a steep start that overshoots to 1.49 before it comes down.", 4)
ax.axhspan(0.7, 1.2, color="#e6f0e6", zorder=0)
ax.plot(xs, cub_flat(xs), color=MUTE, lw=3, ls="--", label="1.5·s − 0.5·s³   gentle start, never above 1")
ax.plot(xs, cub_steep(xs), color=INK, lw=3, label="3.4445·s − 2.7445·s³   steep start, bulges to 1.49")
for f in (cub_flat, cub_steep):
    for x, col in [(RED, WEAK), (BLUE, BIG)]:
        ax.plot(x, f(x), "o", color=col, ms=10, zorder=5)
ax.text(RED + 0.02, cub_flat(RED) - 0.03, f"red → {cub_flat(RED):.3f}", color=WEAK, fontweight="bold", fontsize=13, va="top")
ax.text(RED + 0.02, cub_steep(RED), f"red → {cub_steep(RED):.3f}", color=WEAK, fontweight="bold", fontsize=13, va="center")
for f in (cub_flat, cub_steep):
    ax.text(BLUE + 0.03, f(BLUE), f"blue → {f(BLUE):.3f}", color=BIG, fontweight="bold", fontsize=13, va="center")
ax.set_xlim(0, 1.3)
ax.text(0.647, 1.52, "1.49", ha="center", fontsize=13, fontweight="bold")
ax.text(1.29, 1.17, "green band: 0.7 to 1.2", ha="right", va="top", fontsize=11, color="#2e6b2e")
ax.legend(loc="lower right", fontsize=12)
fig.tight_layout(); fig.savefig("images/idea4.png"); plt.close(fig)

# step 5: add s⁵ — Muon's curve
quint = lambda x: 3.4445 * x - 4.775 * x ** 3 + 2.0315 * x ** 5
fig, ax = idea_axes("Add c·s⁵. Like s³, it acts on big strengths only, even more so. Now s³ may pull down harder (−4.775)\n"
                    "and s⁵ lifts the end back up. Same steep start as the steep cubic, but the top is 1.20 instead of 1.49.", 5)
ax.axhspan(0.7, 1.2, color="#e6f0e6", zorder=0)
ax.plot(xs, cub_steep(xs), color=MUTE, lw=2, ls="--", label="steep cubic from step 4")
ax.plot(xs, quint(xs), color=INK, lw=3.5, label="3.4445·s − 4.775·s³ + 2.0315·s⁵   Muon's curve")
mark(ax, quint)
ax.text(0.555, 1.24, "1.20", ha="center", fontsize=13, fontweight="bold")
ax.text(1.04, 1.17, "green band: 0.7 to 1.2", ha="right", va="top", fontsize=11, color="#2e6b2e")
ax.legend(loc="upper left", fontsize=12)
fig.tight_layout(); fig.savefig("images/idea5.png"); plt.close(fig)

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
