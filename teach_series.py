"""Image series, one move per image.

images/tst1.png … tst4.png   G = U·S·Vᵀ as three moves (turn, stretch, turn), then the goal U·Vᵀ
images/cube.png              X → XXᵀ → XXᵀX: every strength squared, then cubed; the arrows stay
images/round1.png … round5.png  one quintic round per image: the curve on the left, the matrix on the right
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.seterr(all="ignore")  # macOS Accelerate matmul raises false overflow warnings
A_, B_, C_ = 3.4445, -4.775, 2.0315
f = lambda s: A_ * s + B_ * s**3 + C_ * s**5
BIG, WEAK, INK, MUTE, FILL = "#3a6489", "#b5532f", "#222222", "#888888", "#e9dfd0"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15})
th = np.linspace(0, 2 * np.pi, 300)
circle = np.stack([np.cos(th), np.sin(th)])


def rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def board(ax, lim=1.35):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.axhline(0, color="#dddddd", lw=1, zorder=0); ax.axvline(0, color="#dddddd", lw=1, zorder=0)
    ax.set_xticks([]); ax.set_yticks([])


def shape(ax, M, a1, a2, ghost=False, labels=True):
    """Draw M·circle and the blue/red arrows a1, a2. ghost = the previous state, dashed and faded."""
    pts = M @ circle
    if ghost:
        ax.plot(pts[0], pts[1], color=MUTE, lw=1.5, ls="--", zorder=4)
    else:
        ax.fill(pts[0], pts[1], color=FILL, zorder=1)
        ax.plot(pts[0], pts[1], color=INK, lw=2, zorder=2)
    for v, col in [(a1, BIG), (a2, WEAK)]:
        ax.annotate("", xy=v, xytext=(0, 0), zorder=4 if ghost else 3,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5 if ghost else 3, alpha=0.35 if ghost else 1))
        if labels and not ghost:
            n = np.linalg.norm(v)
            ax.text(*(v + v / max(n, 1e-9) * 0.22), f"{n:.2f}", color=col, fontsize=14, fontweight="bold",
                    ha="center", va="center")


# ---------- turn, stretch, turn ----------
# A general example (our G from section 1 has no first turn, so it would hide move 1).
V = rot(50); S = np.array([1.0, 0.3]); U = rot(30)
stages = [  # (matrix so far, blue arrow, red arrow)
    (np.eye(2), V[:, 0], V[:, 1]),
    (V.T, np.array([1.0, 0]), np.array([0, 1.0])),
    (np.diag(S) @ V.T, np.array([S[0], 0]), np.array([0, S[1]])),
    (U @ np.diag(S) @ V.T, U[:, 0] * S[0], U[:, 1] * S[1]),
]
moves = [
    ("Move 1: Vᵀ turns by −50°",
     "Pick the two arrows that will become the stretch directions (they are 90° apart).\n"
     "The first turn brings them onto the axes. Turning changes no length: the circle stays a circle."),
    ("Move 2: S stretches",
     "Stretch along the axes: blue by 1.00, red by 0.30. The circle becomes an ellipse.\n"
     "These two numbers are the strengths (singular values). Only this move changes lengths."),
    ("Move 3: U turns by +30°",
     "The second turn points the ellipse in its final direction. Lengths stay 1.00 and 0.30.\n"
     "The result is the whole matrix G = U·S·Vᵀ: turn, stretch, turn."),
]
for k, (title, note) in enumerate(moves, start=1):
    fig, ax = plt.subplots(figsize=(8.6, 8.4), dpi=130)
    board(ax)
    shape(ax, *stages[k - 1], ghost=True)
    shape(ax, *stages[k])
    ax.text(0, -1.3, "dashed = before this move", ha="center", color=MUTE, fontsize=12)
    ax.set_title(note, fontsize=12.5)
    fig.suptitle(title, fontsize=17, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"images/tst{k}.png"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(14, 7.4), dpi=130)
for a, (M, a1, a2), name in [(ax[0], stages[3], "G = U·S·Vᵀ\nturn, stretch, turn"),
                             (ax[1], (U @ V.T, U[:, 0], U[:, 1]), "goal = U·Vᵀ\nturn, (no stretch), turn")]:
    board(a); shape(a, np.eye(2), V[:, 0], V[:, 1], ghost=True); shape(a, M, a1, a2); a.set_title(name)
    a.text(0, -1.25, "dashed = the circle before the matrix", ha="center", color=MUTE, fontsize=12)
fig.suptitle("What Muon wants: keep both turns, drop the stretch. Every strength becomes 1, the arrows keep their directions.",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig("images/tst4.png"); plt.close(fig)

# ---------- X → XXᵀ → XXᵀX ----------
Ug, Sg = rot(30), np.array([1.0, 0.2])
G = Ug @ np.diag(Sg)
X = G / np.linalg.norm(G)
s = np.linalg.svd(X, compute_uv=False)
mats = [(X, "X", s, "strengths 0.981 and 0.196"),
        (X @ X.T, "X·Xᵀ", s ** 2, "each strength squared"),
        (X @ X.T @ X, "X·Xᵀ·X", s ** 3, "each strength cubed")]
fig, ax = plt.subplots(1, 5, figsize=(19, 5.6), dpi=130, gridspec_kw={"width_ratios": [1, 0.3, 1, 0.3, 1]})
for a, (M, name, sv, sub) in zip(ax[::2], mats):
    assert np.allclose(np.linalg.svd(M, compute_uv=False), sv)
    board(a, 1.25)
    shape(a, M, Ug[:, 0] * sv[0], Ug[:, 1] * sv[1], labels=False)
    a.set_title(f"{name}\n{sub}")
    a.text(-0.05, -1.12, f"blue {sv[0]:.3f}", ha="right", color=BIG, fontsize=13, fontweight="bold")
    a.text(0.05, -1.12, f"red {sv[1]:.4f}", ha="left", color=WEAK, fontsize=13, fontweight="bold")
for a, txt in zip(ax[1::2], ["× Xᵀ\nthen", "× X"]):
    a.axis("off")
    a.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5), xycoords="axes fraction",
               arrowprops=dict(arrowstyle="-|>", lw=3, color=INK))
    a.text(0.5, 0.6, txt.split("\n")[0], ha="center", va="bottom", fontsize=15, transform=a.transAxes)
fig.suptitle("Multiplying X by Xᵀ squares every strength. Multiplying by X again makes it the cube. "
             "The arrows never turn.\nSo X, X·Xᵀ·X and (X·Xᵀ)²·X share the same arrows, with strengths s, s³ and s⁵. "
             "Adding them with the weights 3.4445, −4.775, 2.0315 gives f(s).", fontsize=13.5)
fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig("images/cube.png"); plt.close(fig)


# ---------- one round per image ----------
def quintic_round(M):
    A = M @ M.T
    return A_ * M + B_ * A @ M + C_ * A @ A @ M


Xs = [X]
for _ in range(5):
    Xs.append(quintic_round(Xs[-1]))
svs = [np.linalg.svd(M, compute_uv=False) for M in Xs]
for k in range(5):
    assert np.allclose(np.sort(svs[k + 1]), np.sort(np.abs(f(svs[k]))))

x = np.linspace(0, 1.25, 400)
for k in range(1, 6):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.6), dpi=130, gridspec_kw={"width_ratios": [1.25, 1]})
    a = ax[0]
    a.plot(x, f(x), color=INK, lw=2.5, label="the curve f(s)")
    a.plot(x, x, ":", color=MUTE, label="no change (feed back in)")
    a.axhline(1, ls="--", color=MUTE, lw=1)
    for i, col, name in [(0, BIG, "blue"), (1, WEAK, "red")]:
        path = [(svs[0][i], 0)]
        for j in range(k):
            sj, sn = svs[j][i], svs[j + 1][i]
            path += [(sj, sn), (sn, sn)]
        px, py = zip(*path[:-1])
        a.plot(px[:-1], py[:-1], color=col, lw=1.2, alpha=0.35)       # earlier rounds and the feed-back
        a.plot(px[-2:], py[-2:], color=col, lw=2.6)                     # this round: up to the curve
        a.plot(px[-1], py[-1], "o", color=col, ms=11, zorder=5)
        a.text(0.02, 1.25 - 0.09 * i, f"{name}: {svs[k - 1][i]:.3f} → {svs[k][i]:.3f}",
               color=col, fontweight="bold", fontsize=15, va="top")
    a.set_xlim(0, 1.25); a.set_ylim(0, 1.3)
    a.set_xlabel("strength going in"); a.set_ylabel("strength coming out")
    a.legend(loc="lower right", fontsize=11)
    a.set_title("Go up to the curve: that is this round")
    b = ax[1]
    board(b)
    shape(b, Xs[k - 1], Ug[:, 0] * svs[k - 1][0], Ug[:, 1] * svs[k - 1][1], ghost=True)
    shape(b, Xs[k], Ug[:, 0] * svs[k][0], Ug[:, 1] * svs[k][1])
    b.text(0, -1.25, "dashed = before this round", ha="center", color=MUTE, fontsize=12)
    b.set_title(f"The matrix after round {k}")
    ratio = svs[k][1] / svs[k][0]
    note = {1: "The weak strength more than triples. The strong one is pulled down.",
            2: "Both pass 1. The curve overshoots on purpose: that is the price of fast growth near 0.",
            3: "Both fall back below 1. From here they bounce in a band around 1.",
            4: "Still bouncing. The ellipse stays round.",
            5: "Done. Neither strength is exactly 1, but the update now moves both directions by a similar amount."}[k]
    fig.suptitle(f"Round {k} of 5. {note}  (red ÷ blue = {ratio:.2f})", fontsize=14)
    fig.tight_layout(); fig.savefig(f"images/round{k}.png"); plt.close(fig)
print("strengths per round:", [np.round(v, 3).tolist() for v in svs])
