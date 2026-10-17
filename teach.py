"""Step-by-step teaching images for Muon's quintic round on one 2×2 matrix.

images/step1.png  a matrix turns a circle into an ellipse; half-widths = strengths
images/step2.png  one round = feed each strength into f(s); matrix formula gives the same numbers
images/step3.png  five rounds as a path on the curve, with a table
images/step4.png  the ellipse after each round: it becomes almost a circle
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.seterr(all="ignore")  # macOS Accelerate matmul raises false overflow warnings
A_, B_, C_ = 3.4445, -4.775, 2.0315
f = lambda s: A_ * s + B_ * s**3 + C_ * s**5
BIG, WEAK, INK, MUTE = "#3a6489", "#b5532f", "#222222", "#888888"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15})


def rot(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


# Example update matrix G: stretches one direction by 1.0, the other by 0.2, then turns 30°.
U, S, V = rot(30), np.array([1.0, 0.2]), np.eye(2)
G = U @ np.diag(S) @ V.T
X0 = G / np.linalg.norm(G)  # Muon's first step: divide by the Frobenius norm


def quintic_round(X):
    A = X @ X.T
    return A_ * X + B_ * A @ X + C_ * A @ A @ X


Xs = [X0]
for _ in range(5):
    Xs.append(quintic_round(Xs[-1]))
svs = [np.linalg.svd(X, compute_uv=False) for X in Xs]  # svs[k] = (big, weak) after k rounds

th = np.linspace(0, 2 * np.pi, 300)
circle = np.stack([np.cos(th), np.sin(th)])


def draw_shape(ax, X, title, s=None, lab=True):
    pts = X @ circle
    ax.fill(pts[0], pts[1], color="#e9dfd0", zorder=1)
    ax.plot(pts[0], pts[1], color=INK, lw=2, zorder=2)
    s = np.linalg.svd(X, compute_uv=False) if s is None else s
    for i, col in enumerate([BIG, WEAK]):
        v = U[:, i] * s[i]
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=3), zorder=3)
        if lab:
            ax.text(*(v + U[:, i] * 0.24), f"{s[i]:.2f}", color=col, fontsize=14, fontweight="bold", ha="center", va="center")
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.set_aspect("equal")
    ax.axhline(0, color="#dddddd", lw=1, zorder=0); ax.axvline(0, color="#dddddd", lw=1, zorder=0)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)


def fmt(M):
    return "[[{:6.3f} {:6.3f}]\n [{:6.3f} {:6.3f}]]".format(*M.ravel())


# ---------- step 1 ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 5.4), dpi=130, gridspec_kw={"width_ratios": [1, 0.75, 1]})
draw_shape(ax[0], np.eye(2), "Before: a circle of radius 1", lab=False)
ax[0].text(0, -1.25, "every direction has length 1", ha="center", color=MUTE)
ax[1].axis("off")
ax[1].text(0.5, 0.72, "multiply every point\nby the update matrix G", ha="center", va="center", fontsize=14)
ax[1].text(0.5, 0.45, "G =\n" + fmt(G), ha="center", va="center", family="monospace", fontsize=14)
ax[1].annotate("", xy=(0.95, 0.2), xytext=(0.05, 0.2), arrowprops=dict(arrowstyle="-|>", lw=3, color=INK), xycoords="axes fraction")
draw_shape(ax[2], G, "After: an ellipse")
ax[2].text(0, -1.25, "blue = strong direction, rust = weak direction", ha="center", color=MUTE)
fig.suptitle("Step 1. A matrix stretches space. The two arrow lengths are its strengths (singular values): 1.00 and 0.20.", fontsize=15)
fig.tight_layout(); fig.savefig("images/step1.png"); plt.close(fig)

# ---------- step 2 ----------
s0, s1 = svs[0], svs[1]
fig, ax = plt.subplots(1, 2, figsize=(15, 6.4), dpi=130, gridspec_kw={"width_ratios": [1.25, 1]})
x = np.linspace(0, 1.25, 400)
a = ax[0]
a.plot(x, f(x), color=INK, lw=2.5, label="f(s) = 3.4445·s − 4.775·s³ + 2.0315·s⁵")
a.plot(x, x, ":", color=MUTE, label="no change")
a.axhline(1, ls="--", color=MUTE, lw=1)
for s, col, name in [(s0[0], BIG, "strong"), (s0[1], WEAK, "weak")]:
    a.plot([s, s], [0, f(s)], color=col, lw=2, ls="--")
    a.plot([0, s], [f(s), f(s)], color=col, lw=2, ls="--")
    a.plot(s, f(s), "o", color=col, ms=10, zorder=5)
    a.text(0.01, f(s) + 0.03, f"{name}: {s:.3f} → {f(s):.3f}", color=col, fontweight="bold")
a.set_xlim(0, 1.15); a.set_ylim(-0.02, 1.3)
a.set_xlabel("strength going in (s)"); a.set_ylabel("strength coming out f(s)")
a.legend(loc="lower right", fontsize=11); a.set_title("Read each strength off the curve")
draw_shape(ax[1], Xs[1], "The matrix after round 1", s=s1)
pts = X0 @ circle
ax[1].plot(pts[0], pts[1], color=MUTE, lw=1.5, ls="--")
ax[1].text(0, -1.22, "dashed = before the round (after dividing by ‖G‖)", ha="center", color=MUTE, fontsize=12)
fig.suptitle("Step 2. One round. First G is divided by its size, so 1.00, 0.20 become 0.981, 0.196.\n"
             "Then X_new = 3.4445·X − 4.775·(XXᵀ)X + 2.0315·(XXᵀ)²X. Its strengths come out exactly f(0.981), f(0.196).", fontsize=14)
fig.tight_layout(); fig.savefig("images/step2.png"); plt.close(fig)

# ---------- step 3 ----------
fig, ax = plt.subplots(1, 2, figsize=(15, 6.4), dpi=130, gridspec_kw={"width_ratios": [1.25, 1]})
a = ax[0]
a.plot(x, f(x), color=INK, lw=2.5, label="f(s)")
a.plot(x, x, ":", color=MUTE, label="no change: move across to here")
a.axhline(1, ls="--", color=MUTE, lw=1)
for i, col in [(0, BIG), (1, WEAK)]:
    s = svs[0][i]
    path = [(s, 0)]
    for k in range(5):
        path += [(s, f(s)), (f(s), f(s))]
        s = f(s)
    px, py = zip(*path)
    a.plot(px, py, color=col, lw=1.8, alpha=0.9)
    a.plot(px[1::2], py[1::2], "o", color=col, ms=6)
a.set_xlim(0, 1.25); a.set_ylim(0, 1.3)
a.set_xlabel("strength going in"); a.set_ylabel("strength coming out")
a.legend(loc="lower right", fontsize=11)
a.set_title("Up to the curve = one round. Across to the dotted line = feed it back in.")
t = ax[1]; t.axis("off")
rows = [["round", "strong", "weak", "weak ÷ strong"]] + [[str(k), f"{b:.3f}", f"{w:.3f}", f"{w / b:.2f}"] for k, (b, w) in enumerate(svs)]
tab = t.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
tab.scale(1, 2.3); tab.set_fontsize(14)
for (r, c), cell in tab.get_celld().items():
    if c == 1 and r > 0: cell.get_text().set_color(BIG)
    if c == 2 and r > 0: cell.get_text().set_color(WEAK)
t.set_title("The two strengths, round by round")
fig.suptitle("Step 3. Repeat 5 times. The weak strength climbs from 0.196 to near 1 in two rounds,\n"
             "then both bounce inside a band around 1 instead of settling exactly on it.", fontsize=14)
fig.tight_layout(); fig.savefig("images/step3.png"); plt.close(fig)

# ---------- step 4 ----------
fig, ax = plt.subplots(1, 6, figsize=(18, 4.2), dpi=130)
for k in range(6):
    draw_shape(ax[k], Xs[k], "start (÷‖G‖)" if k == 0 else f"after round {k}", s=svs[k])
fig.suptitle("Step 4. The same matrix after each round. The arrows keep their directions; only their lengths change.\n"
             "Squashed ellipse → almost a circle: the update now moves every direction by a similar amount.", fontsize=14)
fig.tight_layout(); fig.savefig("images/step4.png"); plt.close(fig)

# check: matrix formula == f applied to each strength
for k in range(5):
    assert np.allclose(np.sort(svs[k + 1]), np.sort(np.abs(f(svs[k])))), k
print("strengths per round:", [np.round(s, 3).tolist() for s in svs])
print("G =", np.round(G, 3).tolist())
