"""Figures for the burst post: post_seeds.png, post_letters.png, post_where.png (from confirm, tuned, cause, cause_qkvo, weak .jsonl)."""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
rd = lambda f: [json.loads(l) for l in open(os.path.join(HERE, f))]
C = {"adam": "tab:blue", "burst": "tab:orange"}
LAB = {"adam": "AdamW only", "burst": "100 Muon steps, then AdamW"}

# 1. reverse accuracy over steps, 8 + 4 seeds
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, f in zip(axs, (0.12, 0.06)):
    for arm in ("adam", "burst"):
        rs = [r for r in rd("confirm.jsonl") if r["freq_new"] == f and r["arm"] == arm]
        for j, r in enumerate(rs):
            ax.plot([e["step"] for e in r["log"]], [e["R"] for e in r["log"]], color=C[arm], alpha=0.6, lw=1.3, label=LAB[arm] if j == 0 else None)
    ax.axvspan(0, 100, color="tab:orange", alpha=0.12); ax.text(50, 1.04, "Muon\nburst", ha="center", va="bottom", fontsize=8, color="tab:orange")
    ax.set_xlim(0, 700); ax.set_ylim(-0.02, 1.15); ax.set_xlabel("steps after reverse is added")
    ax.set_title(f"reverse = {f:.0%} of each batch, {len(rs)} seeds (one line each)")
axs[0].set_ylabel("reverse: share of answers fully right"); axs[0].legend(loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "post_seeds.png"), dpi=120); plt.close(fig)

# 2. per answer letter, seed 0
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
cm = plt.get_cmap("viridis")
for ax, arm in zip(axs, ("adam", "burst")):
    r = next(r for r in rd("tuned.jsonl") if r["seed"] == 0 and r["arm"] == arm)
    st = [e["step"] for e in r["log"]]
    for k in range(6):
        ax.plot(st, [e["Rpos"][k] for e in r["log"]], color=cm(k / 5), lw=2, label=f"answer letter {k + 1}")
    if arm == "burst": ax.axvspan(0, 100, color="tab:orange", alpha=0.12)
    ax.axhline(0.1, color="gray", ls=":", lw=1); ax.text(790, 0.03, "chance (1 in 10)", ha="right", fontsize=8, color="gray")
    ax.set_xlim(0, 800); ax.set_xlabel("steps after reverse is added"); ax.set_title(LAB[arm] + " (seed 0)")
axs[0].set_ylabel("share of this answer letter right"); axs[1].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "post_letters.png"), dpi=120); plt.close(fig)

# 3. where the speed-up lives: which weights, which directions
score = lambda r, n: sum(r["res"][n].values()) / len(r["res"][n])
def rows(f, items):
    rs = rd(f); return [(lab, [score(r, n) for r in rs]) for n, lab in items]
A = rows("cause.jsonl", [("restart", "no burst (AdamW only)"), ("burst", "full burst change"),
                          ("only attn1", "only attention, layer 2"), ("only attn0", "only attention, layer 1"),
                          ("only ff0", "only feed-forward, layer 1"), ("only ff1", "only feed-forward, layer 2"),
                          ("only emb+norms", "only embeddings + norms")])
A += rows("cause_qkvo.jsonl", [("only VO", "only attention V/O (what is passed on)"), ("only QK", "only attention Q/K (where to look)")])
B0 = rows("cause.jsonl", [("kick", "random change, same size"), ("respin", "same strengths, random directions")])
B = rows("weak.jsonl", [("restart", "no burst (AdamW only)"), ("burst", "full burst change"),
                         ("burst weak", "burst change minus its 5 strongest directions"), ("burst strong", "only its 5 strongest directions"),
                         ("adam flat", "AdamW's change, directions evened out"), ("adam scaled", "AdamW's change, scaled to burst size")])
B = B[:4] + B0 + B[4:]
fig, axs = plt.subplots(1, 2, figsize=(15, 5.4))
for ax, data, title in zip(axs, (A, B), ("Which weights carry it", "Which directions carry it (all at the burst's size)")):
    for i, (lab, v) in enumerate(data):
        y = len(data) - 1 - i; m = sum(v) / len(v)
        col = "tab:blue" if lab.startswith("no burst") else "tab:orange" if lab.startswith("full") else "0.55"
        ax.barh(y, m, color=col, alpha=0.85); ax.scatter(v, [y] * len(v), s=10, color="k", zorder=3)
        ax.text(1.02, y, f"{m:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(len(data)), [d[0] for d in reversed(data)], fontsize=9); ax.set_xlim(0, 1.1)
    ax.set_xlabel("reverse accuracy, mean over steps 200–500 (dots = 8 seeds)"); ax.set_title(title)
fig.suptitle("Start from the model before the burst, add part of the burst's weight change, then train with AdamW on the same batches")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "post_where.png"), dpi=120); plt.close(fig)
