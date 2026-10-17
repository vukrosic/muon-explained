"""Six views of experiment 2's results, plus the real gradient strengths behind them.

Step 1 (about 1 minute, once): rerun p = 0 at lr 0.01, seed 0, and save the strengths of layer h1's
update (singular values after dividing by its size) at a few steps -> exp2_strengths.json.
Step 2: draw ../images/exp2_explore.png from exp2_results.json and exp2_strengths.json.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SPATH = os.path.join(HERE, "exp2_strengths.json")
KEEP = [10, 100, 400, 900]

if not os.path.exists(SPATH):
    import torch
    from train import run
    calls, saved = [0], {}

    def flat(G):                                   # exact p = 0, recording h1 (2nd Muon matrix each step)
        step, which = divmod(calls[0], 3); calls[0] += 1
        S_in = None
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        if which == 1 and step in KEEP:
            saved[step] = (S / S.norm()).tolist()
        return U @ Vh
    r = run(flat, 0, 0.01)
    print("rerun val", r["val"])
    json.dump(saved, open(SPATH, "w"))

R = json.load(open(os.path.join(HERE, "exp2_results.json")))
S = {int(k): np.array(v) for k, v in json.load(open(SPATH)).items()}
P = [0, 0.25, 0.5, 1]
LR = [0.005, 0.007, 0.01, 0.014, 0.02]
BEST = {0: 0.01, 0.25: 0.007, 0.5: 0.01, 1: 0.01}
COL = {0: "#1f77b4", 0.25: "#2ca02c", 0.5: "#ff7f0e", 1: "#d62728"}
key = lambda p, lr, s=0: f"p={p:g} lr={lr:g}" + (" seed=1" if s else "")

# 1. final loss vs p at each p's best lr
def panel1(a):
    for s, mk in [(0, "o"), (1, "s")]:
        a.plot(P, [R[key(p, BEST[p], s)]["val"] for p in P], mk, color="gray", alpha=.6, label=f"seed {s}")
    mean = [np.mean([R[key(p, BEST[p], s)]["val"] for s in (0, 1)]) for p in P]
    a.plot(P, mean, "k-", lw=2.5, label="mean of 2 seeds")
    a.annotate("Muon: stretches all\nsingular values\nto the same size", (0, mean[0]), (0.06, 1.682), arrowprops=dict(arrowstyle="->"))
    a.annotate("singular values left\nas in the gradient", (1, mean[3]), (0.5, 1.688), arrowprops=dict(arrowstyle="->"))
    a.set(xlabel="p   (singular values S become S^p)", ylabel="final validation loss (lower = better)",
          title="1. Loss vs p (each p at its best learning rate)")
    a.legend()


# 2. final loss vs learning rate, one line per p
def panel2(a):
    for p in P:
        a.plot(LR, [R[key(p, lr)]["val"] for lr in LR], "o-", color=COL[p], label=f"p = {p:g}")
    a.set_xscale("log"); a.set_xticks(LR); a.set_xticklabels([f"{x:g}" for x in LR]); a.minorticks_off()
    a.set(xlabel="learning rate", ylabel="final validation loss", title="2. Loss vs learning rate (seed 0)")
    a.legend()


def curve(p):
    cs = [np.array(R[key(p, BEST[p], s)]["curve"]) for s in (0, 1)]
    return cs[0][:, 0], np.mean([c[:, 1] for c in cs], axis=0)

# 3. training loss vs step at best lr (mean of 2 seeds)
def panel3(a):
    for p in P:
        x, y = curve(p)
        a.plot(x, y, color=COL[p], label=f"p = {p:g}")
    a.set(xlabel="training step", ylabel="training loss (one batch, mean of 2 seeds)", ylim=(1.3, 3.0),
          title="3. Training loss vs step")
    a.axvline(500, color="gray", ls=":"); a.text(510, 2.9, "learning rate starts\ndecaying", fontsize=9, va="top")
    a.legend()


# 4. gap to p = 0 vs step
def panel4(a):
    x0, y0 = curve(0)
    for p in P[1:]:
        x, y = curve(p)
        a.plot(x, y - y0, color=COL[p], label=f"p = {p:g}  minus  p = 0")
    a.axhline(0, color="k", lw=.8)
    a.set(xlabel="training step", ylabel="loss difference (above 0 = worse than p = 0)",
          title="4. Gap to equal push vs step")
    a.legend()


# 5. real gradient strengths, sorted
def panel5(a):
    for st, c in zip(KEEP, ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]):
        a.plot(np.arange(1, len(S[st]) + 1), S[st], color=c, label=f"step {st}")
    a.set_yscale("log")
    a.set(xlabel="direction rank (1 = strongest)", ylabel="strength (update divided by its size)",
          title="5. Real strengths of one update (layer h1)")
    a.legend()


# 6. push each direction gets, for one real update (step 400)
def panel6(a):
    s = S[400]
    for p in P:
        w = s ** p; w = w * np.sqrt(len(s)) / np.linalg.norm(w)
        a.plot(s, w, color=COL[p], lw=2, label=f"p = {p:g}")
    a.set_xscale("log"); a.set_yscale("log")
    a.set(xlabel="gradient strength of a direction (step 400, h1)", ylabel="push the update gives it",
          title="6. Push vs strength (same total size for every p)")
    a.legend()


PANELS = [panel1, panel2, panel3, panel4, panel5, panel6]
fig, ax = plt.subplots(2, 3, figsize=(17, 10))
for f, a in zip(PANELS, ax.flat):
    f(a)
fig.tight_layout()
out = os.path.join(HERE, "..", "images", "exp2_explore.png")
fig.savefig(out, dpi=110)
print(out)
for i, f in enumerate(PANELS, 1):                 # each chart also as its own square image
    with plt.rc_context({"font.size": 13}):
        fig, a = plt.subplots(figsize=(7, 7))
        f(a)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "..", "images", f"exp2_chart{i}.png"), dpi=160)
        plt.close(fig)
s = S[400]
for st in KEEP:
    v = S[st]
    print(f"step {st}: strongest {v[0]:.3f}, median {np.median(v):.4f}, weakest {v[-1]:.2e}, "
          f"strongest/median {v[0]/np.median(v):.0f}x, top 10 directions hold {np.sum(v[:10]**2):.0%} of the update")
for p in P:
    w = s ** p; w = w * np.sqrt(len(s)) / np.linalg.norm(w)
    print(f"p={p:g}: push strongest {w[0]:.2f}, median {np.median(w):.3f}, weakest {w[-1]:.2e}")
