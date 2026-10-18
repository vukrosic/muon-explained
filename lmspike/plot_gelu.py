"""images/lm_gelu.png: GELU model, embedding lr 0.01. Left: stable rank of the last hidden matrix over the first 300 steps (seed 0).
Right: validation loss when the first N hidden steps use Adam and the rest Muon (3-seed means)."""
import json, os
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for f in ("lmact.jsonl", "lmgswitch.jsonl", "lmgstat.jsonl") for l in open(os.path.join(HERE, f))]
g = [r for r in rows if r["act"] == "gelu" and r["emb_lr"] == 0.01 and not r.get("cap")]
def pick(**k): return [r for r in g if all(r.get(a) == b for a, b in k.items()) and r.get("early_n", 0) == k.get("early_n", 0)]
muon = pick(opt="muon", switch=0, hid_lr=0.03); adam = pick(opt="adam", switch=0, hid_lr=0.003, warm_hid=0)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for runs, c, label in ((adam, "C3", "Adam"), (muon, "C0", "Muon"), (pick(opt="adam", opt2="muon", switch=10, muon_lr=0.03), "C1", "10 Adam steps, then Muon")):
    r = [x for x in runs if x.get("wstat") and x["seed"] == 0][0]
    w = [x for x in r["wstat"] if x[0] <= 300]
    ax[0].plot([x[0] for x in w], [x[3][0] ** 2 / x[3][1] ** 2 for x in w], color=c, label=label)
ax[0].set_yscale("log"); ax[0].set_xlabel("step"); ax[0].set_ylabel("stable rank of last hidden matrix (512 max)"); ax[0].legend()
def mean(rs):
    one = {r["seed"]: r["val"] for r in rs}                 # seed-0 runs were repeated with weight logging; same values, count once
    return sum(one.values()) / len(one)
ns = [1, 3, 10, 25, 100]
vals = [mean(pick(opt="adam", opt2="muon", switch=n, muon_lr=0.03)) for n in ns]
ax[1].plot(ns, vals, "o-", color="C1", label="first N steps Adam, then Muon")
ax[1].axhline(mean(muon), color="C0", ls="--", label=f"Muon all the way {mean(muon):.3f}")
ax[1].axhline(mean(adam), color="C3", ls="--", label=f"Adam all the way {mean(adam):.3f}")
tiny = mean(pick(opt="adam", opt2="muon", switch=10, muon_lr=0.03, early_n=10))
ax[1].plot([10], [tiny], "s", color="C2", ms=9, label=f"10 Adam steps at 1/30 lr, then Muon {tiny:.3f}")
ax[1].set_xscale("log"); ax[1].set_xlabel("N Adam steps before switching to Muon"); ax[1].set_ylabel("validation loss (3 seeds)"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "images", "lm_gelu.png"), dpi=120)
print([round(v, 4) for v in vals], round(tiny, 4), round(mean(muon), 4), round(mean(adam), 4))
