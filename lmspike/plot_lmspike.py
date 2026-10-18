"""Charts for the character-model start experiments -> ../images/lm_start.png"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for f in ("lmspike.jsonl", "lmwarm.jsonl", "lmswitch.jsonl", "lmemb.jsonl") for l in open(os.path.join(HERE, f))]


def pick(**kw):
    out = [r for r in rows if all(r.get(k, d) == v for k, (v, d) in kw.items())]
    return out


fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for label, sel, c in [("Adam, no warmup", dict(opt=("adam", None), hid_lr=(0.003, None), warm=(0, 0), switch=(0, 0), warm_hid=(0, 0), emb_lr=(0.003, None), clip=(0, None)), "#c0392b"),
                      ("Adam, warmup 100", dict(opt=("adam", None), hid_lr=(0.003, None), warm=(100, 0), switch=(0, 0), emb_lr=(0.003, None)), "#e67e22"),
                      ("Muon", dict(opt=("muon", None), switch=(0, 0), emb_lr=(0.003, None)), "#2471a3")]:
    for i, r in enumerate(sorted(pick(**sel), key=lambda r: r["seed"])[:3]):
        d = [x for x in r["dead"] if x[0] <= 300]
        ax[0].plot([x[0] for x in d], [100 * x[2] for x in d], color=c, alpha=0.8, label=label if i == 0 else None)
ax[0].set_xlabel("step"); ax[0].set_ylabel("% of layer-2 units dead (off on all 4096 inputs)")
ax[0].set_title("Adam kills units in the first 50 steps; they never return"); ax[0].legend()

ns = [0, 5, 10, 25, 100, 300]
means = []
for n in ns:
    if n == 0:
        rs = pick(opt=("adam", None), warm=(50, 0), switch=(0, 0), b2=(0.999, None), emb_lr=(0.003, None), hid_lr=(0.003, None), warm_hid=(0, 0))
    else:
        rs = pick(opt=("muon", None), switch=(n, 0), warm2=(50 if n < 300 else 0, 0),   # the switch at 300 needs no warmup
                   b2=(0.999, None), hid_lr=(0.003, None))
    means.append(sum(r["val"] for r in rs) / len(rs))
    ax[1].scatter([n] * len(rs), [r["val"] for r in rs], color="#2471a3", s=14, alpha=0.5)
ax[1].plot(ns, means, "o-", color="#2471a3", label="Muon for the first N steps, then Adam")
mu = pick(opt=("muon", None), switch=(0, 0), emb_lr=(0.003, None))
ax[1].axhline(sum(r["val"] for r in mu) / len(mu), color="#2471a3", ls=":", label="Muon all 1000 steps")
ad = pick(opt=("adam", None), warm_hid=(200, 0), emb_lr=(0.003, None), hid_lr=(0.003, None))
ax[1].axhline(sum(r["val"] for r in ad) / len(ad), color="#27ae60", ls="--", label="Adam, warmup on hidden layers only (200)")
ax[1].set_xscale("symlog", linthresh=5); ax[1].set_xticks(ns); ax[1].set_xticklabels(ns)
ax[1].set_xlabel("N"); ax[1].set_ylabel("validation loss (3 seeds)"); ax[1].set_title("Only the first 25 steps matter"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "..", "images", "lm_start.png"), dpi=130)
print([round(m, 4) for m in means])
