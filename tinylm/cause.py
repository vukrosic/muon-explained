"""Which part of the 100-step Muon burst makes the following AdamW learn reverse fast?

From the phase-1 model m1 and the burst model mb (= tinylm burst arm at step 100), build hybrids, then train each with a fresh
AdamW on the same phase-2 batches (steps 100..500 of the tinylm burst arm) and log reverse exact match.
  restart      100 AdamW steps, then fresh AdamW (the no-burst control)
  nothing      m1 unchanged
  burst        mb
  only G       m1 + the burst's change on group G
  reset G      mb with group G put back to m1
  kick         m1 + random change on each Muon matrix, same Frobenius norm as the burst's change
  respin       m1 + burst change with the same singular values but random singular vectors (per Muon matrix)
SET=qkvo instead runs restart, burst and only/reset of the attention QK rows and VO rows (both layers, and layer 1 alone).
usage: MUON_LR=0.012 python3 cause.py SEED      (DSEED defaults to SEED, as in tinylm)
"""
import copy, json, os, sys, torch, torch.nn.functional as F
seed = int(sys.argv[1]); sys.argv = sys.argv[:1]
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "tinylm.py")).read().split("if __name__")[0])
ds = int(DSEED) if DSEED else seed
TOTAL, STOPS = E("P2", 700), [200, 300, 400, 500]
NEW_F = [(1 - FREQ_NEW) / 2, (1 - FREQ_NEW) / 2, FREQ_NEW]
GROUPS = {"emb+norms": lambda n: "embedding" in n or "norm" in n,
          "attn0": lambda n: n.startswith("transformer_blocks.0.attention.qkvo"),
          "attn1": lambda n: n.startswith("transformer_blocks.1.attention.qkvo"),
          "ff0": lambda n: n.startswith("transformer_blocks.0.feed_forward"),
          "ff1": lambda n: n.startswith("transformer_blocks.1.feed_forward")}
MUON_W = lambda n: not GROUPS["emb+norms"](n)


def adam(m, g, steps, total=TOTAL, t0=0, log=None):   # tinylm train(), opt="adam", lr LR2; logs R at t0 + i in STOPS
    opt = torch.optim.AdamW(list(m.parameters()), lr=LR2, weight_decay=WD)
    for i in range(steps):
        sc = min(1, 2 * (1 - i / total)) * min(1, (i + 1) / WARM)
        s = mixed(BS, "CSR", NEW_F, g)
        loss = F.cross_entropy(answer_logits(m, s).reshape(-1, V), s[:, L + 2:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), CFG.grad_clip)
        for gr in opt.param_groups: gr["lr"] = LR2 * sc
        opt.step()
        if log is not None and t0 + i + 1 in STOPS: log[t0 + i + 1] = evaluate(m, torch.Generator().manual_seed(999))["R"]


def hybrid(fn):   # fn(name, p1, pb) -> new weight
    m = copy.deepcopy(m1); P1d, PBd = dict(m1.named_parameters()), dict(mb.named_parameters())
    with torch.no_grad():
        for n, p in m.named_parameters(): p.copy_(fn(n, P1d[n], PBd[n]))
    return m


def rand_orth(k, gen): return torch.linalg.qr(torch.randn(k, k, generator=gen))[0]


torch.manual_seed(seed); g = torch.Generator().manual_seed(ds)
m1 = MinimalLLM(CFG)
train(m1, P1, "CS", [0.5, 0.5], g, "adam", [], 0, torch.Generator().manual_seed(999))
g2 = torch.Generator().manual_seed(ds + 100)
mb = copy.deepcopy(m1); train(mb, BURST, "CSR", NEW_F, g2, "muon", [], 0, torch.Generator().manual_seed(999))
state = g2.get_state()                                # every variant sees the same batches from step 100 on
rg = torch.Generator().manual_seed(seed + 7)

def kick(n, a, b):
    if not MUON_W(n): return a
    r = torch.randn(a.shape, generator=rg); return a + r * (b - a).norm() / r.norm()

def respin(n, a, b):
    if not MUON_W(n): return a
    U, S, Vh = torch.linalg.svd(b - a, full_matrices=False)
    k = S.numel(); return a + rand_orth(a.shape[0], rg)[:, :k] @ torch.diag(S) @ rand_orth(a.shape[1], rg)[:k]

def rows(layers, lo, hi):   # burst change only on rows lo:hi of qkvo_proj in `layers` (rows: Q, K, V, O blocks of d_model)
    def fn(n, a, b):
        if not any(n == f"transformer_blocks.{l}.attention.qkvo_proj" for l in layers): return a
        out = a.clone(); out[lo:hi] = b[lo:hi]; return out
    return fn

D = CFG.d_model
if os.environ.get("SET") == "qkvo":
    starts = {"burst": mb}
    for nm, ls in (("", (0, 1)), (" L1", (1,))):
        for part, (lo, hi) in (("QK", (0, 2 * D)), ("VO", (2 * D, 4 * D))):
            f = rows(ls, lo, hi)
            starts[f"only {part}{nm}"] = hybrid(f)
            starts[f"reset {part}{nm}"] = hybrid(lambda n, a, b, f=f: b - f(n, a, b) + a)
GROUPS = {} if os.environ.get("SET") == "qkvo" else GROUPS
if os.environ.get("SET") != "qkvo": starts = {"nothing": copy.deepcopy(m1), "burst": mb}
for G, f in GROUPS.items():
    starts[f"only {G}"] = hybrid(lambda n, a, b, f=f: b if f(n) else a)
    starts[f"reset {G}"] = hybrid(lambda n, a, b, f=f: a if f(n) else b)
if GROUPS: starts["kick"], starts["respin"] = hybrid(kick), hybrid(respin)
res = {}
m = copy.deepcopy(m1); g3 = torch.Generator().manual_seed(ds + 100); adam(m, g3, BURST)
g3.set_state(state); res["restart"] = {}; adam(m, g3, STOPS[-1] - BURST, t0=BURST, log=res["restart"])
for name, m in starts.items():
    g2.set_state(state); res[name] = {}; adam(m, g2, STOPS[-1] - BURST, t0=BURST, log=res[name])
with open(os.path.join(HERE, os.environ.get("OUT", "cause.jsonl")), "a") as fh:
    fh.write(json.dumps(dict(seed=seed, dseed=ds, freq_new=FREQ_NEW, muon_lr=MUON_LR, lr2=LR2, total=TOTAL, res=res)) + "\n")
print(seed, {k: v[500] for k, v in res.items()}, flush=True)
