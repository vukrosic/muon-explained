"""Is it the burst's weak directions? Each variant = phase-1 model m1 + a change on every Muon weight matrix (attention, FF),
then a fresh AdamW on the same phase-2 batches (steps 100..500), as in cause.py. Embeddings and norms stay at m1.
dB = burst change, dA = change from 100 AdamW steps; per matrix, every variant except restart is scaled to ||dB||.
  restart       100 AdamW steps, then fresh AdamW (no burst)
  burst         m1 + dB
  adam scaled   m1 + dA * ||dB|| / ||dA||                 AdamW's own directions, burst size
  adam flat     m1 + (dA with all singular values equal)  AdamW's directions, spread evenly like Muon
  burst strong  m1 + top-K singular directions of dB
  burst weak    m1 + dB without its top-K directions
usage: MUON_LR=0.012 python3 weak.py SEED       env K (default 5), OUT (weak.jsonl)
"""
import copy, json, os, sys, torch, torch.nn.functional as F
seed = int(sys.argv[1]); sys.argv = sys.argv[:1]
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "tinylm.py")).read().split("if __name__")[0])
ds = int(DSEED) if DSEED else seed
TOTAL, STOPS, K = E("P2", 700), [200, 300, 400, 500], E("K", 5)
NEW_F = [(1 - FREQ_NEW) / 2, (1 - FREQ_NEW) / 2, FREQ_NEW]
MUON_W = lambda n: not ("embedding" in n or "norm" in n)


def adam(m, g, steps, total=TOTAL, t0=0, log=None):   # same as cause.py: tinylm train(), opt="adam", lr LR2
    opt = torch.optim.AdamW(list(m.parameters()), lr=LR2, weight_decay=WD)
    for i in range(steps):
        sc = min(1, 2 * (1 - i / total)) * min(1, (i + 1) / WARM)
        s = mixed(BS, "CSR", NEW_F, g)
        loss = F.cross_entropy(answer_logits(m, s).reshape(-1, V), s[:, L + 2:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), CFG.grad_clip)
        for gr in opt.param_groups: gr["lr"] = LR2 * sc
        opt.step()
        if log is not None and t0 + i + 1 in STOPS: log[t0 + i + 1] = evaluate(m, torch.Generator().manual_seed(999))["R"]


torch.manual_seed(seed); m1 = MinimalLLM(CFG)
train(m1, P1, "CS", [0.5, 0.5], torch.Generator().manual_seed(ds), "adam", [], 0, torch.Generator().manual_seed(999))
g2 = torch.Generator().manual_seed(ds + 100)
mb = copy.deepcopy(m1); train(mb, BURST, "CSR", NEW_F, g2, "muon", [], 0, torch.Generator().manual_seed(999))
state = g2.get_state()                               # every variant sees the same batches from step 100 on
ma = copy.deepcopy(m1); g3 = torch.Generator().manual_seed(ds + 100); adam(ma, g3, BURST)
P1d, PB, PA = dict(m1.named_parameters()), dict(mb.named_parameters()), dict(ma.named_parameters())


def variant(fn):   # fn(dB, dA) -> change for one Muon matrix
    m = copy.deepcopy(m1)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if MUON_W(n): p.add_(fn((PB[n] - P1d[n]).detach(), (PA[n] - P1d[n]).detach()))
    return m


def svd(d): return torch.linalg.svd(d, full_matrices=False)
def to_norm(x, ref): return x * ref.norm() / x.norm()
def strong(dB): U, S, Vh = svd(dB); return to_norm(U[:, :K] @ torch.diag(S[:K]) @ Vh[:K], dB)
def weak(dB): U, S, Vh = svd(dB); return to_norm(U[:, K:] @ torch.diag(S[K:]) @ Vh[K:], dB)
def flat(dA, dB): U, S, Vh = svd(dA); return to_norm(U @ Vh, dB)

starts = {"burst": variant(lambda dB, dA: dB), "adam scaled": variant(lambda dB, dA: to_norm(dA, dB)),
          "adam flat": variant(lambda dB, dA: flat(dA, dB)), "burst strong": variant(lambda dB, dA: strong(dB)),
          "burst weak": variant(lambda dB, dA: weak(dB))}
res = {"restart": {}}
g3.set_state(state); adam(ma, g3, STOPS[-1] - BURST, t0=BURST, log=res["restart"])
for name, m in starts.items():
    g2.set_state(state); res[name] = {}; adam(m, g2, STOPS[-1] - BURST, t0=BURST, log=res[name])
with open(os.path.join(HERE, os.environ.get("OUT", "weak.jsonl")), "a") as fh:
    fh.write(json.dumps(dict(seed=seed, dseed=ds, k=K, freq_new=FREQ_NEW, muon_lr=MUON_LR, lr2=LR2, total=TOTAL, res=res)) + "\n")
print(seed, {k: round(sum(v.values()) / 4, 2) for k, v in res.items()}, flush=True)
