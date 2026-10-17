"""Where does the rare op's gradient sit in the update's spectrum, and how much of it does each shaping keep?

Train like arith.py (Muon p = 0, WD 1). At a few steps, for each Muon matrix: each op's gradient on 1024 of its training
pairs, weighted by the op's batch share (so the four add up to the expected batch gradient G). From G's SVD:
- share of the op's gradient energy that lies in G's strongest 10% of directions and in the weakest 50%;
- how fast one step of U S^p V^T (scaled to Muon's size) lowers that op's loss, relative to op 0 (most common):
  <O_p, G_op>/|G_op| divided by the same for op 0. 1 = the rare op gets as much descent per unit of its gradient as op 0.
Usage: FREQ=... python3 spectrum.py ORDER SEED   -> spectrum.jsonl
"""
import json, os, sys
import torch, torch.nn.functional as F
import arith as A

order, seed = sys.argv[1], int(sys.argv[2])
KEEP = (10, 100, 300, 600, 1000, 1500, 2000, 2999)
torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
tr, te = A.data(order, gen)
m = A.Net(); muon = [m.inp.weight, m.h1.weight, m.h2.weight]; names = ["inp", "h1", "h2"]
adam = torch.optim.AdamW([m.emb.weight, m.out.weight, m.out.bias], lr=A.ADAM_LR, weight_decay=0)
mom = [torch.zeros_like(w) for w in muon]; freq = torch.tensor(A.FREQ)
probe = [r[torch.randperm(len(r), generator=gen)[:1024]] for r in tr]


def op_grads():
    gs = []
    for i, r in enumerate(probe):
        m.zero_grad(); (A.FREQ[i] * F.cross_entropy(m(r[:, :3]), r[:, 3])).backward()
        gs.append([w.grad.double().clone() for w in muon])
    return gs


rows = []
for step in range(A.STEPS):
    if step in KEEP:
        gs = op_grads()
        for j, nm in enumerate(names):
            G = sum(g[j] for g in gs)
            try:
                U, s, Vh = torch.linalg.svd(G, full_matrices=False)
            except torch._C._LinAlgError:                              # LAPACK rarely fails to converge
                e, V = torch.linalg.eigh(G.T @ G); e, V = e.flip(0), V.flip(1)      # same factors via eigh, strong first
                s = e.clamp_min(0).sqrt().clamp_min(1e-12 * G.norm()); U, Vh = (G @ V) / s, V.T
            k10, k50 = max(1, len(s) // 10), len(s) // 2
            rec = dict(step=step, layer=nm, test=[round(A.acc(m, t), 3) for t in te])
            for i, g in enumerate(gs):
                c = U.T @ g[j] @ Vh.T                                  # op gradient in G's singular basis
                e = (c ** 2).sum(1); e = e / (e.sum() + 1e-30)                 # energy per left direction (strong first)
                rec[f"top10_{i}"] = round(e[:k10].sum().item(), 3); rec[f"bot50_{i}"] = round(e[k50:].sum().item(), 3)
            for p in (0, 0.5, 1):
                O = (U * s.clamp_min(1e-30) ** p) @ Vh; O = O / O.norm()
                d = [(O * g[j]).sum().item() / (g[j].norm().item() + 1e-30) for g in gs]
                rec[f"rel_p{p}"] = [round(x / d[0], 3) for x in d]
            rows.append(rec)
        print(step, [(r["layer"], r["top10_3"], r["bot50_3"], r["rel_p0"][3], r["rel_p0.5"][3], r["rel_p1"][3]) for r in rows[-3:]], flush=True)
    sched = min(1.0, (1 - step / A.STEPS) / A.DECAY)
    k = torch.multinomial(freq, A.B, replacement=True, generator=gen)
    x = torch.cat([tr[i][torch.randint(len(tr[i]), (int((k == i).sum()),), generator=gen)] for i in range(4)])
    loss = F.cross_entropy(m(x[:, :3]), x[:, 3]); m.zero_grad(); loss.backward()
    for gr in adam.param_groups: gr["lr"] = A.ADAM_LR * sched
    adam.step()
    with torch.no_grad():
        for w, buf in zip(muon, mom):
            buf.mul_(A.BETA).add_(w.grad); w.mul_(1 - A.WD * A.MUON_LR * sched)
            w.add_(A.shaped(w.grad + A.BETA * buf, 0), alpha=-A.MUON_LR * sched)
with open(os.path.join(A.HERE, "spectrum.jsonl"), "a") as f:
    f.write(json.dumps(dict(order=order, seed=seed, freq=A.FREQ, rows=rows)) + "\n")
