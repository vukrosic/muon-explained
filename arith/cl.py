"""Continual learning on modular arithmetic: train tasks one after another, measure every task after every phase.

A task is an op and a relabeling: "+0" is a + b mod N; "+1" is the same pairs with every answer renamed by a random
permutation (adversarial change: same inputs, new answers). Each phase trains on one task only, for PHASE steps, at a
constant learning rate (no restart, optimizer state kept). Every EVERY steps: held-out accuracy on every task in the list.
Same model and optimizers as arith.py, except Muon uses NS Newton-Schulz rounds (NS=0: exact SVD as in arith.py).
Usage: TASKS=+0,-0,*0,+0 OPT=adam python3 cl.py SEED   -> one line in cl.jsonl (OUT)
LR_MULT: learning rate as a share of arith.py's peak. CLIP=c: clip total gradient norm (both optimizers).
REPLAY=r: share of each batch drawn from earlier tasks (0 = none).
"""
import json, os, sys, time
import torch, torch.nn.functional as F
import arith as A

SEED = int(sys.argv[1])
TASKS = os.environ.get("TASKS", "+0,-0,*0,/0").split(",")
PHASE, EVERY = int(os.environ.get("PHASE", 2000)), int(os.environ.get("EVERY", 100))
LR_MULT, CLIP = float(os.environ.get("LR_MULT", 1)), float(os.environ.get("CLIP", 0))
REPLAY = float(os.environ.get("REPLAY", 0))
NS = int(os.environ.get("NS", 5))                            # Muon: Newton-Schulz rounds (quintic, Keller's coefficients); 0 = exact SVD
SUB = 1000                                                   # training rows per task used for the logged train accuracy
OUT = os.environ.get("OUT", "cl.jsonl")
N = A.N

torch.manual_seed(SEED); gen = torch.Generator().manual_seed(SEED)
base_tr, base_te = A.data(A.SYMS[:4], gen)                 # per op in "+-*/" order: train / held-out rows
perms = {0: torch.arange(N)}
names = list(dict.fromkeys(TASKS))                           # distinct tasks, first-seen order


def task(name):
    op, k = name[0], int(name[1:])
    if k not in perms:
        perms[k] = torch.randperm(N, generator=torch.Generator().manual_seed(1000 + k))
    i = A.SYMS.index(op)
    tr, te = base_tr[i].clone(), base_te[i].clone()
    tr[:, 3], te[:, 3] = perms[k][tr[:, 3]], perms[k][te[:, 3]]
    return tr, te


data = {n: task(n) for n in names}


def ortho(G):
    if NS == 0:
        return A.shaped(G, 0)
    X = G / (G.norm() + 1e-7)
    for _ in range(NS):
        S = X @ X.T
        X = 3.4445 * X + (-4.7750 * S + 2.0315 * S @ S) @ X
    return X * (min(G.shape) ** 0.5 / (X.norm() + 1e-12))  # same total size as exact Muon


def snap(ph):
    return dict(step=step, phase=ph, test=[round(A.acc(m, data[n][1]), 4) for n in names],
                train=[round(A.acc(m, data[n][0][:SUB]), 4) for n in names])


m = A.Net()
muon = [m.inp.weight, m.h1.weight, m.h2.weight]
adam = torch.optim.AdamW([m.emb.weight, m.out.weight, m.out.bias], lr=A.ADAM_LR * LR_MULT, weight_decay=0)
hid = torch.optim.AdamW(muon, lr=A.HID_LR * LR_MULT, weight_decay=A.WD) if A.OPT == "adam" else None
mom = [torch.zeros_like(w) for w in muon]
log, ends = [], []
t0 = time.perf_counter()
step = 0
for ph, name in enumerate(TASKS):
    tr = data[name][0]
    old = [data[n][0] for n in dict.fromkeys(TASKS[:ph]) if n != name]
    for s in range(PHASE):
        if step % EVERY == 0:
            log.append(snap(ph))
        nr = round(512 * REPLAY) if old else 0
        x = tr[torch.randint(len(tr), (512 - nr,), generator=gen)]
        if nr:
            pool = torch.cat(old)
            x = torch.cat([x, pool[torch.randint(len(pool), (nr,), generator=gen)]])
        loss = F.cross_entropy(m(x[:, :3]), x[:, 3])
        m.zero_grad(); loss.backward()
        if CLIP:
            torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
        adam.step()
        if hid:
            hid.step()
        else:
            with torch.no_grad():
                lr = A.MUON_LR * LR_MULT
                for w, buf in zip(muon, mom):
                    buf.mul_(A.BETA).add_(w.grad)
                    w.mul_(1 - A.WD * lr)
                    w.add_(ortho(w.grad + A.BETA * buf), alpha=-lr)
        step += 1
    with torch.no_grad():                                   # h1 units that output 0 on every held-out input of this task
        h = A.ACT(m.inp(m.emb(data[name][1][:, :3]).flatten(1)))
        dead = (A.ACT(m.h1(h)) <= 0).all(0).float().mean().item()
    ends.append(dict(phase=ph, task=name, test=[round(A.acc(m, data[n][1]), 4) for n in names],
                     train=[round(A.acc(m, data[n][0][:SUB]), 4) for n in names], dead_h1=round(dead, 3)))
    print(f"{A.OPT} s{SEED} phase {ph} {name}: " + " ".join(f"{n} {a:.2f}" for n, a in zip(names, ends[-1]["test"])) + f"  dead {dead:.2f}", flush=True)
log.append(snap(len(TASKS) - 1))
print(f"[{time.perf_counter() - t0:.0f} s]")
with open(os.path.join(A.HERE, OUT), "a") as f:
    f.write(json.dumps(dict(opt=A.OPT, seed=SEED, tasks=TASKS, names=names, phase=PHASE, lr_mult=LR_MULT, clip=CLIP,
                            replay=REPLAY, ns=NS, wd=A.WD, hid_lr=A.HID_LR if A.OPT == "adam" else None, act=os.environ.get("ACT", "relu"), ends=ends, log=log)) + "\n")
