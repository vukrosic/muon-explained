"""Continual learning on modular arithmetic: train tasks one after another, measure every task after every phase.

A task is an op and a relabeling: "+0" is a + b mod N; "+1" is the same pairs with every answer renamed by a random
permutation (adversarial change: same inputs, new answers). "+q25": only a random 25% of the answers are renamed (each
of them moves); the other 75% keep their answer, and "+0|+q25" logs how many of those unchanged rows of "+0" survive.
Each phase trains on one task only, for PHASE steps, at a constant learning rate (no restart, optimizer state kept).
Every EVERY steps: held-out accuracy on every task in the list.
Same model and optimizers as arith.py, except Muon uses NS Newton-Schulz rounds (NS=0: exact SVD as in arith.py).
Usage: TASKS=+0,-0,*0,+0 OPT=adam python3 cl.py SEED   -> one line in cl.jsonl (OUT)
LR_MULT: learning rate as a share of arith.py's peak. CLIP=c: clip total gradient norm (both optimizers).
REPLAY=r: share of each batch drawn from earlier tasks (0 = none).
CBP=r: continual backprop (Dohare et al. 2024): every step replace a share r of each hidden layer's units, the
  lowest-utility ones older than 100 steps (new input weights from the initial distribution, output weights 0).
WD2=w: weight decay from the second phase on (default: the same WD as the first phase).
SP=a,s: shrink and perturb (Ash & Adams 2020) at every task change: w <- a*w + s*(fresh initial weights).
At every phase end: dead units, mean |w|, effective rank of each hidden layer's features, size and effective rank of
each layer's change over the phase, and (Ramasesh et al. 2021) the previous task's accuracy when single layers or the
top layers are put back to their values at the end of the previous phase.
"""
import copy, json, math, os, sys, time
import torch, torch.nn.functional as F
import arith as A

SEED = int(sys.argv[1])
TASKS = os.environ.get("TASKS", "+0,-0,*0,/0").split(",")
PHASE, EVERY = int(os.environ.get("PHASE", 2000)), int(os.environ.get("EVERY", 100))
LR_MULT, CLIP = float(os.environ.get("LR_MULT", 1)), float(os.environ.get("CLIP", 0))
REPLAY = float(os.environ.get("REPLAY", 0))
CBP = float(os.environ.get("CBP", 0)); ETA, MATURE = 0.99, 100
SP = [float(v) for v in os.environ["SP"].split(",")] if os.environ.get("SP") else None
WD2 = float(os.environ.get("WD2", A.WD))
NS = int(os.environ.get("NS", 5))                            # Muon: Newton-Schulz rounds (quintic, Keller's coefficients); 0 = exact SVD
SUB = 1000                                                   # training rows per task used for the logged train accuracy
OUT = os.environ.get("OUT", "cl.jsonl")
N = A.N

torch.manual_seed(SEED); gen = torch.Generator().manual_seed(SEED)
base_tr, base_te = A.data(A.SYMS[:4], gen)                 # per op in "+-*/" order: train / held-out rows
perms = {"0": torch.arange(N)}
names = list(dict.fromkeys(TASKS))                           # distinct tasks, first-seen order


def perm(k):
    if k not in perms:
        g = torch.Generator().manual_seed(1000 + int(k[1:]) if k[0] == "q" else 1000 + int(k))
        if k[0] == "q":                                      # move a random share of the answers, each to another one of them
            S = torch.randperm(N, generator=g)[: round(N * int(k[1:]) / 100)]
            p = torch.arange(N); p[S] = S.roll(1)
        else:
            p = torch.randperm(N, generator=g)
        perms[k] = p
    return perms[k]


def task(name):
    op, k = name[0], name[1:]
    i = A.SYMS.index(op)
    tr, te = base_tr[i].clone(), base_te[i].clone()
    tr[:, 3], te[:, 3] = perm(k)[tr[:, 3]], perm(k)[te[:, 3]]
    return tr, te


data = {n: task(n) for n in names}
keep = {f"{n[0]}0|{n}": base_te[A.SYMS.index(n[0])][perm(n[1:])[base_te[A.SYMS.index(n[0])][:, 3]] == base_te[A.SYMS.index(n[0])][:, 3]]
        for n in names if n[1] == "q"}                      # held-out rows of the original op whose answer the relabel keeps


def ortho(G):
    if NS == 0:
        return A.shaped(G, 0)
    X = G / (G.norm() + 1e-7)
    for _ in range(NS):
        S = X @ X.T
        X = 3.4445 * X + (-4.7750 * S + 2.0315 * S @ S) @ X
    return X * (min(G.shape) ** 0.5 / (X.norm() + 1e-12))  # same total size as exact Muon


def acc(net, rows):
    with torch.no_grad():
        return (net(rows[:, :3]).argmax(1) == rows[:, 3]).float().mean().item()


def snap(ph):
    return dict(step=step, phase=ph, test=[round(acc(m, data[n][1]), 4) for n in names],
                train=[round(acc(m, data[n][0][:SUB]), 4) for n in names], keep={k: round(acc(m, r), 4) for k, r in keep.items()})


def feats(net, x):
    hs = [A.ACT(net.inp(net.emb(x).flatten(1)))]
    hs.append(A.ACT(net.h1(hs[-1]))); hs.append(A.ACT(net.h2(hs[-1])))
    return net.out(hs[-1]), hs


def erank(M):                                                # exp(entropy) of the normalized singular values
    s = torch.linalg.svdvals(M.double())
    p = s / s.sum().clamp_min(1e-30)
    return math.exp(-(p * p.clamp_min(1e-30).log()).sum().item())


def fresh(w):                                                # nn.Linear's initial distribution
    b = 1 / w.shape[1] ** 0.5
    return torch.empty_like(w).uniform_(-b, b)


m = A.Net()
muon = [m.inp.weight, m.h1.weight, m.h2.weight]
adam = torch.optim.AdamW([m.emb.weight, m.out.weight, m.out.bias], lr=A.ADAM_LR * LR_MULT, weight_decay=0)
hid = torch.optim.AdamW(muon, lr=A.HID_LR * LR_MULT, weight_decay=A.WD) if A.OPT == "adam" else None
mom = [torch.zeros_like(w) for w in muon]
LAYERS, NEXT = [m.inp, m.h1, m.h2], [m.h1, m.h2, m.out]
util, fmean, age, due = ([torch.zeros(256) for _ in LAYERS] for _ in range(4))
replaced = 0
GROUPS = {"emb": ["emb.weight"], "inp": ["inp.weight"], "h1": ["h1.weight"], "h2": ["h2.weight"], "out": ["out.weight", "out.bias"]}


def clear(w, j, dim):                                        # optimizer state of the replaced weights back to 0
    for opt in (adam, hid):
        if opt and w in opt.state:
            for v in opt.state[w].values():
                if torch.is_tensor(v) and v.shape == w.shape:
                    v.index_fill_(dim, j, 0)
    if hid is None:
        for p, buf in zip(muon, mom):
            if p is w:
                buf.index_fill_(dim, j, 0)


def cbp(hs):
    global replaced
    for l, (L, Nx) in enumerate(zip(LAYERS, NEXT)):
        h = hs[l]
        age[l] += 1
        fmean[l] = ETA * fmean[l] + (1 - ETA) * h.mean(0)
        f = fmean[l] / (1 - ETA ** age[l])
        y = (h - f).abs().mean(0) * Nx.weight.abs().sum(0) / L.weight.abs().sum(1).clamp_min(1e-12)
        util[l] = ETA * util[l] + (1 - ETA) * y
        due[l] += CBP * 256
        k = int(due[l][0])
        if not k:
            continue
        due[l] -= k
        ok = (age[l] > MATURE).nonzero().flatten()
        if not len(ok):
            continue
        j = ok[(util[l][ok] / (1 - ETA ** age[l][ok])).argsort()[:k]]
        L.weight[j] = fresh(L.weight)[j]; Nx.weight[:, j] = 0
        util[l][j] = fmean[l][j] = age[l][j] = 0
        clear(L.weight, j, 0); clear(Nx.weight, j, 1)
        replaced += len(j)


def diagnose(ph, start, prev_end, prev_task):
    name = TASKS[ph]
    x = data[name][1][:2000, :3]
    with torch.no_grad():
        _, hs = feats(m, data[name][1][:, :3])
        dead = (hs[1] <= 0).all(0).float().mean().item()
        _, hs = feats(m, x)
        now = {k: v.detach().clone() for k, v in m.state_dict().items()}
        d = dict(dead_h1=round(dead, 3), dead=[round((h <= 0).all(0).float().mean().item(), 3) for h in hs],
                 wmag=round(torch.cat([w.abs().flatten() for w in muon + [m.out.weight]]).mean().item(), 5),
                 feat_erank=[round(erank(h), 1) for h in hs], upd={})
        for g in ["inp", "h1", "h2", "out"]:
            D = now[f"{g}.weight"] - start[f"{g}.weight"]
            s = torch.linalg.svdvals(D.double())
            d["upd"][g] = dict(fro=round(D.norm().item(), 3), spec=round(s[0].item(), 3), erank=round(erank(D), 1),
                               rel=round((D.norm() / start[f"{g}.weight"].norm()).item(), 3))
        if prev_task is not None and prev_task != name:      # put layers back to their end-of-previous-phase values
            rows = data[prev_task][1]
            def back(gs):
                net = copy.deepcopy(m)
                net.load_state_dict({**now, **{k: prev_end[k] for g in gs for k in GROUPS[g]}})
                return round(acc(net, rows), 4)
            d["reset_one"] = {g: back([g]) for g in GROUPS}
            d["reset_top"] = {"+".join(gs): back(gs) for gs in (["out"], ["out", "h2"], ["out", "h2", "h1"], ["out", "h2", "h1", "inp"])}
    return d


log, ends = [], []
t0 = time.perf_counter()
step = 0
prev_end, start = None, {k: v.detach().clone() for k, v in m.state_dict().items()}
for ph, name in enumerate(TASKS):
    tr = data[name][0]
    old = [data[n][0] for n in dict.fromkeys(TASKS[:ph]) if n != name]
    wd = A.WD if ph == 0 else WD2
    if hid:
        hid.param_groups[0]["weight_decay"] = wd
    if SP and ph:
        with torch.no_grad():
            for w in muon + [m.out.weight]:
                w.copy_(SP[0] * w + SP[1] * fresh(w))
    for s in range(PHASE):
        if step % EVERY == 0:
            log.append(snap(ph))
        nr = round(512 * REPLAY) if old else 0
        x = tr[torch.randint(len(tr), (512 - nr,), generator=gen)]
        if nr:
            pool = torch.cat(old)
            x = torch.cat([x, pool[torch.randint(len(pool), (nr,), generator=gen)]])
        logits, hs = feats(m, x[:, :3])
        loss = F.cross_entropy(logits, x[:, 3])
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
                    w.mul_(1 - wd * lr)
                    w.add_(ortho(w.grad + A.BETA * buf), alpha=-lr)
        if CBP:
            with torch.no_grad():
                cbp([h.detach() for h in hs])
        step += 1
    d = diagnose(ph, start, prev_end, TASKS[ph - 1] if ph else None)
    ends.append(dict(phase=ph, task=name, test=[round(acc(m, data[n][1]), 4) for n in names],
                     train=[round(acc(m, data[n][0][:SUB]), 4) for n in names], **d))
    prev_end = start = {k: v.detach().clone() for k, v in m.state_dict().items()}
    print(f"{A.OPT} s{SEED} phase {ph} {name}: " + " ".join(f"{n} {a:.2f}" for n, a in zip(names, ends[-1]["test"]))
          + f"  dead {d['dead_h1']:.2f}  reset {d.get('reset_top', '')}", flush=True)
log.append(snap(len(TASKS) - 1))
print(f"[{time.perf_counter() - t0:.0f} s]")
with open(os.path.join(A.HERE, OUT), "a") as f:
    f.write(json.dumps(dict(opt=A.OPT, seed=SEED, tasks=TASKS, names=names, phase=PHASE, lr_mult=LR_MULT, clip=CLIP,
                            replay=REPLAY, cbp=CBP, sp=SP, replaced=replaced, ns=NS, wd=A.WD, wd2=WD2,
                            hid_lr=A.HID_LR if A.OPT == "adam" else None, act=os.environ.get("ACT", "relu"), ends=ends, log=log)) + "\n")
