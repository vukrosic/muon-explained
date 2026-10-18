"""Forgetting: take a trained model (saved by arith.py with SAVE) and keep training with some ops removed from the batch.
Does an op survive when its partner is still trained?

Same optimizers as arith.py, fresh optimizer state, constant learning rate = LR_MULT x the peak, weight decay WD.
Usage: FREQ2=0.48,0.48,0,0.04 OPT=adam python3 forget.py MODEL.pt LABEL   -> forget.jsonl
SAVE2=path keeps the result for a later relearn run. LIMIT=i:K trains op i on only its first K training pairs (relearning).
FOUT names the jsonl. WARM=W: learning rate rises linearly from 0 over the first W steps. DIAG=i logs h1 effective rank and dead units on op i. ADD_AT=i:D keeps op i out of the batch until step D. RESET=emb,out,hid puts those parts back to a fresh init first. SP=lam,sig: shrink and perturb first, W = lam W + sig W_fresh on the 3 hidden matrices. KICK=K (OPT=adam): every K-th step the hidden matrices take one Muon step (lr MUON_LR x KICK_LR) instead of Adam. B2, EPS: Adam beta2 and eps (all Adam groups). CLIP=c: clip the total gradient norm to c. SCHED=decay: lr as in arith.py (full for half the steps, then linear to 0) instead of constant.
"""
import json, os, sys
import torch, torch.nn.functional as F
import arith as A

path, label = sys.argv[1], sys.argv[2]
FREQ2 = [float(f) for f in os.environ["FREQ2"].split(",")]
STEPS2, LR_MULT = int(os.environ.get("STEPS2", 3000)), float(os.environ.get("LR_MULT", 0.5))
ck = torch.load(path)
tr, te, order, seed = ck["tr"], ck["te"], ck["order"], ck["seed"]
BS = seed + 100 + int(os.environ.get("BSEED", 0))          # BSEED: another batch order for the same model
torch.manual_seed(BS); gen = torch.Generator().manual_seed(BS)
m = A.Net(); m.load_state_dict(ck["model"])
full_tr = tr
if os.environ.get("RESET"):
    fresh = A.Net(); parts = os.environ["RESET"].split(",")
    names = [n for p_ in parts for n in {"emb": ["emb"], "out": ["out"], "hid": ["inp", "h1", "h2"]}[p_]]
    for n in names:
        getattr(m, n).load_state_dict(getattr(fresh, n).state_dict())
if os.environ.get("SP"):
    lam, sig = map(float, os.environ["SP"].split(","))
    fresh = A.Net()
    with torch.no_grad():
        for n in ("inp", "h1", "h2"):
            getattr(m, n).weight.mul_(lam).add_(sig * getattr(fresh, n).weight)
if os.environ.get("LIMIT"):
    i_lim, k_lim = map(int, os.environ["LIMIT"].split(":"))
    tr = [t[:k_lim] if i == i_lim else t for i, t in enumerate(tr)]
muon = [m.inp.weight, m.h1.weight, m.h2.weight]
B2, EPS, CLIP = float(os.environ.get("B2", 0.999)), float(os.environ.get("EPS", 1e-8)), float(os.environ.get("CLIP", 0))
adam = torch.optim.AdamW([m.emb.weight, m.out.weight, m.out.bias], lr=A.ADAM_LR * LR_MULT, weight_decay=0, betas=(0.9, B2), eps=EPS)
hid = torch.optim.AdamW(muon, lr=A.HID_LR * LR_MULT, weight_decay=A.WD, betas=(0.9, B2), eps=EPS) if A.OPT == "adam" else None
mom = [torch.zeros_like(w) for w in muon]
freq = torch.tensor(FREQ2)
DECAYING = os.environ.get("SCHED") == "decay"
WARM = int(os.environ.get("WARM", 0))
KICK, KICK_LR = int(os.environ.get("KICK", 0)), float(os.environ.get("KICK_LR", 1))   # OPT=adam: every KICK-th step is one Muon step
DIAG_OP = int(os.environ.get("DIAG", -1)); DIAG = DIAG_OP >= 0
ADD_AT = tuple(map(int, os.environ["ADD_AT"].split(":"))) if os.environ.get("ADD_AT") else None
log, gstats = [], []
for step in range(STEPS2 + 1):
    if step % A.EVERY == 0 or step in (10, 20, 30):
        log.append(dict(step=step, train=[round(A.acc(m, r), 4) for r in full_tr], test=[round(A.acc(m, r), 4) for r in te]))
        if DIAG:                                      # h1: effective rank of the weight, share of units dead on op DIAG's inputs
            with torch.no_grad():
                sv = torch.linalg.svdvals(m.h1.weight); q = sv / sv.sum()
                xd = te[DIAG_OP][:, :3]; h = F.relu(m.h1(F.relu(m.inp(m.emb(xd).flatten(1)))))
            log[-1].update(erank=round(torch.exp(-(q * q.log()).sum()).item(), 1), dead=round((h.amax(0) <= 0).float().mean().item(), 3))
    if step == STEPS2:
        break
    if ADD_AT and step < ADD_AT[1]:
        f_now = freq.clone(); f_now[ADD_AT[0]] = 0
    else:
        f_now = freq
    k = torch.multinomial(f_now, A.B, replacement=True, generator=gen)
    x = torch.cat([tr[i][torch.randint(len(tr[i]), (int((k == i).sum()),), generator=gen)] for i in range(len(tr))])
    loss = F.cross_entropy(m(x[:, :3]), x[:, 3])
    m.zero_grad(); loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP if CLIP else float("inf")).item()
    gstats.append(gnorm)
    sched = min(1.0, (1 - step / STEPS2) / A.DECAY) if DECAYING else 1.0
    if WARM:
        sched *= min(1.0, (step + 1) / WARM)
    for gr in adam.param_groups:
        gr["lr"] = A.ADAM_LR * LR_MULT * sched
    adam.step()
    if hid and not (KICK and step % KICK == KICK - 1):
        for gr in hid.param_groups:
            gr["lr"] = A.HID_LR * LR_MULT * sched
        hid.step(); continue
    if hid:                                           # Muon kick: this step the hidden matrices get one Muon update (no momentum)
        with torch.no_grad():
            for w in muon:
                w.add_(A.shaped(w.grad, 0), alpha=-A.MUON_LR * KICK_LR * sched)
        continue
    with torch.no_grad():
        for w, buf in zip(muon, mom):
            buf.mul_(A.BETA).add_(w.grad)
            w.mul_(1 - A.WD * A.MUON_LR * LR_MULT * sched)
            w.add_(A.shaped(w.grad + A.BETA * buf, 0), alpha=-A.MUON_LR * LR_MULT * sched)
e = log[-1]
print(f"{label:>34} {order} {A.OPT} freq2 {FREQ2}  test " + " ".join(f"{op}:{t:.2f}" for op, t in zip(order, e["test"])), flush=True)
if os.environ.get("SAVE2"):
    torch.save(dict(model=m.state_dict(), tr=full_tr, te=te, order=order, seed=seed), os.environ["SAVE2"])
with open(os.path.join(A.HERE, os.environ.get("FOUT", "forget.jsonl")), "a") as f:
    f.write(json.dumps(dict(model=os.path.basename(path), label=label, order=order, seed=seed, opt=A.OPT, freq2=FREQ2, steps2=STEPS2,
                            lr_mult=LR_MULT, sched="decay" if DECAYING else "const", wd=A.WD, limit=os.environ.get("LIMIT"), bseed=int(os.environ.get("BSEED", 0)), sp=os.environ.get("SP"), reset=os.environ.get("RESET"), add_at=os.environ.get("ADD_AT"), warm=WARM, kick=KICK, kick_lr=KICK_LR, b2=B2, eps=EPS, clip=CLIP, n_clipped=sum(g > CLIP for g in gstats) if CLIP else None,
                            gnorm_pct=[round(sorted(gstats)[int(q * (len(gstats) - 1))], 3) for q in (0.5, 0.9, 0.99, 1.0)], log=log)) + "\n")
