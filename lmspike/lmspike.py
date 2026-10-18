"""Does the arithmetic spike chain (Adam loss spike -> dead ReLU units -> rare data lost) also happen in the character model?

Same data and model as the speedrun (16 characters in, three 512-wide ReLU layers). Hidden matrices: Adam or Muon (2 rounds).
Logs every step's loss and gradient norm, and every 25 steps the share of dead units (off on all 4096 fixed validation inputs).
At the end: validation loss on all positions, and on positions whose next character is rare (the rarest characters
that together make up 2% of the training text).
Usage: OPT=adam HID_LR=0.003 CLIP=0 python3 lmspike.py SEED     -> one line in lmspike.jsonl
"""
import json, math, os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from train import Net, batch, train, val, CTX, V, ns
import train as T
ACT = os.environ.get("ACT", "relu")
if ACT != "relu":                                          # swap the model's activation; "dead" then means never positive
    class _F:
        def __getattr__(s, k): return getattr(F, k)
        relu = staticmethod(lambda h: F.leaky_relu(h, 0.1) if ACT == "leaky" else F.gelu(h))
    T.F = _F()

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = int(sys.argv[1])
OPT = os.environ.get("OPT", "adam")
HID_LR = float(os.environ.get("HID_LR", 0.003))
EMB_LR = float(os.environ.get("EMB_LR", 0.003))
CLIP = float(os.environ.get("CLIP", 0))
STEPS = int(os.environ.get("STEPS", 1000))
WD = float(os.environ.get("WD", 0.0))
B2 = float(os.environ.get("B2", 0.999))
DECAY = 0.85
WARM = int(os.environ.get("WARM", 0))                     # linear lr warmup over the first WARM steps
OUT = os.environ.get("OUT", "lmspike.jsonl")
SWITCH = int(os.environ.get("SWITCH", 0))                 # > 0: from this step on, the hidden matrices use OPT2
OPT2 = os.environ.get("OPT2", "muon" if OPT == "adam" else "adam")
WARM_HID = int(os.environ.get("WARM_HID", 0))             # lr warmup over WARM_HID steps on the hidden matrices only
WARM2 = int(os.environ.get("WARM2", 0))                   # lr warmup over WARM2 steps after the switch
LR1 = float(os.environ.get("LR1", 0))                     # > 0 with OPT=adam OPT2=adam: constant hidden lr before the switch
RESET2 = os.environ.get("RESET2") == "1"                  # 1: fresh Adam state for the hidden matrices at the switch
MUON_LR = float(os.environ.get("MUON_LR", 0.02))
EARLY_N = int(os.environ.get("EARLY_N", 0))                # hidden lr times EARLY_MULT for the first EARLY_N steps (any optimizer)
EARLY_MULT = float(os.environ.get("EARLY_MULT", 1))
OUTBIAS = os.environ.get("OUTBIAS") == "1"                # 1: output bias starts at log character frequencies (the model starts out predicting the unigram)
T.BATCH = BATCH = int(os.environ.get("BATCH", 512))      # examples per step (batch() reads train.BATCH)
T.HID = HID = int(os.environ.get("HID", 512))            # hidden width (Net reads train.HID when it is built)
MWD = float(os.environ.get("MWD", 0))                       # Muon weight decay, multiplied by lr (as in the speedrun)
SHIFT = os.environ.get("SHIFT", "")                         # perm: characters relabeled by a fixed random permutation; rev: text reversed
SHIFT_AT = int(os.environ.get("SHIFT_AT", 0))               # step at which SHIFT happens (0: from the start); > 0: the lr schedule restarts there
SHIFT_EVERY = int(os.environ.get("SHIFT_EVERY", 0))         # > 0: a new SHIFT every SHIFT_EVERY steps (perm: a new permutation each time), schedule restarts each time
if SHIFT_EVERY: SHIFT_AT = SHIFT_EVERY
SHIFT_RESET = os.environ.get("SHIFT_RESET") == "1"          # 1: fresh optimizer state for the hidden matrices at SHIFT_AT (Adam moments / Muon momentum)
SHIFT_SHRINK = float(os.environ.get("SHIFT_SHRINK", 1))     # at SHIFT_AT: hidden matrices *= SHIFT_SHRINK, then + SHIFT_NOISE * fresh init (shrink and perturb)
SHIFT_NOISE = float(os.environ.get("SHIFT_NOISE", 0))
ROUNDS = int(os.environ.get("ROUNDS", 2))                   # Newton–Schulz rounds for Muon; != 2: rescaled to the 2-round size, so only the shape changes
EMB_RESET = {"": "", "0": "", "1": "both"}.get(os.environ.get("EMB_RESET", ""), os.environ.get("EMB_RESET"))   # 1 (both) / emb / out: at every change, that table back to a fresh init, fresh AdamW state
SOFTCAP = float(os.environ.get("SOFTCAP", 0))               # > 0: logits = SOFTCAP * tanh(logits / SOFTCAP) (Gemma 2 / modded-nanogpt logit soft-capping), in training and validation
LSMOOTH = float(os.environ.get("LSMOOTH", 0))               # label smoothing in the training loss only (validation stays plain cross-entropy)
SHIFT_FRAC = float(os.environ.get("SHIFT_FRAC", 1))         # < 1 with SHIFT=perm: only this share of the characters is relabeled (among themselves), the rest keep their labels
CAP = float(os.environ.get("CAP", 0))                       # > 0: diagnostic, after every step clip each hidden matrix's singular values to <= CAP (exact SVD)

if SHIFT == "prose":
    import numpy as np, pydoc_data.topics as _pt
    _s = "".join(c for c in " ".join(_pt.topics.values()) if 32 <= ord(c) < 127 or c == "\n")
    _d = T.lut[torch.from_numpy(np.frombuffer(_s.encode("ascii"), dtype=np.uint8).astype(np.int64))]
    alt_tr, alt_va = _d[: int(0.9 * len(_d))], _d[int(0.9 * len(_d)):]
counts = torch.bincount(train, minlength=V).float()
order = counts.argsort()
cum = counts[order].cumsum(0) / counts.sum()
rare = torch.zeros(V, dtype=torch.bool); rare[order[cum <= 0.02]] = True


def hidden(m, x):
    a0 = m.inp(m.emb(x).flatten(1)); z0 = T.F.relu(a0); a1 = m.h1(z0); z1 = T.F.relu(a1); a2 = m.h2(z1)
    return a0, a1, a2


def full_val(m):
    idx = torch.arange(len(val) - CTX); tot = torch.zeros(()); totr = torch.zeros(()); nr = 0
    with torch.no_grad():
        for i in range(0, len(idx), 16384):
            ix = idx[i:i + 16384]; y = val[ix + CTX]
            l = F.cross_entropy(m(val[ix[:, None] + torch.arange(CTX)]), y, reduction="none")
            tot += l.sum(); r = rare[y]; totr += l[r].sum(); nr += int(r.sum())
    return (tot / len(idx)).item(), (totr / nr).item()


torch.manual_seed(SEED)
m = Net()
if SOFTCAP:
    m.forward = lambda x, f=m.forward: SOFTCAP * torch.tanh(f(x) / SOFTCAP)
if OUTBIAS:
    with torch.no_grad(): m.out.bias.copy_(torch.log(counts.clamp_min(1) / counts.sum()))
DSEED = int(os.environ.get("DSEED", SEED))                 # batch-order seed (default: same as SEED, which sets the init)
g = torch.Generator().manual_seed(1000 + DSEED)
probe_ix = torch.randint(0, len(val) - CTX - 1, (4096,), generator=torch.Generator().manual_seed(7))
probe = val[probe_ix[:, None] + torch.arange(CTX)]
hid = [m.inp.weight, m.h1.weight, m.h2.weight]
adam = torch.optim.AdamW([m.emb.weight, m.out.weight, m.out.bias], lr=EMB_LR, weight_decay=0, betas=(0.9, B2))
hopt = torch.optim.AdamW(hid, lr=HID_LR, weight_decay=WD, betas=(0.9, B2))
mom = [torch.zeros_like(p) for p in hid]
m1 = [torch.zeros_like(p) for p in hid]; sq = [torch.zeros_like(p) for p in hid]   # Adam moments for the adamshape start
losses, gnorms, dead, wstat, upd = [], [], [], [], []   # upd: per step, [sum of ||Δw|| over hidden matrices, lr multiplier]   # wstat: [step, (Frobenius norm, top singular value) per hidden matrix]
shifts = list(range(SHIFT_EVERY, STEPS, SHIFT_EVERY)) if SHIFT and SHIFT_EVERY else [SHIFT_AT] if SHIFT else []
starts = [0] + [s for s in shifts if s]
v_pres = []                                                   # validation loss just before each shift
for step in range(STEPS):
    if step in shifts:
        if step: v_pres.append(round(full_val(m)[0], 5))
        if SHIFT == "perm":
            P = torch.randperm(V, generator=torch.Generator().manual_seed(99 + shifts.index(step)))
            if SHIFT_FRAC < 1:
                gp = torch.Generator().manual_seed(199 + shifts.index(step)); k = round(SHIFT_FRAC * V)
                idx = torch.randperm(V, generator=gp)[:k]; P = torch.arange(V); P[idx] = idx[torch.randperm(k, generator=gp)]
            train, val, rare = P[train], P[val], rare[P.argsort()]
        elif SHIFT == "prose":                # alternate Python source and the English Python docs (pydoc_data.topics)
            train, val, alt_tr, alt_va = alt_tr, alt_va, train, val
        else:
            train, val = train.flip(0), val.flip(0)
        probe = val[(probe_ix % (len(val) - CTX - 1))[:, None] + torch.arange(CTX)]
        if step and (SHIFT_SHRINK != 1 or SHIFT_NOISE):
            torch.manual_seed(500 + SEED); fresh = Net()
            with torch.no_grad():
                for p, q in zip(hid, [fresh.inp.weight, fresh.h1.weight, fresh.h2.weight]): p.mul_(SHIFT_SHRINK).add_(q, alpha=SHIFT_NOISE)
        if EMB_RESET and step:
            torch.manual_seed(700 + SEED + step); fresh = Net()
            with torch.no_grad():
                if EMB_RESET in ("both", "emb"): m.emb.weight.copy_(fresh.emb.weight)
                if EMB_RESET in ("both", "out"): m.out.weight.copy_(fresh.out.weight); m.out.bias.copy_(fresh.out.bias)
            adam = torch.optim.AdamW([m.emb.weight, m.out.weight, m.out.bias], lr=EMB_LR, weight_decay=0, betas=(0.9, B2))
        if SHIFT_RESET and step:
            hopt = torch.optim.AdamW(hid, lr=HID_LR, weight_decay=WD, betas=(0.9, B2)); mom = [torch.zeros_like(p) for p in hid]
    t0 = max(s for s in starts if s <= step); L = min([s for s in starts if s > step] + [STEPS]) - t0
    t = step - t0
    sched = min(1.0, (1 - t / L) / DECAY) * (min(1.0, (t + 1) / WARM) if WARM else 1.0)
    if SWITCH and WARM2 and step >= SWITCH:
        sched *= min(1.0, (step - SWITCH + 1) / WARM2)
    if step % 25 == 0 or step < 25:
        with torch.no_grad():
            dead.append([step] + [round(float(((z > 0).sum(0) == 0).float().mean()), 4) for z in hidden(m, probe)])
            wstat.append([step] + [[round(p.norm().item(), 3), round(torch.linalg.matrix_norm(p, 2).item(), 3)] for p in hid])
    x, y = batch(train, g)
    loss = F.cross_entropy(m(x), y, label_smoothing=LSMOOTH)
    m.zero_grad(); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP if CLIP else math.inf)
    losses.append(round(loss.item(), 4)); gnorms.append(round(gn.item(), 4))
    if os.environ.get("UPD") == "1": before = [p.detach().clone() for p in hid]
    for gr in adam.param_groups: gr["lr"] = EMB_LR * sched
    adam.step()
    cur = OPT2 if SWITCH and step >= SWITCH else OPT
    em = EARLY_MULT if step < EARLY_N else 1.0
    if SWITCH and RESET2 and step == SWITCH:
        hopt = torch.optim.AdamW(hid, lr=HID_LR, weight_decay=WD, betas=(0.9, B2))
    if cur == "adam":
        hs = sched * (min(1.0, (t + 1) / WARM_HID) if WARM_HID else 1.0)
        for gr in hopt.param_groups: gr["lr"] = LR1 if (LR1 and SWITCH and step < SWITCH) else HID_LR * hs * em
        hopt.step()
    elif cur != "frozen":                     # frozen: hidden matrices not updated at all
        with torch.no_grad():
            for p, buf, a1, v2 in zip(hid, mom, m1, sq):
                buf.mul_(0.95).add_(p.grad)
                M = p.grad + 0.95 * buf
                O = ns(M, 2)
                if ROUNDS != 2:
                    Ok = ns(M, ROUNDS); O = Ok * (O.norm() / (Ok.norm() + 1e-12))
                if cur == "nsgd":                     # same size as Muon's update, but the gradient's own shape
                    O = M * (O.norm() / (M.norm() + 1e-12))
                if cur == "adamshape":                # same size as Muon's update, but Adam's shape (betas 0.9, 0.999, bias-corrected)
                    a1.mul_(0.9).add_(p.grad, alpha=0.1)
                    v2.mul_(0.999).add_(p.grad ** 2, alpha=0.001)
                    u = (a1 / (1 - 0.9 ** (step + 1))) / ((v2 / (1 - 0.999 ** (step + 1))).sqrt() + 1e-8)
                    O = u * (O.norm() / (u.norm() + 1e-12))
                if MWD: p.mul_(1 - MWD * (MUON_LR if SWITCH else HID_LR) * sched * em)
                p.add_(O, alpha=-(MUON_LR if SWITCH else HID_LR) * sched * em)
    if os.environ.get("UPD") == "1": upd.append([round(sum((p.detach() - q).norm().item() for p, q in zip(hid, before)), 5), round(sched * em, 5)])
    if CAP:
        with torch.no_grad():
            for p in hid:
                U, S, Vh = torch.linalg.svd(p, full_matrices=False)
                if S[0] > CAP: p.copy_((U * S.clamp_max(CAP)) @ Vh)
v, vr = full_val(m)
v_pre = v_pres[0] if v_pres else None
with torch.no_grad():
    dead.append([STEPS] + [round(float(((z > 0).sum(0) == 0).float().mean()), 4) for z in hidden(m, probe)])
# spike: loss more than 1.0 above the median of the previous 50 steps
spikes = [i for i in range(50, STEPS) if losses[i] > sorted(losses[i - 50:i])[25] + 1.0]
print(f"{ACT} {f'{SHIFT}@{SHIFT_AT}{"+" if SHIFT_EVERY else ""}{" reset" if SHIFT_RESET else ""}{f" shrink {SHIFT_SHRINK:g}+{SHIFT_NOISE:g}" if SHIFT_SHRINK != 1 or SHIFT_NOISE else ""} (before {v_pre:.4f}) ' if SHIFT and SHIFT_AT else f'{SHIFT} ' if SHIFT else ''}{f'batch {BATCH} ' if BATCH != 512 else ''}{f'hid {HID} ' if HID != 512 else ''}{f'mwd {MWD:g} ' if MWD else ''}{f'wd {WD:g} ' if WD else ''}{f'steps {STEPS} ' if STEPS != 1000 else ''}{'outbias ' if OUTBIAS else ''}{f'cap {CAP:g} ' if CAP else ''}{f'softcap {SOFTCAP:g} ' if SOFTCAP else ''}{f'lsmooth {LSMOOTH:g} ' if LSMOOTH else ''}{f'frac {SHIFT_FRAC:g} ' if SHIFT_FRAC < 1 else ''}{f'early {EARLY_N}x{EARLY_MULT:g} ' if EARLY_N else ''}{f'warm_hid {WARM_HID} ' if WARM_HID else ''}{'reset ' if RESET2 else ''}{f'rounds {ROUNDS} ' if ROUNDS != 2 else ''}{f'emb_reset {EMB_RESET} ' if EMB_RESET else ''}{f'lr1 {LR1:g} ' if LR1 else ''}{OPT}{f'(lr {MUON_LR:g})->{OPT2}@{SWITCH}' if SWITCH else ''} warm {WARM} emb {EMB_LR:g} lr {HID_LR:g} clip {CLIP:g} b2 {B2:g} seed {SEED}{f' data {DSEED}' if DSEED != SEED else ''}: val {v:.4f} rare {vr:.4f}  spikes {len(spikes)} {spikes[:6]}"
      f"  gn max {max(gnorms):.1f}  dead end {dead[-1][1:]}", flush=True)
with open(os.path.join(HERE, OUT), "a") as f:
    f.write(json.dumps(dict(opt=OPT, act=ACT, switch=SWITCH, opt2=OPT2 if SWITCH else None, warm=WARM, warm2=WARM2, muon_lr=MUON_LR if SWITCH else None, lr1=LR1, reset2=RESET2, warm_hid=WARM_HID, hid_lr=HID_LR, emb_lr=EMB_LR, clip=CLIP, b2=B2, wd=WD, mwd=MWD, hid=HID, batch=BATCH, outbias=OUTBIAS, cap=CAP, softcap=SOFTCAP, lsmooth=LSMOOTH, shift_frac=SHIFT_FRAC, emb_reset=EMB_RESET, upd=upd, rounds=ROUNDS, early_n=EARLY_N, early_mult=EARLY_MULT, seed=SEED, dseed=DSEED, steps=STEPS, shift=SHIFT, shift_at=SHIFT_AT, shift_every=SHIFT_EVERY, val_pres=v_pres, shift_reset=SHIFT_RESET, shift_shrink=SHIFT_SHRINK, shift_noise=SHIFT_NOISE, val_pre=v_pre and round(v_pre, 5), val=round(v, 5),
                            val_rare=round(vr, 5), n_rare_chars=int(rare.sum()), spikes=spikes, loss=losses, gnorm=gnorms, dead=dead, wstat=wstat)) + "\n")
