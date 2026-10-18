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
g = torch.Generator().manual_seed(1000 + SEED)
probe_ix = torch.randint(0, len(val) - CTX - 1, (4096,), generator=torch.Generator().manual_seed(7))
probe = val[probe_ix[:, None] + torch.arange(CTX)]
hid = [m.inp.weight, m.h1.weight, m.h2.weight]
adam = torch.optim.AdamW([m.emb.weight, m.out.weight, m.out.bias], lr=EMB_LR, weight_decay=0, betas=(0.9, B2))
hopt = torch.optim.AdamW(hid, lr=HID_LR, weight_decay=WD, betas=(0.9, B2))
mom = [torch.zeros_like(p) for p in hid]
m1 = [torch.zeros_like(p) for p in hid]; sq = [torch.zeros_like(p) for p in hid]   # Adam moments for the adamshape start
losses, gnorms, dead = [], [], []
for step in range(STEPS):
    sched = min(1.0, (1 - step / STEPS) / DECAY) * (min(1.0, (step + 1) / WARM) if WARM else 1.0)
    if SWITCH and WARM2 and step >= SWITCH:
        sched *= min(1.0, (step - SWITCH + 1) / WARM2)
    if step % 25 == 0 or step < 25:
        with torch.no_grad():
            dead.append([step] + [round(float(((z > 0).sum(0) == 0).float().mean()), 4) for z in hidden(m, probe)])
    x, y = batch(train, g)
    loss = F.cross_entropy(m(x), y)
    m.zero_grad(); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP if CLIP else math.inf)
    losses.append(round(loss.item(), 4)); gnorms.append(round(gn.item(), 4))
    for gr in adam.param_groups: gr["lr"] = EMB_LR * sched
    adam.step()
    cur = OPT2 if SWITCH and step >= SWITCH else OPT
    if SWITCH and RESET2 and step == SWITCH:
        hopt = torch.optim.AdamW(hid, lr=HID_LR, weight_decay=WD, betas=(0.9, B2))
    if cur == "adam":
        hs = sched * (min(1.0, (step + 1) / WARM_HID) if WARM_HID else 1.0)
        for gr in hopt.param_groups: gr["lr"] = LR1 if (LR1 and SWITCH and step < SWITCH) else HID_LR * hs
        hopt.step()
    elif cur != "frozen":                     # frozen: hidden matrices not updated at all
        with torch.no_grad():
            for p, buf, a1, v2 in zip(hid, mom, m1, sq):
                buf.mul_(0.95).add_(p.grad)
                M = p.grad + 0.95 * buf
                O = ns(M, 2)
                if cur == "nsgd":                     # same size as Muon's update, but the gradient's own shape
                    O = M * (O.norm() / (M.norm() + 1e-12))
                if cur == "adamshape":                # same size as Muon's update, but Adam's shape (betas 0.9, 0.999, bias-corrected)
                    a1.mul_(0.9).add_(p.grad, alpha=0.1)
                    v2.mul_(0.999).add_(p.grad ** 2, alpha=0.001)
                    u = (a1 / (1 - 0.9 ** (step + 1))) / ((v2 / (1 - 0.999 ** (step + 1))).sqrt() + 1e-8)
                    O = u * (O.norm() / (u.norm() + 1e-12))
                p.add_(O, alpha=-(MUON_LR if SWITCH else HID_LR) * sched)
v, vr = full_val(m)
with torch.no_grad():
    dead.append([STEPS] + [round(float(((z > 0).sum(0) == 0).float().mean()), 4) for z in hidden(m, probe)])
# spike: loss more than 1.0 above the median of the previous 50 steps
spikes = [i for i in range(50, STEPS) if losses[i] > sorted(losses[i - 50:i])[25] + 1.0]
print(f"{ACT} {f'warm_hid {WARM_HID} ' if WARM_HID else ''}{'reset ' if RESET2 else ''}{f'lr1 {LR1:g} ' if LR1 else ''}{OPT}{f'(lr {MUON_LR:g})->{OPT2}@{SWITCH}' if SWITCH else ''} warm {WARM} emb {EMB_LR:g} lr {HID_LR:g} clip {CLIP:g} b2 {B2:g} seed {SEED}: val {v:.4f} rare {vr:.4f}  spikes {len(spikes)} {spikes[:6]}"
      f"  gn max {max(gnorms):.1f}  dead end {dead[-1][1:]}", flush=True)
with open(os.path.join(HERE, OUT), "a") as f:
    f.write(json.dumps(dict(opt=OPT, act=ACT, switch=SWITCH, opt2=OPT2 if SWITCH else None, warm=WARM, warm2=WARM2, muon_lr=MUON_LR if SWITCH else None, lr1=LR1, reset2=RESET2, warm_hid=WARM_HID, hid_lr=HID_LR, emb_lr=EMB_LR, clip=CLIP, b2=B2, wd=WD, seed=SEED, steps=STEPS, val=round(v, 5),
                            val_rare=round(vr, 5), n_rare_chars=int(rare.sum()), spikes=spikes, loss=losses, gnorm=gnorms, dead=dead)) + "\n")
