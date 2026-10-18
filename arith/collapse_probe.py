"""What happens when a fresh Adam run suddenly switches most units off? Replays forget.py's Adam run exactly
(same batches, same schedule) and logs every step: batch loss per op, gradient size, update size, and the share of units
in each layer that are off on every x (op 2) test input.
Usage: python3 collapse_probe.py MODEL.pt BSEED FROM TO OUT.json    (env as forget.py: WD=1 HID_LR=0.003 OPT=adam)"""
import json, os, sys
import torch, torch.nn.functional as F
import arith as A

path, bseed, lo, hi, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
STEPS2, WARM = int(os.environ.get("STEPS2", 9000)), int(os.environ.get("WARM", 500))
ck = torch.load(path); tr, te, seed = ck["tr"], ck["te"], ck["seed"]
BS = seed + 100 + bseed
torch.manual_seed(BS); gen = torch.Generator().manual_seed(BS)
m = A.Net(); m.load_state_dict(ck["model"])
hidw = [m.inp.weight, m.h1.weight, m.h2.weight]
adam = torch.optim.AdamW([m.emb.weight, m.out.weight, m.out.bias], lr=A.ADAM_LR, weight_decay=0)
hid = torch.optim.AdamW(hidw, lr=A.HID_LR, weight_decay=A.WD)
freq = torch.tensor([0.55, 0.30, 0.12, 0.03])
log = []


def dead(xd):
    with torch.no_grad():
        a = F.relu(m.inp(m.emb(xd).flatten(1))); b = F.relu(m.h1(a)); c = F.relu(m.h2(b))
        return [round((h.amax(0) <= 0).float().mean().item(), 3) for h in (a, b, c)]


for step in range(hi + 1):
    k = torch.multinomial(freq, A.B, replacement=True, generator=gen)
    x = torch.cat([tr[i][torch.randint(len(tr[i]), (int((k == i).sum()),), generator=gen)] for i in range(len(tr))])
    logits = m(x[:, :3]); per = F.cross_entropy(logits, x[:, 3], reduction="none"); loss = per.mean()
    m.zero_grad(); loss.backward()
    sched = min(1.0, (1 - step / STEPS2) / A.DECAY) * min(1.0, (step + 1) / WARM)
    for gr in adam.param_groups: gr["lr"] = A.ADAM_LR * sched
    for gr in hid.param_groups: gr["lr"] = A.HID_LR * sched
    if step >= lo:
        before = [w.detach().clone() for w in hidw]
        ops = x[:, 1] - A.N; n_op = [int((ops == i).sum()) for i in range(4)]
        rec = dict(step=step, loss=round(loss.item(), 4), op_loss=[round(per[ops == i].mean().item(), 3) if n_op[i] else None for i in range(4)],
                   grad=[round(w.grad.norm().item(), 4) for w in hidw], wnorm=[round(w.norm().item(), 2) for w in hidw],
                   logit_max=round(logits.abs().max().item(), 1), out_norm=round(m.out.weight.norm().item(), 2))
        if step % 5 == 0: rec["dead_x"] = dead(te[2][:, :3]); rec["dead_plus"] = dead(te[0][:, :3])
    adam.step(); hid.step()
    if step >= lo:
        rec["upd"] = [round((w - b).norm().item(), 4) for w, b in zip(hidw, before)]
        log.append(rec)
json.dump(log, open(out, "w"))
