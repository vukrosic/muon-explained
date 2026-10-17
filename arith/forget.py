"""Forgetting: take a trained model (saved by arith.py with SAVE) and keep training with some ops removed from the batch.
Does an op survive when its partner is still trained?

Same optimizers as arith.py, fresh optimizer state, constant learning rate = LR_MULT x the peak, weight decay WD.
Usage: FREQ2=0.48,0.48,0,0.04 OPT=adam python3 forget.py MODEL.pt LABEL   -> forget.jsonl
SAVE2=path keeps the result for a later relearn run. LIMIT=i:K trains op i on only its first K training pairs (relearning).
FOUT names the jsonl.
"""
import json, os, sys
import torch, torch.nn.functional as F
import arith as A

path, label = sys.argv[1], sys.argv[2]
FREQ2 = [float(f) for f in os.environ["FREQ2"].split(",")]
STEPS2, LR_MULT = int(os.environ.get("STEPS2", 3000)), float(os.environ.get("LR_MULT", 0.5))
ck = torch.load(path)
tr, te, order, seed = ck["tr"], ck["te"], ck["order"], ck["seed"]
torch.manual_seed(seed + 100); gen = torch.Generator().manual_seed(seed + 100)
m = A.Net(); m.load_state_dict(ck["model"])
full_tr = tr
if os.environ.get("LIMIT"):
    i_lim, k_lim = map(int, os.environ["LIMIT"].split(":"))
    tr = [t[:k_lim] if i == i_lim else t for i, t in enumerate(tr)]
muon = [m.inp.weight, m.h1.weight, m.h2.weight]
adam = torch.optim.AdamW([m.emb.weight, m.out.weight, m.out.bias], lr=A.ADAM_LR * LR_MULT, weight_decay=0)
hid = torch.optim.AdamW(muon, lr=A.HID_LR * LR_MULT, weight_decay=A.WD) if A.OPT == "adam" else None
mom = [torch.zeros_like(w) for w in muon]
freq = torch.tensor(FREQ2)
log = []
for step in range(STEPS2 + 1):
    if step % A.EVERY == 0 or step in (10, 20, 30):
        log.append(dict(step=step, train=[round(A.acc(m, r), 4) for r in full_tr], test=[round(A.acc(m, r), 4) for r in te]))
    if step == STEPS2:
        break
    k = torch.multinomial(freq, A.B, replacement=True, generator=gen)
    x = torch.cat([tr[i][torch.randint(len(tr[i]), (int((k == i).sum()),), generator=gen)] for i in range(len(tr))])
    loss = F.cross_entropy(m(x[:, :3]), x[:, 3])
    m.zero_grad(); loss.backward()
    adam.step()
    if hid:
        hid.step(); continue
    with torch.no_grad():
        for w, buf in zip(muon, mom):
            buf.mul_(A.BETA).add_(w.grad)
            w.mul_(1 - A.WD * A.MUON_LR * LR_MULT)
            w.add_(A.shaped(w.grad + A.BETA * buf, 0), alpha=-A.MUON_LR * LR_MULT)
e = log[-1]
print(f"{label:>34} {order} {A.OPT} freq2 {FREQ2}  test " + " ".join(f"{op}:{t:.2f}" for op, t in zip(order, e["test"])), flush=True)
if os.environ.get("SAVE2"):
    torch.save(dict(model=m.state_dict(), tr=full_tr, te=te, order=order, seed=seed), os.environ["SAVE2"])
with open(os.path.join(A.HERE, os.environ.get("FOUT", "forget.jsonl")), "a") as f:
    f.write(json.dumps(dict(model=os.path.basename(path), label=label, order=order, seed=seed, opt=A.OPT, freq2=FREQ2, steps2=STEPS2,
                            lr_mult=LR_MULT, wd=A.WD, limit=os.environ.get("LIMIT"), log=log)) + "\n")
