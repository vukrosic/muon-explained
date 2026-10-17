"""What per-neuron normalization does to record 3's updates (seed 0, ROW_BETA 0.99).

At a few steps, for each Muon matrix: how unequal the rows (neurons) of the Newton-Schulz output are,
how much the normalization rescales them, how much it turns the update, and whether the rows it
boosts belong to neurons that are rarely active (ReLU output > 0).
"""
import torch, torch.nn.functional as F
import speedrun as S
from train import Net, batch, train

S.ROW_BETA = 0.99
KEEP = (50, 200, 500, 900)
torch.manual_seed(0); m = Net(); g = torch.Generator().manual_seed(1000)
names = ["inp", "h1", "h2"]
ps = [m.inp.weight, m.h1.weight, m.h2.weight]
adam = torch.optim.AdamW([m.emb.weight, m.out.weight, m.out.bias], lr=S.ADAM_LR, betas=S.ADAM_BETAS, weight_decay=0)
mom = [torch.zeros_like(p) for p in ps]
rows = [torch.zeros(p.shape[0], 1) for p in ps]
q = lambda t, a: torch.quantile(t, a).item()

for step in range(S.STEPS):
    sched = min(1.0, (1 - step / S.STEPS) / S.DECAY)
    x, y = batch(train, g)
    acts = []
    hooks = [mod.register_forward_hook(lambda _m, _i, o: acts.append((o > 0).float().mean(0))) for mod in (m.inp, m.h1, m.h2)]
    loss = F.cross_entropy(m(x), y)
    for h in hooks: h.remove()
    m.zero_grad(); loss.backward()
    for gr in adam.param_groups: gr["lr"] = S.ADAM_LR * sched
    adam.step()
    with torch.no_grad():
        for i, (p, buf, r) in enumerate(zip(ps, mom, rows)):
            buf.mul_(S.BETA).add_(p.grad)
            p.mul_(1 - S.WD * S.MUON_LR * sched)
            O = S.flatten(p.grad + S.BETA * buf)
            r.mul_(S.ROW_BETA).add_(O.pow(2).mean(1, keepdim=True), alpha=1 - S.ROW_BETA)
            size = O.norm()
            N = O / (r / (1 - S.ROW_BETA ** (step + 1))).sqrt().add(1e-8)
            N = N * (size / (N.norm() + 1e-12))
            if step in KEEP:
                rs = O.norm(dim=1)                                   # size of each neuron's row before
                f = N.norm(dim=1) / rs                               # how much the normalization multiplied it
                active = (acts[i] * (step > 0)).clone()               # share of the batch where this neuron fired
                cos = (O * N).sum() / (O.norm() * N.norm())
                dead = (acts[i] == 0).float().mean().item()
                corr = torch.corrcoef(torch.stack([f, acts[i]]))[0, 1].item()
                print(f"step {step:4d} {names[i]:>3}: row size p90/p10 {q(rs, .9) / q(rs, .1):5.2f}   "
                      f"factor p10 {q(f, .1):.2f} p90 {q(f, .9):.2f} max {f.max().item():.2f}   "
                      f"turn {torch.rad2deg(torch.acos(cos.clamp(max=1))).item():4.1f} deg   "
                      f"neurons never firing {dead:.0%}   corr(factor, firing rate) {corr:+.2f}")
            p.add_(N, alpha=-S.MUON_LR * sched)
