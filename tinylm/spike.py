"""Replay phase 1 of `DSEED=0 tinylm.py 0 adam` step by step and log AdamW's internals around the loss spike.
usage: python3 spike.py [EPS]"""
import sys, torch, torch.nn.functional as F
sys.argv, eps = sys.argv[:1], float(sys.argv[1]) if len(sys.argv) > 1 else 1e-8
exec(open("tinylm.py").read().split("if __name__")[0])
torch.manual_seed(0); g = torch.Generator().manual_seed(0); m = MinimalLLM(CFG)
setup_muon_optimizer(m, CFG)                                   # same calls as train(), so the run is identical
opt = torch.optim.AdamW(list(m.parameters()), lr=LR, weight_decay=WD, eps=eps)
ps = list(m.parameters())
for i in range(700):
    sc = min(1, 2 * (1 - i / P1)) * min(1, (i + 1) / WARM)
    s = mixed(BS, "CS", [0.5, 0.5], g)
    loss = F.cross_entropy(answer_logits(m, s).reshape(-1, V), s[:, L + 2:].reshape(-1))
    opt.zero_grad(); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(m.parameters(), CFG.grad_clip).item()
    old = [p.detach().clone() for p in ps]
    for gr in opt.param_groups: gr["lr"] = LR * sc
    opt.step()
    upd = sum((p.detach() - o).pow(2).sum() for p, o in zip(ps, old)).sqrt().item()
    # size of AdamW's running gradient-size estimate: median of sqrt(v) over all weights
    v = torch.cat([opt.state[p]["exp_avg_sq"].flatten() for p in ps]).sqrt().median().item()
    if i % 50 == 0 or 560 <= i <= 615 or "ALL" in os.environ:
        print(f"step {i:4d}  loss {loss.item():.4f}  grad norm {gn:.4f}  median sqrt(v) {v:.2e}  update norm {upd:.4f}")
