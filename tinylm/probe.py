"""What does the 100-step Muon burst change? Compare the model after 100 phase-2 steps of AdamW vs 100 steps of Muon (burst).
Same runs as tuned.jsonl: DSEED=0, LR2=0.003, MUON_LR=0.012. usage: MUON_LR=0.012 python3 probe.py SEED
Measures, on 256 reverse sequences:
  attn  - attention weight from the position that writes answer letter k onto input letter 7-k (best head over both layers)
  grad  - size of the gradient that answer letter k's loss puts on the attention Q and K weights (both layers)"""
import copy, os, sys, torch, torch.nn.functional as F
seed = int(sys.argv[1]); sys.argv = sys.argv[:1]
os.environ.setdefault("DSEED", "0")
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinylm.py")).read().split("if __name__")[0])
import models.layers as layers

def run_adam(m, g, stops, total=P2):         # tinylm train(), opt="adam", lr LR2; yields at each step in `stops`
    opt = torch.optim.AdamW(list(m.parameters()), lr=LR2, weight_decay=WD)
    for i in range(max(stops)):
        sc = min(1, 2 * (1 - i / total)) * min(1, (i + 1) / WARM)
        s = mixed(BS, "CSR", NEW_F, g)
        loss = F.cross_entropy(answer_logits(m, s).reshape(-1, V), s[:, L + 2:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), CFG.grad_clip)
        for gr in opt.param_groups: gr["lr"] = LR2 * sc
        opt.step()
        if i + 1 in stops: yield i + 1

NEW_F = [(1 - FREQ_NEW) / 2, (1 - FREQ_NEW) / 2, FREQ_NEW]
torch.manual_seed(seed); g = torch.Generator().manual_seed(int(DSEED))
m1 = MinimalLLM(CFG)
train(m1, P1, "CS", [0.5, 0.5], g, "adam", [], 0, torch.Generator().manual_seed(999))      # phase 1, as in tinylm.py
STOPS = [100, 200, 300, 400, 500]
models = {"phase-1 end": m1}
m = copy.deepcopy(m1)
for st in run_adam(m, torch.Generator().manual_seed(int(DSEED) + 100), STOPS): models[f"AdamW {st}"] = copy.deepcopy(m)
m = copy.deepcopy(m1); g3 = torch.Generator().manual_seed(int(DSEED) + 100)   # control: 100 AdamW steps, then a fresh AdamW (same restart as the burst arm)
list(run_adam(m, g3, [BURST]))
for st in run_adam(m, g3, [x - BURST for x in STOPS if x > BURST]): models[f"restart {st + BURST}"] = copy.deepcopy(m)
m = copy.deepcopy(m1); g2 = torch.Generator().manual_seed(int(DSEED) + 100)
train(m, BURST, "CSR", NEW_F, g2, "muon", [], 0, torch.Generator().manual_seed(999)); models[f"burst {BURST}"] = copy.deepcopy(m)
for st in run_adam(m, g2, [x - BURST for x in STOPS if x > BURST]): models[f"burst {st + BURST}"] = copy.deepcopy(m)   # as tinylm: AdamW continues on g2

S = batch(256, "R", torch.Generator().manual_seed(4242))
attn = []
sdpa = F.scaled_dot_product_attention
def sdpa_record(Q, K, Vv, is_causal=True, dropout_p=0.0):           # same math as SDPA, keeps the weights
    T = Q.shape[-2]
    w = (Q @ K.transpose(-2, -1)) / Q.shape[-1] ** 0.5
    w = w.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf")).softmax(-1)
    out = w @ Vv
    assert torch.allclose(out, sdpa(Q, K, Vv, is_causal=True), atol=1e-5)
    attn.append(w); return out

for name, m in models.items():
    attn.clear(); layers.F.scaled_dot_product_attention = sdpa_record
    with torch.no_grad(): hit = answer_logits(m, S).argmax(-1) == S[:, L + 2:]
    layers.F.scaled_dot_product_attention = sdpa
    A = torch.stack(attn)                                              # layer, batch, head, query, key
    att = []
    for k in range(1, L + 1):                                          # query index L+k writes answer k; it lives at input index L+1-k
        a = A[:, :, :, L + k, L + 1 - k].mean(1)                       # layer, head
        att.append(a.max().item())
    gq = []
    for k in range(L):
        m.zero_grad()
        F.cross_entropy(answer_logits(m, S)[:, k], S[:, L + 2 + k]).backward()
        gq.append(sum(b.attention.qkvo_proj.grad[:2 * CFG.d_model].norm() ** 2 for b in m.transformer_blocks).sqrt().item())
    print(f"{name:12}  letters right {' '.join(f'{v:.2f}' for v in hit.float().mean(0))}")
    print(f"{'':12}  attn on right letter {' '.join(f'{v:.2f}' for v in att)}")
    print(f"{'':12}  Q/K grad from letter {' '.join(f'{v:.3f}' for v in gq)}")
