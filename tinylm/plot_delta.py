"""Picture of what the 100-step Muon burst changes: dW = weights after 100 phase-2 steps - weights at phase-1 end,
for the burst (Muon) and for 100 AdamW steps on the same batches.
usage: MUON_LR=0.012 python3 plot_delta.py SEED      (DSEED defaults to SEED) -> delta_SEED.png
"""
import copy, os, sys, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
seed = int(sys.argv[1]); sys.argv = sys.argv[:1]
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "tinylm.py")).read().split("if __name__")[0])
ds = int(DSEED) if DSEED else seed
NEW_F = [(1 - FREQ_NEW) / 2, (1 - FREQ_NEW) / 2, FREQ_NEW]
torch.manual_seed(seed); m1 = MinimalLLM(CFG)
train(m1, P1, "CS", [0.5, 0.5], torch.Generator().manual_seed(ds), "adam", [], 0, torch.Generator().manual_seed(999))
runs = {}
for arm, lr in (("AdamW", LR2), ("Muon burst", LR)):
    m = copy.deepcopy(m1)
    train(m, BURST, "CSR", NEW_F, torch.Generator().manual_seed(ds + 100), "muon" if arm != "AdamW" else "adam", [], 0,
          torch.Generator().manual_seed(999), lr)
    runs[arm] = {n: (p - dict(m1.named_parameters())[n]).detach() for n, p in m.named_parameters()}
D = CFG.d_model
parts = {"attn0 QK": ("transformer_blocks.0.attention.qkvo_proj", 0, 2 * D), "attn0 VO": ("transformer_blocks.0.attention.qkvo_proj", 2 * D, 4 * D),
         "attn1 QK": ("transformer_blocks.1.attention.qkvo_proj", 0, 2 * D), "attn1 VO": ("transformer_blocks.1.attention.qkvo_proj", 2 * D, 4 * D),
         "ff0": ("transformer_blocks.0.feed_forward.up_proj.weight", 0, None), "ff1": ("transformer_blocks.1.feed_forward.up_proj.weight", 0, None)}
get = lambda arm, k: runs[arm][parts[k][0]][parts[k][1]:parts[k][2]]
col = {"AdamW": "tab:blue", "Muon burst": "tab:orange"}
fig = plt.figure(figsize=(14, 8)); gs = fig.add_gridspec(2, 3)
ax = fig.add_subplot(gs[0, 0]); x = torch.arange(len(parts)).float()
for j, arm in enumerate(runs):
    ax.bar(x + (j - 0.5) * 0.4, [get(arm, k).norm().item() for k in parts], 0.4, label=arm, color=col[arm])
ax.set_xticks(x, list(parts), rotation=30); ax.set_ylabel("size of change ||dW||"); ax.set_title("How much each weight block moved in 100 steps"); ax.legend()
ax = fig.add_subplot(gs[0, 1])
for arm in runs:
    s = torch.linalg.svdvals(get(arm, "attn1 VO")); ax.plot((s / s.sum()).numpy(), "o-", ms=3, label=arm, color=col[arm])
ax.set_yscale("log"); ax.set_xlabel("direction (sorted strongest first)"); ax.set_ylabel("share of the change")
ax.set_title("Layer-2 attention V/O change, split into directions"); ax.legend()
ax = fig.add_subplot(gs[0, 2])
for arm in runs:
    s = torch.linalg.svdvals(get(arm, "attn1 VO")) ** 2; c = (s.cumsum(0) / s.sum()).numpy()
    ax.plot(range(1, len(c) + 1), c, label=arm, color=col[arm])
ax.set_xlabel("number of strongest directions"); ax.set_ylabel("fraction of the change they hold")
ax.set_title("How many directions the change uses"); ax.set_ylim(0, 1.02); ax.legend()
for j, arm in enumerate(runs):
    ax = fig.add_subplot(gs[1, j]); w = get(arm, "attn1 VO"); v = w.abs().max().item()
    ax.imshow(w.numpy(), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto"); ax.set_title(f"{arm}: layer-2 V/O change (rows V then O)")
    ax.set_xlabel("input feature (64)"); ax.set_ylabel("row (128)")
ax = fig.add_subplot(gs[1, 2]); w = torch.cat([get("AdamW", "attn1 VO").flatten(), get("Muon burst", "attn1 VO").flatten()])
for arm in runs:
    ax.hist(get(arm, "attn1 VO").flatten().numpy(), 80, alpha=0.6, label=arm, color=col[arm], log=True)
ax.set_xlabel("change of one weight"); ax.set_title("Layer-2 V/O: per-weight changes"); ax.legend()
fig.suptitle(f"What 100 steps change (seed {seed}): weights after - weights at phase-1 end, same batches")
fig.tight_layout(); fig.savefig(os.path.join(HERE, f"delta_{seed}.png"), dpi=110)
for k in parts: print(k, {a: round(get(a, k).norm().item(), 3) for a in runs})
