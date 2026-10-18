"""Muon bursts in a small GPT-style character language model.

Model: 2 transformer blocks, width 128, 4 heads, pre-LayerNorm, GELU MLP (4x), context 64, untied input/output tables of 256 rows.
Text: the Python stdlib source (experiments/corpus.txt, ids 0..V-1, V = 96) and, for one shift, the English Python docs.
Task lines (shift "task"): 8 random lowercase letters, "|", the same letters reversed, newline (18 tokens, the code's own characters).

Phase 1 (always the same): code only, AdamW on everything, 4000 steps.
  python3 lmburst.py p1 SEED                          -> ckpt/p1_sSEED.pt (model + AdamW state)
Phase 2: a shift (new data) and an arm (what the optimizer does at the switch).
  python3 lmburst.py p2 SEED SHIFT ARM [key=value ...] -> one line in OUT (default runs.jsonl)
Shifts:  perm    characters relabeled by a fixed random permutation (same ids, new meaning)
         newsym  the same code written with ids 128..223, whose input and output rows phase 1 never trained
         rev     code reversed
         prose   English Python docs (pydoc_data.topics)
         task    88% code + 12% task lines (a new task format; scored by line accuracy on the reversed half)
         mixperm 88% code + 12% perm code;  mixnew  88% code + 12% newsym code
         multi   a new permutation every SEG steps (schedule restarts each time)
Arms:    adam    AdamW continues with phase 1's moments, lr back to peak, no warmup
         adamw   same, with lr warmup over WARM2 steps
         fresh   untrained model, fresh AdamW, warmup 100 (as phase 1)
         reset   fresh AdamW moments for all parameters, warmup WARM2 steps
         burst   fresh AdamW moments; the hidden matrices take BURST Muon steps first (from BURST_AT), then AdamW with warmup WARM2
         muon    hidden matrices on Muon for all of phase 2; the rest fresh AdamW with warmup WARM2
"Hidden matrices": the four 2-D weights of each block (qkv, attention out, MLP in, MLP out). Embedding, output table, LayerNorms
and biases always use AdamW.
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(int(os.environ.get("THREADS", 2)))
DEV = os.environ.get("DEV", "cpu")
CTX, D, NL, NH, VOCAB = 64, int(os.environ.get("WIDTH", 128)), int(os.environ.get("LAYERS", 2)), 4, 256
B = int(os.environ.get("BATCH", 64))
LR, B2, WD = float(os.environ.get("LR", 3e-3)), float(os.environ.get("B2", 0.95)), 0.1
MUON_LR = float(os.environ.get("MUON_LR", 0.02))

# ---------- data ----------
text = open(os.path.join(HERE, "..", "experiments", "corpus.txt")).read()
chars = sorted(set(text)); V = len(chars)
lut = torch.zeros(128, dtype=torch.long); lut[[ord(c) for c in chars]] = torch.arange(V)
enc = lambda s: lut[torch.from_numpy(np.frombuffer(s.encode("ascii"), dtype=np.uint8).astype(np.int64))]
code = enc(text); split = int(0.9 * len(code)); code_tr, code_va = code[:split], code[split:]


def prose():
    import pydoc_data.topics as pt
    s = "".join(c for c in " ".join(pt.topics.values()) if 32 <= ord(c) < 127 or c == "\n")
    d = enc(s); k = int(0.9 * len(d)); return d[:k], d[k:]


LETTERS = lut[torch.arange(ord("a"), ord("z") + 1)]
LINE, NL8 = 18, 8
BAR, NLN = int(lut[ord("|")]), int(lut[ord("\n")])


def task_lines(n, g):
    """n task lines, tensor (n, 18): 8 letters | reversed \n."""
    w = LETTERS[torch.randint(26, (n, NL8), generator=g)]
    return torch.cat([w, torch.full((n, 1), BAR), w.flip(1), torch.full((n, 1), NLN)], 1)


def task_windows(n, g):
    k = CTX // LINE + 2
    rows = task_lines(n * k, g).view(n, -1); off = torch.randint(0, LINE, (n,), generator=g)
    return torch.stack([r[o:o + CTX + 1] for r, o in zip(rows, off.tolist())])


def text_windows(src, n, g):
    ix = torch.randint(0, len(src) - CTX - 1, (n,), generator=g)
    return torch.stack([src[i:i + CTX + 1] for i in ix])


def mix(parts, g):
    """parts: list of (share, sampler(n, g)). Returns a batch of B windows, shares rounded per batch."""
    ns = [round(s * B) for s, _ in parts]; ns[0] += B - sum(ns)
    w = torch.cat([f(n, g) for n, (_, f) in zip(ns, parts) if n > 0])
    return w[torch.randperm(len(w), generator=g)]


# ---------- model ----------
class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        s.qkv, s.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)
        s.fc, s.fc2 = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)

    def forward(s, x, keep=None):
        q, k, v = s.qkv(s.ln1(x)).view(x.shape[0], x.shape[1], 3, NH, D // NH).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(x.shape)
        x = x + s.proj(a)
        h = s.fc(s.ln2(x))
        if keep is not None: keep.append(h)
        return x + s.fc2(F.gelu(h))


class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb, s.pos = nn.Embedding(VOCAB, D), nn.Parameter(torch.zeros(CTX, D))
        s.blocks = nn.ModuleList(Block() for _ in range(NL))
        s.lnf, s.out = nn.LayerNorm(D), nn.Linear(D, VOCAB)
        for n, p in s.named_parameters():
            if p.dim() == 2 and n != "pos": nn.init.normal_(p, 0, 0.02)
            if n.endswith("proj.weight") or n.endswith("fc2.weight"): nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * NL))
            if n.endswith("bias"): nn.init.zeros_(p)
        nn.init.normal_(s.pos, 0, 0.01)

    def forward(s, x, keep=None):
        h = s.emb(x) + s.pos[: x.shape[1]]
        for b in s.blocks: h = b(h, keep)
        return s.out(s.lnf(h))

    def hidden(s):
        return [p for b in s.blocks for p in (b.qkv.weight, b.proj.weight, b.fc.weight, b.fc2.weight)]


def groups(m, hid=True):
    hs = set(id(p) for p in m.hidden())
    mats = [p for n, p in m.named_parameters() if p.dim() == 2 and n not in ("pos",) and (hid or id(p) not in hs) and "emb" not in n]
    rest = [p for n, p in m.named_parameters() if id(p) not in set(map(id, mats)) and (hid or id(p) not in hs)]
    return [dict(params=mats, weight_decay=WD), dict(params=rest, weight_decay=0.0)]


def adamw(params_groups):
    return torch.optim.AdamW(params_groups, lr=LR, betas=(0.9, B2), eps=1e-8)


def ns5(G):
    """Muon's Newton–Schulz orthogonalization (5 rounds of the quintic, Keller Jordan's coefficients)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7); tr = X.shape[0] > X.shape[1]
    if tr: X = X.T
    for _ in range(5):
        A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X


def muon_step(ps, bufs, lr):
    with torch.no_grad():
        for p, buf in zip(ps, bufs):
            buf.mul_(0.95).add_(p.grad)
            O = ns5(p.grad + 0.95 * buf) * max(1, p.shape[0] / p.shape[1]) ** 0.5
            p.mul_(1 - WD * lr); p.add_(O, alpha=-lr)


# ---------- evaluation ----------
def win_loss(m, w):
    with torch.no_grad():
        tot = 0.0
        for i in range(0, len(w), 256):
            x = w[i:i + 256].to(DEV)
            tot += F.cross_entropy(m(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten(), reduction="sum").item()
        return tot / (len(w) * CTX)


def task_eval(m, n=384):
    """Teacher-forced on fresh task lines: share of lines with all 8 reversed letters right, and loss on those letters."""
    lines = task_lines(n * 3, torch.Generator().manual_seed(11)).view(n, 3 * LINE)
    w = torch.cat([lines, lines[:, :CTX + 1 - 3 * LINE]], 1)
    with torch.no_grad(): lg = m(w[:, :-1].to(DEV)).cpu()
    y = w[:, 1:]; pos = torch.tensor([l * LINE + j for l in range(3) for j in range(NL8, 2 * NL8)])
    l = F.cross_entropy(lg[:, pos].flatten(0, 1), y[:, pos].flatten())
    ok = (lg[:, pos].argmax(-1) == y[:, pos]).view(n, 3, NL8).all(-1)
    return round(ok.float().mean().item(), 4), round(l.item(), 4)


def off_units(m, w):
    """Per block: share of MLP units whose pre-activation is <= 0 on every position of w (off), and mean share positive."""
    keep = []
    with torch.no_grad(): m(w[:, :-1].to(DEV), keep)
    return [[round((h.amax((0, 1)) <= 0).float().mean().item(), 4), round((h > 0).float().mean().item(), 4)] for h in keep]


def sched(t, T, warm):
    """Full lr for the first half, then linear to 0; times linear warmup over `warm` steps."""
    s = min(1.0, (1 - t / T) / 0.5)
    return s * (min(1.0, (t + 1) / warm) if warm else 1.0)


# ---------- data for a shift ----------
def make_perm(i):
    P = torch.arange(VOCAB); P[:V] = torch.randperm(V, generator=torch.Generator().manual_seed(99 + i)); return P


def phase1_sampler():
    return [(1.0, lambda n, g: text_windows(code_tr, n, g))]


def shift_data(shift, seg=0):
    """Returns (train parts, dict of eval sets {name: windows}) for phase 2."""
    ev = torch.Generator().manual_seed(7)
    old = {"code": text_windows(code_va, 384, ev)}
    if shift in ("perm", "multi", "mixperm"):
        P = make_perm(seg); tr, va = P[code_tr], P[code_va]
    elif shift in ("newsym", "mixnew"):
        tr, va = code_tr + 128, code_va + 128
    elif shift == "rev":
        tr, va = code_tr.flip(0), code_va.flip(0)
    elif shift == "prose":
        tr, va = prose()
    if shift == "task":
        parts = [(0.88, lambda n, g: text_windows(code_tr, n, g)), (0.12, task_windows)]
        new = {"task": task_windows(384, ev)}
    else:
        new = {"new": text_windows(va, 384, ev)}
        f = lambda n, g, tr=tr: text_windows(tr, n, g)
        parts = [(0.88, phase1_sampler()[0][1]), (0.12, f)] if shift.startswith("mix") else [(1.0, f)]
    return parts, new, old


# ---------- runs ----------
def phase1(seed, steps):
    torch.manual_seed(seed); m = GPT().to(DEV); opt = adamw(groups(m)); g = torch.Generator().manual_seed(1000 + seed)
    parts = phase1_sampler(); t0 = time.time(); losses = []
    for t in range(steps):
        for gr in opt.param_groups: gr["lr"] = LR * sched(t, steps, 100)
        x = mix(parts, g).to(DEV)
        loss = F.cross_entropy(m(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        losses.append(round(loss.item(), 4))
        if t % 500 == 0: print(t, losses[-1], f"{time.time() - t0:.0f}s", flush=True)
    ev = torch.Generator().manual_seed(7)
    res = dict(code=round(win_loss(m, text_windows(code_va, 384, ev)), 4), task=task_eval(m))
    os.makedirs(os.path.join(HERE, "ckpt"), exist_ok=True)
    torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), seed=seed, steps=steps, res=res), os.path.join(HERE, "ckpt", f"p1_s{seed}.pt"))
    print("phase1", seed, res, f"{time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(HERE, "phase1.jsonl"), "a") as f:
        f.write(json.dumps(dict(seed=seed, steps=steps, lr=LR, b2=B2, width=D, layers=NL, res=res, loss=losses[::10])) + "\n")


def phase2(seed, shift, arm, kw):
    steps = int(kw.get("steps", 3000)); warm2 = int(kw.get("warm2", 500)); nb = int(kw.get("burst", 100)); b_at = int(kw.get("burst_at", 0))
    seg = int(kw.get("seg", 0)) or steps; dseed = int(kw.get("dseed", seed))
    ck = torch.load(os.path.join(HERE, "ckpt", kw.get("ckpt", f"p1_s{seed}.pt")), weights_only=False)
    if arm == "fresh":
        torch.manual_seed(5000 + seed); m = GPT().to(DEV)
    else:
        m = GPT().to(DEV); m.load_state_dict(ck["model"])
    hid = m.hidden()
    split_opt = arm in ("burst", "muon")
    if split_opt:                                           # non-hidden params in one AdamW, hidden ones in another (or Muon)
        opt = adamw(groups(m, hid=False)); hopt = adamw([dict(params=hid, weight_decay=WD)])
    else:
        opt = adamw(groups(m))
        if arm in ("adam", "adamw"):
            opt.load_state_dict(ck["opt"])
    bufs = [torch.zeros_like(p) for p in hid]
    g = torch.Generator().manual_seed(3000 + dseed)
    log = []; t0 = time.time(); segi = 0
    parts, new, old = shift_data(shift, 0)
    probe = next(iter(new.values()))[:64]

    def evaluate(t):
        e = dict(step=t, new={k: round(win_loss(m, w), 4) for k, w in new.items()}, old=round(win_loss(m, old["code"]), 4), off=off_units(m, probe))
        if shift == "task": e["task"] = task_eval(m)
        log.append(e)

    marks = {0, 10, 30, 100, 200, 300, 500, 1000}
    for t in range(steps):
        if shift == "multi" and t and t % seg == 0:
            segi += 1; parts, new, old_ = shift_data(shift, segi); probe = next(iter(new.values()))[:64]
        ts = t % seg
        if t % 100 == 0 or t in marks or (shift == "multi" and ts in (0, 100)): evaluate(t)
        w = warm2 if arm in ("adamw", "reset", "burst", "muon") else (100 if arm == "fresh" else 0)
        if shift == "multi" and arm == "fresh" and ts == 0 and t:
            torch.manual_seed(5000 + seed + t); m = GPT().to(DEV); hid = m.hidden(); opt = adamw(groups(m))
        s = sched(ts, seg, w if (shift != "multi" or arm == "fresh" or t < seg) else 0)
        x = mix(parts, g).to(DEV)
        loss = F.cross_entropy(m(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
        for o in ([opt, hopt] if split_opt else [opt]): o.zero_grad(set_to_none=True)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        for gr in opt.param_groups: gr["lr"] = LR * s
        opt.step()
        if split_opt:
            in_burst = arm == "muon" or (b_at <= t < b_at + nb)
            if in_burst:
                muon_step(hid, bufs, MUON_LR * (s if arm == "muon" else 1.0))
            else:
                if arm == "burst" and t == b_at + nb: hopt = adamw([dict(params=hid, weight_decay=WD)])
                hw = sched(ts, seg, 0) * (min(1.0, (t - b_at - nb + 1) / warm2) if t >= b_at + nb else 1.0)
                for gr in hopt.param_groups: gr["lr"] = LR * hw
                hopt.step()
    evaluate(steps)
    f = log[-1]
    print(f"{shift:>7} {arm:>6} s{seed} d{dseed} {kw}: new {f['new']} old {f['old']} task {f.get('task')}  off@100 {[l['off'] for l in log if l['step'] == 100]}  {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(HERE, kw.get("out", "runs.jsonl")), "a") as fo:
        fo.write(json.dumps(dict(seed=seed, dseed=dseed, shift=shift, arm=arm, kw=kw, lr=LR, b2=B2, muon_lr=MUON_LR, width=D, layers=NL,
                                 p1=ck["res"], log=log)) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "p1":
        phase1(int(sys.argv[2]), int(os.environ.get("STEPS1", 4000)))
    else:
        phase2(int(sys.argv[2]), sys.argv[3], sys.argv[4], dict(a.split("=") for a in sys.argv[5:]))
