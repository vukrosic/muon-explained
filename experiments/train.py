"""Shared setup for every experiment: data, model, and one training run with a swappable flattening step.

Data: the Python standard library's own source files (public, already on any machine with Python,
no download). Files are taken in sorted order until 3 million characters, printable ASCII only.
Model: tiny character-level MLP language model, 16 characters in, next character out.
Muon on the 3 hidden 512-wide matrices, AdamW (lr 3e-3) on the embedding and output layer.
"""
import glob, os, sys, sysconfig, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = 1000
CTX, EMB, HID, BATCH = 16, 32, 512, 512

cache = os.path.join(HERE, "corpus.txt")
if not os.path.exists(cache):
    parts, n = [], 0
    for fp in sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py"))):
        t = open(fp, errors="ignore").read()
        t = "".join(c for c in t if 32 <= ord(c) < 127 or c == "\n")
        parts.append(t); n += len(t)
        if n > 3_000_000:
            break
    open(cache, "w").write("".join(parts))
text = open(cache).read()
chars = sorted(set(text)); V = len(chars)
lut = torch.zeros(128, dtype=torch.long); lut[[ord(c) for c in chars]] = torch.arange(V)
data = lut[torch.from_numpy(np.frombuffer(text.encode("ascii"), dtype=np.uint8).astype(np.int64))]
split = int(0.9 * len(data))
train, val = data[:split], data[split:]
PYTHON = sys.version.split()[0]


def batch(src, g):
    ix = torch.randint(0, len(src) - CTX - 1, (BATCH,), generator=g)
    x = torch.stack([src[i:i + CTX] for i in ix])
    return x, src[ix + CTX]


KJ = (3.4445, -4.775, 2.0315)


def ns(G, rounds, extra_gentle=0):
    """Muon's flattening: divide by the size, then `rounds` rounds of the quintic curve."""
    X = G / (G.norm() + 1e-7)
    a, b, c = KJ
    for _ in range(rounds):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    for _ in range(extra_gentle):          # gentle cubic 1.5s − 0.5s³ pulls the band toward 1
        X = 1.5 * X - 0.5 * (X @ X.T) @ X
    return X


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, EMB)
        self.inp = nn.Linear(CTX * EMB, HID, bias=False)
        self.h1 = nn.Linear(HID, HID, bias=False)
        self.h2 = nn.Linear(HID, HID, bias=False)
        self.out = nn.Linear(HID, V)

    def forward(self, x):
        z = F.relu(self.inp(self.emb(x).flatten(1)))
        z = F.relu(self.h1(z))
        z = F.relu(self.h2(z))
        return self.out(z)


def run(flatten, seed, lr0):
    """One training run. Only `flatten` and the Muon learning rate lr0 differ between runs;
    the seed fixes the starting weights and the order of the training batches."""
    torch.manual_seed(seed)
    m = Net()
    muon_ps = [m.inp.weight, m.h1.weight, m.h2.weight]
    adam = torch.optim.AdamW([m.emb.weight, m.out.weight, m.out.bias], lr=3e-3, weight_decay=0)
    mom = [torch.zeros_like(p) for p in muon_ps]
    g = torch.Generator().manual_seed(1000 + seed)
    beta = 0.95
    curve, t_flat = [], 0.0
    for step in range(STEPS):
        sched = min(1.0, 2 * (1 - step / STEPS))         # flat, then linear decay over the last half
        x, y = batch(train, g)
        loss = F.cross_entropy(m(x), y)
        m.zero_grad(); loss.backward()
        for gr in adam.param_groups:
            gr["lr"] = 3e-3 * sched
        adam.step()
        with torch.no_grad():
            for p, buf in zip(muon_ps, mom):
                buf.mul_(beta).add_(p.grad)
                upd = p.grad + beta * buf                    # Nesterov momentum
                t0 = time.perf_counter()
                O = flatten(upd)
                t_flat += time.perf_counter() - t0
                scale = max(1, p.shape[0] / p.shape[1]) ** 0.5
                p.add_(O, alpha=-lr0 * sched * scale)
        if step % 50 == 0:
            curve.append([step, loss.item()])
    gv = torch.Generator().manual_seed(7)                 # the same 40 validation batches for every run
    with torch.no_grad():
        vl = np.mean([F.cross_entropy(m(xv), yv).item() for xv, yv in (batch(val, gv) for _ in range(40))])
    return dict(val=float(vl), curve=curve, t_flat=t_flat)
