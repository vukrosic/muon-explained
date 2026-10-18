"""What does the very first hidden-layer update look like? GELU model, seed 0, one batch.
Adam's first step is lr * sign(g) (bias-corrected m / sqrt(v) = g / |g|). Muon's is lr * NS2(g).
Prints for each hidden matrix: stable rank of the gradient, and spectral norm / stable rank of each update."""
import os, sys
import torch
os.environ.setdefault("ACT", "gelu")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
import train as T
import torch.nn.functional as F
class _F:
    def __getattr__(s, k): return getattr(F, k)
    relu = staticmethod(F.gelu)
T.F = _F()

def sr(M): return (M.norm() ** 2 / torch.linalg.matrix_norm(M, 2) ** 2).item()

for seed in (0, 1, 2):
    torch.manual_seed(seed); m = T.Net(); g = torch.Generator().manual_seed(1000 + seed)
    x, y = T.batch(T.train, g)
    F.cross_entropy(m(x), y).backward()
    for name, p in (("inp", m.inp.weight), ("h1", m.h1.weight), ("h2", m.h2.weight)):
        G = p.grad; A = 0.003 * G.sign(); M = 0.03 * T.ns(G, 2)
        s = torch.linalg.svdvals(G)
        print(f"seed {seed} {name}: grad stable rank {sr(G):.1f}, top-1 share of grad energy {(s[0]**2/(s**2).sum()).item():.2f} | "
              f"Adam step: spectral {torch.linalg.matrix_norm(A, 2).item():.3f}, stable rank {sr(A):.1f} | "
              f"Muon step: spectral {torch.linalg.matrix_norm(M, 2).item():.3f}, stable rank {sr(M):.1f}")
