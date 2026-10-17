"""Is "b -> b⁻¹" (and "b -> −b") a linear map on the learned number embeddings?
Fit e[f(b)] ≈ R e[b] by least squares on half of the numbers, measure R² on the other half.
Control: the same fit for a random permutation f."""
import sys
import torch
from arith import Net, N

def r2_heldout(E, f, g, reps=20, lam=0.1):
    out = []
    for r in range(reps):
        idx = torch.randperm(len(f), generator=g)
        fit, test = idx[: len(f) // 2], idx[len(f) // 2:]
        X, Y = E[f[:, 0]], E[f[:, 1]]
        Xf = X[fit]                                        # ridge: 48 points in 64 dims would overfit
        R = torch.linalg.solve(Xf.T @ Xf + lam * torch.eye(Xf.shape[1], dtype=Xf.dtype) * (Xf ** 2).sum() / len(Xf), Xf.T @ Y[fit])
        res = Y[test] - X[test] @ R
        out.append(1 - (res ** 2).sum() / ((Y[test] - Y[test].mean(0)) ** 2).sum())
    return torch.stack(out).mean().item()

g = torch.Generator().manual_seed(0)
for path in sys.argv[1:]:
    m = Net(); m.load_state_dict(torch.load(path)["model"])
    E = m.emb.weight[:N].detach().double()
    E = E - E.mean(0)
    inv = torch.tensor([(b, pow(b, N - 2, N)) for b in range(1, N)])
    neg = torch.tensor([(b, (-b) % N) for b in range(N)])
    perm = torch.randperm(N, generator=g)
    rnd = torch.tensor([(b, perm[b].item()) for b in range(N)])
    for lam in (0.01, 0.1, 1.0):
      print(f"{path.split('/')[-1]:>22} ridge {lam}  held-out R²:  b→b⁻¹ {r2_heldout(E, inv, g, lam=lam):+.2f}   b→−b {r2_heldout(E, neg, g, lam=lam):+.2f}"
          f"   random b→π(b) {r2_heldout(E, rnd, g, lam=lam):+.2f}   (embedding rank 99% energy: "
          f"{int((torch.linalg.svdvals(E).pow(2).cumsum(0) / torch.linalg.svdvals(E).pow(2).sum() < .99).sum()) + 1})")
