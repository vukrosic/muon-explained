"""Why can't a trained model learn a new op? Per layer: share of ReLU units never active on the new op's inputs,
weight size, effective rank (exp of the entropy of normalized singular values), and the new op's starting gradient size.
Usage: python3 plastic_diag.py MODEL.pt OP_INDEX ..."""
import sys, torch, torch.nn.functional as F
import arith as A
for path, i in zip(sys.argv[1::2], map(int, sys.argv[2::2])):
    ck = torch.load(path); m = A.Net(); m.load_state_dict(ck["model"])
    x = torch.cat([ck["tr"][i], ck["te"][i]])
    with torch.no_grad():
        h0 = F.relu(m.inp(m.emb(x[:, :3]).flatten(1))); h1 = F.relu(m.h1(h0)); h2 = F.relu(m.h2(h1))
    row = []
    for name, h, w in (("inp", h0, m.inp.weight), ("h1", h1, m.h1.weight), ("h2", h2, m.h2.weight)):
        s = torch.linalg.svdvals(w.detach()); q = s / s.sum()
        row.append(f"{name}: dead {(h.amax(0) <= 0).float().mean():.2f} on {(h > 0).float().mean():.2f} |W| {w.norm():5.1f} erank {torch.exp(-(q * q.log()).sum()):5.1f}")
    loss = F.cross_entropy(m(x[:, :3]), x[:, 3]); m.zero_grad(); loss.backward()
    g = " ".join(f"{n}:{p.grad.norm():.3f}" for n, p in (("inp", m.inp.weight), ("h1", m.h1.weight), ("h2", m.h2.weight)))
    print(f"{path.split('/')[-1]:>24} op{i} loss {loss:.2f}  " + " | ".join(row) + "  grad " + g)
