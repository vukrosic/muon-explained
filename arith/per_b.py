"""Rare-op held-out accuracy split by how many rare training pairs share the same b (and the same a)."""
import sys, collections
import torch
from arith import Net, N
for path in sys.argv[1:]:
    d = torch.load(path); m = Net(); m.load_state_dict(d["model"])
    tr, te = d["tr"][3], d["te"][3]
    cb = collections.Counter(tr[:, 2].tolist()); ca = collections.Counter(tr[:, 0].tolist())
    with torch.no_grad():
        ok = (m(te[:, :3]).argmax(1) == te[:, 3]).float()
    print(path, f"rare pairs {len(tr)}, held-out {ok.mean():.2f}")
    for name, c, col in (("b", cb, 2), ("a", ca, 0)):
        n = torch.tensor([c.get(v, 0) for v in te[:, col].tolist()])
        line = []
        for lo, hi in ((0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 99)):
            sel = (n >= lo) & (n <= hi)
            if sel.sum():
                line.append(f"{lo}-{hi}: {ok[sel].mean():.2f} ({int(sel.sum())})")
        print(f"   by train pairs with same {name}:  " + "   ".join(line))
