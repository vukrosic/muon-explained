"""Does the model compute a ÷ b as a × b⁻¹ (and a + b as a − (−b))?
For every pair, compare hidden states (centered, cosine) of:
  rare      (a, ÷, b)
  translated (a, ×, b⁻¹)       same answer, the partner op on the translated number
  control   (a', ×, c')         same answer, the partner op on a random other pair
If translated >> control in a layer, the model has already turned ÷ b into × b⁻¹ there."""
import sys, json, os
import torch, torch.nn.functional as F
from arith import Net, N

def layers(m, x):
    e = m.emb(x).flatten(1)
    pre0 = m.inp(e); h0 = F.relu(pre0); h1 = F.relu(m.h1(h0)); h2 = F.relu(m.h2(h1)); lo = m.out(h2)
    return dict(pre0=pre0, h0=h0, h1=h1, h2=h2, logits=lo)

def cos(u, v, mu):
    u, v = u - mu, v - mu
    return F.cosine_similarity(u, v, dim=1).mean().item()

tok = lambda op: N + "+-*/".index(op)
g = torch.Generator().manual_seed(0)
out = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mech.jsonl"), "a")
for path in sys.argv[1:]:
    d = torch.load(path); m = Net(); m.load_state_dict(d["model"]); m.eval()
    for rare, part, trans, ans in (("/", "*", lambda b: pow(b, N - 2, N), lambda a, b: a * pow(b, N - 2, N) % N),
                                    ("+", "-", lambda b: (-b) % N, lambda a, b: (a + b) % N)):
        A, Bb = torch.meshgrid(torch.arange(N), torch.arange(1, N), indexing="ij")
        A, Bb = A.flatten(), Bb.flatten()
        y = torch.tensor([ans(a, b) for a, b in zip(A.tolist(), Bb.tolist())])
        T = torch.tensor([trans(b) for b in Bb.tolist()])
        # control: random a' (nonzero for ×), then c' with a' op c' = y
        if part == "*":
            a2 = torch.randint(1, N, (len(y),), generator=g)
            c2 = torch.tensor([yy * pow(aa, N - 2, N) % N for yy, aa in zip(y.tolist(), a2.tolist())])
        else:
            a2 = torch.randint(0, N, (len(y),), generator=g)
            c2 = (a2 - y) % N
        xs = {k: torch.stack([a, torch.full_like(a, tok(o)), b], 1) for k, (a, o, b) in
              dict(rare=(A, rare, Bb), translated=(A, part, T), control=(a2, part, c2)).items()}
        with torch.no_grad():
            H = {k: layers(m, x) for k, x in xs.items()}
            allx = torch.cat(list(xs.values()))
            mu = {k: v.mean(0) for k, v in layers(m, allx).items()}
            ok = {k: (H[k]["logits"].argmax(1) == y).float().mean().item() for k in xs}
        res = {L: (round(cos(H["rare"][L], H["translated"][L], mu[L]), 3), round(cos(H["rare"][L], H["control"][L], mu[L]), 3))
               for L in mu}
        print(f"{os.path.basename(path):>22} {rare} vs {part}: acc rare {ok['rare']:.2f} translated {ok['translated']:.2f}   "
              + "  ".join(f"{L}: {t:+.2f} / {c:+.2f}" for L, (t, c) in res.items()))
        out.write(json.dumps(dict(model=os.path.basename(path), rare=rare, partner=part, acc=ok, cos=res)) + "\n")
