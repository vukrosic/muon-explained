"""Diagnostic (not a speedrun entry, it uses an SVD): does Muon push weak directions too hard?

Record 3 settings. Each step, the update keeps its arrows (U, V from an exact SVD of the momentum matrix) and gets
strengths S^p, then is scaled to the same total size as record 3's 2-round output on that step. So only the shape of the
strengths changes; learning rates and step sizes stay record 3's. p = 0: every direction pushed equally (exact Muon).
p > 0: weak directions pushed less. Record 3 itself (2 rounds) is in runs.jsonl for seeds 0 to 2.
Usage: python3 rare_push.py P SEED     -> one line appended to rare_push.jsonl
"""
import contextlib, json, os, sys
import torch
import speedrun as S

P, SEED = float(sys.argv[1]), int(sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__))
two_rounds = S.flatten


def shaped(G, v=None):
    ref = two_rounds(G, v)                                   # record 3's update, used only for its total size
    U, s, Vh = torch.linalg.svd(G, full_matrices=False)
    O = (U * s.clamp_min(1e-30) ** P) @ Vh
    return O * (ref.norm() / (O.norm() + 1e-12))


S.flatten = shaped
S.Guard = contextlib.nullcontext                            # the SVD is the point of this diagnostic
S.SEED = SEED
S.HERE = os.path.join(HERE, "rare_push_tmp")                 # keep speedrun's runs.jsonl for legal runs only
os.makedirs(S.HERE, exist_ok=True)
v, gflop = S.main(f"rare push p={P:g} seed={SEED}")
with open(os.path.join(HERE, "rare_push.jsonl"), "a") as f:
    f.write(json.dumps(dict(p=P, seed=SEED, val=round(v, 5))) + "\n")
