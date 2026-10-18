"""Muon lr times MULT for the first N steps, record 3 settings otherwise (legal: no extra compute).
Usage: python3 early_boost.py N MULT SEED     -> one line appended to early_boost.jsonl"""
import json, os, sys
import speedrun as S

N, MULT, SEED = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
HERE = os.path.dirname(os.path.abspath(__file__))
S.EARLY_N, S.EARLY_MULT, S.SEED = N, MULT, SEED
S.HERE = os.path.join(HERE, "early_boost_tmp")               # keep runs.jsonl for seed-0 attempts only
os.makedirs(S.HERE, exist_ok=True)
v, gflop = S.main(f"early boost {N}x{MULT:g} seed={SEED}")
with open(os.path.join(HERE, "early_boost.jsonl"), "a") as f:
    f.write(json.dumps(dict(n=N, mult=MULT, seed=SEED, val=round(v, 5), gflop=round(gflop))) + "\n")
