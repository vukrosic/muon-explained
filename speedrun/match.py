"""Shape of ROUNDS Newton-Schulz rounds, step size of MATCH rounds (legal: the MATCH-round size is read on the way).
Record 3 settings otherwise. Usage: python3 match.py ROUNDS MATCH SEED [NAME=value ...]   -> one line in match.jsonl"""
import json, os, sys
import speedrun as S

R, M, SEED = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
HERE = os.path.dirname(os.path.abspath(__file__))
S.ROUNDS, S.MATCH, S.SEED = R, M, SEED
for kv in sys.argv[4:]:
    k, v = kv.split("=", 1); assert hasattr(S, k), k; setattr(S, k, eval(v))
S.HERE = os.path.join(HERE, "match_tmp")                    # keep runs.jsonl for seed-0 attempts only
os.makedirs(S.HERE, exist_ok=True)
v, gflop = S.main(f"rounds {R} at {M}-round size seed={SEED} {' '.join(sys.argv[4:])}")
with open(os.path.join(HERE, "match.jsonl"), "a") as f:
    f.write(json.dumps(dict(rounds=R, match=M, seed=SEED, extra=sys.argv[4:], val=round(v, 5), gflop=round(gflop))) + "\n")
