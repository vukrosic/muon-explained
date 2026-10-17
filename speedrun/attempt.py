"""Try a change without editing speedrun.py: python3 attempt.py "description" ROUNDS=2 MUON_LR=0.014
Each NAME=value overrides that setting in speedrun.py's OPTIMIZER section; the run is logged in runs.jsonl.
A record is copied into speedrun.py itself, so speedrun.py always holds the current record."""
import sys
import speedrun as S

sets = sys.argv[2:]
for kv in sets:
    k, v = kv.split("=", 1)
    assert hasattr(S, k), k
    assert k not in ("STEPS", "SEED", "BASELINE_GFLOP"), f"{k} is fixed by the rules"
    setattr(S, k, eval(v))
S.main(sys.argv[1], " ".join(sets))
