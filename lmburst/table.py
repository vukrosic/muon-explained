"""Summarize a runs jsonl: per (shift, arm, settings), ranges over seeds of new-data loss (task: line accuracy), old-code loss, MLP units off.
Usage: python3 table.py [runs.jsonl] [shift ...]"""
import json, sys, collections, os

HERE = os.path.dirname(os.path.abspath(__file__))
fn = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".jsonl") else "runs.jsonl"
only = [a for a in sys.argv[1:] if not a.endswith(".jsonl")]
rows = [json.loads(l) for l in open(os.path.join(HERE, fn))]
grp = collections.defaultdict(list)
for r in rows:
    if only and r["shift"] not in only: continue
    extra = {k: v for k, v in r["kw"].items() if k not in ("out", "dseed")}
    grp[(r["shift"], r["arm"], json.dumps(extra), r["lr"], r.get("b2"))].append(r)


def at(r, s):
    for e in r["log"]:
        if e["step"] == s: return e
    return None


def rng(v, f="{:.3f}"):
    v = [x for x in v if x is not None]
    if not v: return "-"
    return (f.format(min(v)) + "–" + f.format(max(v))) if len(v) > 1 else f.format(v[0])


def newv(e, shift):
    if e is None: return None
    return e["task"][0] if shift == "task" else e["new"]["new"]


for (shift, arm, extra, lr, b2), rs in sorted(grp.items()):
    end = rs[0]["log"][-1]["step"]
    steps = [s for s in (100, 500, 1000, end) if s <= end]
    cols = "  ".join(f"@{s} {rng([newv(at(r, s), shift) for r in rs])}" for s in steps)
    old = rng([r["log"][-1]["old"] for r in rs])
    off = " / ".join(rng([max(o[0] for o in at(r, s)["off"]) if at(r, s) else None for r in rs], "{:.2f}") for s in (0, 100, 1000))
    xtra = ""
    if shift == "task": xtra = f"  task-letter loss end {rng([r['log'][-1]['task'][1] for r in rs])}"
    print(f"{shift:>7} {arm:>6} {extra} lr {lr:g} b2 {b2} n={len(rs)}  new {cols}  old end {old}  off max-block @0/100/1000 {off}{xtra}")
