# Research threads

One row per thread. Update the row when a result lands or a post goes out. Open questions without a running thread go in [QUESTIONS.md](QUESTIONS.md).

Status: **running** (experiments in progress), **open** (result in, next step known), **parked**, **done**.

| # | thread | status | headline result so far | next step | posted on X |
|---|---|---|---|---|---|
| 1 | Muon burst, arithmetic | running | 100 Muon steps when × arrives: × held-out 0.76 (8/8 runs) vs 0.04 for AdamW alone, fresh model 0.73. Five published fixes ≤ 0.09. 50k steps at constant lr: AdamW at hidden lr 0.003 stays at × 0.03 (4 runs, 2 seeds); at lr 0.001 it learns × alone (0.44–0.61 at 9k, 0.81–0.96 at 20k, 1 seed). Burst and fresh model at constant lr 0.003 peak at 0.60–0.74, then collapse to ≤ 0.07. | Burst vs AdamW at lr 0.001 over the same 9k steps; dose, timing, mechanism controls. Folder [burst/](burst/) | draft only, on hold: "stuck" holds only at lr 0.003 |
| 2 | Muon burst, language model | running | none yet | Find a data change that makes AdamW stuck in a char transformer, then run the burst arms. Folder [lmburst/](lmburst/) | no |
| 3 | Rare ops: Muon learns a 12% op that equal-shape alternatives don't | open | p = 0 (Muon) memorizes the 12% op's training pairs 0.98–1.00; p = 0.5 and p = 1 stay at 0.01–0.03. The 3% op: nobody. [arith](arith/README.md#results-seeds-0-and-1) | Held-out accuracy on the 12% op, more seeds, a bigger model | no |
| 4 | Continual learning: Adam loses plasticity, Muon doesn't | open | Adam's update rank 205 → 12 over 8 phases, Muon 186–207. Adam shows dead units, growing weights, falling rank (Dohare's signs); Muon none. [arith](arith/README.md#what-the-forgetting-literature-predicts-tested-on-this-model) | Link to 2609.24678 (Muon beats AdamW in continual learning); test on a transformer | no |
| 5 | Forgetting–plasticity trade-off | open | Muon learns a new op more easily and forgets a removed op 2–10× faster | Retention vs plasticity chart over lr and weight decay, 3 seeds. [QUESTIONS.md #1](QUESTIONS.md) | no |
| 6 | Char LM, repeated data changes (lmspike) | open | 12 relabelings: Muon stays at 1.74–1.76, Adam climbs to 1.94 (64–91% of the last layer dead). Cause: the old output table; resetting it at each change fixes both. Code ↔ English: nothing breaks. [lmspike 16–27](lmspike/README.md) | Overlaps thread 2; decide if the output-reset rule (only pays once the old output is worse than uniform) is a post | no |
| 7 | Partial push S^p (speedrun) | open | p = 0.25: 1.6215, p = 0.5: 1.6209, Muon p = 0: 1.6311 (3 seeds each). [speedrun](speedrun/README.md) | Cheap approximation of S^p without SVD | draft: x-posting-hq/2026-10-17-day5-muon-partway.md |
| 8 | Speedrun records | open | Record 3: 1.6297 at 7,399 GFLOP vs baseline 1.6768 at 14,647 (2 Newton–Schulz rounds, tuned lrs) | Early lr boost, 3 rounds at the cost of 2 | earlier days |
| 9 | Equal push (README section 7) | done | [exp2](experiments/exp2-equal-push.md) | none | draft: x-posting-hq/2026-10-17-muon-equal-push.md |
| 10 | Weak-direction conjecture (umbrella over 1, 3, 4, 7) | parked | Muon helps where the useful signal sits in weak directions: rare data, new data, units Adam turned off | Write it up only if threads 1–3 hold on a transformer | no |
