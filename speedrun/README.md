# Laptop speedrun

Train the tiny character model from the [experiments](../experiments/README.md) on a fixed number of tokens and reach the **lowest validation loss**, by changing only the optimizer.

## Rules

- **Fixed:**
  - the data (Python standard library source, 90% train / 10% validation);
  - the model (three 512-wide layers);
  - **1000 steps × batch 512 = 512,000 training predictions**;
  - seed 0, so every run starts from the same weights and sees the same batches in the same order;
  - how loss is measured: mean next-character loss over every position of the validation text.
- **Compute cap:** the whole training run may use at most as many matrix-multiply FLOPs as the baseline (14,647 GFLOP, counted by PyTorch's `FlopCounterMode`). Matrix decompositions (SVD, eigh, QR, inverse, ...) are refused. Without the cap, extra optimizer work (say 50 Newton–Schulz rounds) would be free.
- **Free:** everything in the OPTIMIZER section of [speedrun.py](speedrun.py): which optimizer, Newton–Schulz rounds and coefficients, learning rates, momentum, weight decay, schedule.
- **Score:** validation loss. With a fixed seed it is identical on every repeat and does not depend on how fast or busy the computer is. Wall-clock seconds are printed but not scored.
- Every run is logged in [runs.jsonl](runs.jsonl).

## Records

| # | change | validation loss | seed 1 check | GFLOP |
|---|---|---|---|---|
| 1 | baseline: Muon (5 rounds, lr 0.01, momentum 0.95), AdamW lr 0.003, lr decays over the last 50% | 1.6768 | 1.6649 | 14,647 |
| 3 | 2 rounds, Muon lr 0.02, Muon weight decay 0.05, AdamW lr 0.036, lr decays over the last 85% | **1.6297** | 1.6201 | 7,399 |

Record 2 belonged to the earlier time-based version below.

**Record 3, how it was found.** One setting at a time, kept only if the loss dropped, then pushed further until it got worse:

| step | change | loss |
|---|---|---|
| start | baseline | 1.6768 |
| 1 | 2 rounds instead of 5, Muon lr 0.01 → 0.02 (fewer rounds shrink the update, so lr doubles) | 1.6671 |
| 2 | AdamW lr on the embedding and output layer 0.003 → 0.012 | 1.6623 |
| 3 | lr decay over the last 85% of steps instead of 50% | 1.6537 |
| 4 | Muon weight decay 0.05 (the lr-scaled decay Moonlight uses) | 1.6439 |
| 5 | AdamW lr 0.012 → 0.036 | **1.6297** |

The seed 1 check (not part of the score) shows the gain is not luck on seed 0: 0.047 better on seed 0, 0.045 better on seed 1.

**Tried and worse** (all in runs.jsonl):

- More rounds with everything else tuned: 3 rounds 1.6883, 5 rounds 1.6684, against 1.6540 for 2 rounds with the same settings. Exact orthogonalization is not what this model needs.
- 1 round with Polar Express's first coefficients: 1.6411 vs 1.6409 for 2 rounds at the same settings, at a third less compute. A tie.
- Weight decay 0.2 (1.6805) or more; AdamW lr 0.05 (1.6354); AdamW beta2 0.99 or 0.95; beta1 0.8; momentum 0.9 or 0.97; cubic rounds; dividing by the largest strength instead of the total size.

## Earlier version: time to reach 1.68 (retired)

The first version scored training seconds to reach validation loss 1.68: baseline 24.1 s, record 2 (3 rounds) 15.8 s. It was retired because wall-clock time changes with whatever else runs on the laptop. A game in the background moved the same run from 5.6 s to 7.5 s. Those runs are in [runs-v1-wallclock.jsonl](runs-v1-wallclock.jsonl).

## Run it

```bash
python3 speedrun.py "short description of the change"
python3 attempt.py "description" ROUNDS=1 MUON_LR=0.025   # try settings without editing speedrun.py
```
