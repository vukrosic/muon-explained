# Laptop speedrun

Train the tiny character model from the [experiments](../experiments/README.md) to a validation loss of **1.68** in the least training time, by changing only the optimizer.

## Rules

- **Fixed:**
  - the data (Python standard library source, 90% train / 10% validation);
  - the model (three 512-wide layers);
  - batch size 512;
  - seed 0, so every run starts from the same weights and sees the same batches in the same order;
  - 8 CPU threads;
  - how loss is measured: mean next-character loss over every position of the validation text.
- **Free:** everything in the OPTIMIZER section of [speedrun.py](speedrun.py). That includes which optimizer, Newton–Schulz rounds and coefficients, learning rates, momentum, schedule and number of steps.
- **Score:** training seconds (model setup and validation not counted). Run twice and report the mean, because the same run varies by up to 3 s on the laptop.
- A run counts only if its validation loss is 1.68 or lower. With a fixed seed, the loss is identical on every repeat.
- Every run is logged in [runs.jsonl](runs.jsonl).

## Records

Apple M4, PyTorch 2.8.0, Python 3.12.8.

| # | change | steps | validation loss | training seconds (2 runs) | mean |
|---|---|---|---|---|---|
| 1 | baseline: Muon, 5 rounds, lr 0.01 | 1000 | 1.6768 | 25.5, 22.6 | 24.1 |
| 2 | 3 rounds instead of 5 | 1000 | 1.6736 | 15.8, 15.7 | 15.8 |

**Record 2.** Newton–Schulz takes 16 ms of each step with 5 rounds and 9.6 ms with 3, because this model's three 512×512 matrices see only 512 examples per step. That is about two thirds of the baseline's training time. 3 rounds cut the time by 34% and the loss did not get worse (1.6736 vs 1.6768).

## Candidates waiting for a clean timing

These reached the target, but their times were measured while a game used about 1.7 CPU cores, so the seconds are too high and vary. Their validation losses are exact (fixed seed). Each becomes a record only after 2 runs on an idle laptop.

| change from record 2 | steps | validation loss | seconds (busy laptop) |
|---|---|---|---|
| 2 rounds, Muon lr 0.02 | 1000 | 1.6671 | 12.5 |
| 2 rounds, Muon lr 0.02, AdamW lr 0.012 | 920 | 1.6765 | 11.9 |
| 1 round with Polar Express's first coefficients (8.29, −23.60, 17.30), Muon lr 0.025, AdamW lr 0.012 | 920 | 1.6775 | 8.9 |
| **0 rounds** (update = momentum gradient ÷ its total size), lr 0.15, AdamW lr 0.012 | 1050 | 1.6773 | 7.5 |

What the search found so far (every run is in [runs.jsonl](runs.jsonl)):

- **Fewer steps cost more than they save.** With 2 rounds, going from 1000 to 900 steps raised the loss from 1.6671 to 1.6937. Cheaper steps beat fewer steps.
- **A higher AdamW learning rate on the embedding and output layer (0.012 instead of 0.003) is what made 1 and 0 rounds possible.** 1 round went from 1.7052 to 1.6807; 0 rounds then reached 1.6872 at 1000 steps.
- **With 0 rounds a step costs about 6 ms instead of 12.5 ms,** so 1050 steps of plain normalized momentum beat 920 steps of Muon.
- Did not help: cubic rounds, momentum 0.9 or 0.97, dividing by the largest strength instead of the total size, a longer lr decay.

## Run it

```bash
python3 speedrun.py "short description of the change"
python3 attempt.py "description" ROUNDS=0 MUON_LR=0.15 ADAM_LR=0.012 STEPS=1050   # try settings without editing speedrun.py
```
