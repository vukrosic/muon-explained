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

## Run it

```bash
python3 speedrun.py "short description of the change"
```
