# Experiments

Small, fast tests of the ideas in the [main README](../README.md). Each training run takes about one minute on a laptop CPU. Shared data, model and training loop: [train.py](train.py).

| # | question | answer | details |
|---|---|---|---|
| 2 | Does every direction need an equal push? | Barely pushing weak directions hurts; pushing them somewhat less does not. | [exp2-equal-push.md](exp2-equal-push.md) |

Experiment 1 (Muon with 2 to 5 rounds against exact SVD) ran on a private text and will be added after rerunning it on this public data.

Speed contest on the same model, changing only the optimizer: [speedrun](../speedrun/README.md).
