# Rare operations: what does Muon learn about them?

**Question.** Muon pushes every direction of the update equally, including weak directions that only a few examples in the batch ask for. The [speedrun test](../speedrun/README.md#does-muon-push-weak-directions-too-hard) found that pushing weak directions less (p = 0.25, 0.5) gives lower average loss. Average loss is dominated by common data. Does the model then know less about rare things?

**Task** ([arith.py](arith.py)). `a op b = ?` modulo 97, with 4 operations + − × ÷. Every operation has 4,704 training pairs and 4,705 held-out pairs. Batches draw the operations at different rates: 55%, 30%, 12%, 3%. Two orders, so rarity is not mixed up with difficulty:

- order A: + 55%, − 30%, × 12%, ÷ 3%
- order B: ÷ 55%, × 30%, − 12%, + 3%

Model: embeddings of the 3 tokens → three Muon matrices (192×256, 256×256, 256×256) → answer. 3000 steps × batch 512, Muon lr 0.02, momentum 0.95, weight decay 0.05, AdamW lr 0.01 on embeddings and output.

Muon matrices get U S^p V^T from an exact SVD, scaled to the total size of exact Muon. p = 0 is exact Muon (every direction pushed equally). p = 0.5 pushes weak directions less. p = 1 pushes them as the gradient says. Only the shape of the update changes, not its size.

![rare operation memorized only by Muon](../images/arith_rare.png)

## Results (seeds 0 and 1)

Accuracy at step 3000. "train" = its own training pairs, "test" = held-out pairs.

| p | order | seed | 55% op test | 30% op test | 12% op train | 3% op train |
|---|---|---|---|---|---|---|
| 0 | A | 0 | 0.92 | 0.92 | **1.00** | 0.02 |
| 0 | B | 0 | 0.70 | 0.72 | **1.00** | 0.02 |
| 0 | A | 1 | 0.30 | 0.31 | **0.99** | 0.02 |
| 0 | B | 1 | 0.80 | 0.81 | **0.98** | 0.02 |
| 0.5 | A | 0 | 0.50 | 0.59 | 0.03 | 0.02 |
| 0.5 | B | 0 | 0.99 | 0.99 | 0.01 | 0.01 |
| 0.5 | A | 1 | 0.38 | 0.42 | 0.03 | 0.02 |
| 0.5 | B | 1 | 0.74 | 0.78 | 0.01 | 0.01 |
| 1 | A | 0 | 0.83 | 0.88 | 0.03 | 0.02 |
| 1 | B | 0 | 0.04 | 0.06 | 0.01 | 0.01 |

Guessing gives 0.01.

**What it means.**

- **Only Muon learns the 12% operation.** With equal pushes it answers 98–100% of that operation's training pairs, in all 4 runs. With p = 0.5 or 1 it stays at 1–3%, in all 6 runs. So pushing weak directions less does make the model know less about the rarer operation.
- **Rarity decides, not difficulty.** In order A, × and ÷ are the rare ones and are not learned. In order B, ÷ and × are common and learn the rule (70–99% on unseen pairs), while − and + are rare and are not learned.
- **Nobody learns the 3% operation** in 3000 steps.
- **Nobody learns a rule for the 12% operation**, only its training pairs: held-out accuracy stays at 0.01–0.03.
- **Common operations:** no clear winner. Held-out accuracy swings from 0.30 to 0.99 between seeds and orders for the same p.
- **Muon memorizes the 12% operation late,** after step 1700. The learning rate starts falling at step 1500.

**Limits.** 2 seeds, one learning rate for every p, one tiny model, one task. 3000 steps is too short for the 3% operation.

**How this fits the speedrun result.** On text, p = 0.25–0.5 gave about 0.01 lower average loss. Here, at 12% frequency, the same change loses an operation completely. Lower average loss and knowing less about rare data can both be true.

## Run it

```bash
WD=0.05 python3 arith.py 0 "+-*/" 0     # p, order (most common first), seed -> one line in arith.jsonl
python3 plot_arith.py
```

arith.jsonl also has 3 exploratory p = 0 runs at weight decay 1 and 3; the chart and table use weight decay 0.05.

## Where does Muon store the rare operation?

**Question.** If we delete the weakest directions of the trained weights, does only the rare operation break?

**Setup** ([cut.py](cut.py)). Two models trained with Muon (p = 0, seed 0), one per rarity order. In all 3 Muon matrices, each weight matrix is split into its singular directions, and a share of the weakest (or the strongest) is set to zero. Then each operation's accuracy on its training pairs is measured again. Results in [cut.jsonl](cut.jsonl).

![deleting weak vs strong directions](../images/arith_cut.png)

| weakest directions deleted | 0% | 30% | 40% | 50% | 60% | 75% |
|---|---|---|---|---|---|---|
| 55% op (order A / B) | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.98 | 0.96 / 0.91 | 0.90 / 0.72 | 0.58 / 0.33 |
| 12% op (order A / B) | 0.99 / 1.00 | 0.98 / 0.99 | 0.86 / 0.86 | **0.47 / 0.40** | 0.15 / 0.14 | 0.04 / 0.02 |

| strongest directions deleted | 0% | 1% | 2% | 5% |
|---|---|---|---|---|
| 55% op (order A / B) | 1.00 / 1.00 | 0.30 / 0.52 | 0.12 / 0.37 | 0.04 / 0.11 |
| 12% op (order A / B) | 0.99 / 1.00 | 0.74 / 0.61 | 0.30 / 0.40 | 0.08 / 0.09 |

**What it means.**

- **The weakest 30% of directions hold nothing.** Deleting them leaves every operation at 98–100%.
- **The rare operation lives in the middle directions.** Deleting the weakest 50% drops it to 0.47 and 0.40, while the 55% operation keeps 0.96 and 0.91.
- **The common operations lean on the strongest directions a bit more.** Deleting the top 1% leaves the common op at 0.30 / 0.52 and the rare op at 0.74 / 0.61. With order B the gap is small.
- **Everything shares the top directions.** Deleting the strongest 5% breaks every operation.

**Limits.** One seed per order. The rare operation is only memorized, not learned as a rule, so this shows where memorized facts sit, not where rules sit.

## A rare operation rides on its partner

**Question.** At what frequency does a rare operation stop being memorized and start being learned as a rule?

**Setup.** 3 common operations share the batch equally, 1 rare operation gets 2, 4, 8, 12, 16 or 25%. Order A: ÷ rare. Order B: + rare. Weight decay 1 for every optimizer (at 0.05 Adam memorizes everything and generalizes nothing, see [adam_lr.jsonl](adam_lr.jsonl)). Muon p = 0, Muon p = 0.5, AdamW (lr 0.003 on the hidden matrices, best of 4). 3000 steps, seed 0. Results in [freq.jsonl](freq.jsonl), [partner.jsonl](partner.jsonl), [adam_long.jsonl](adam_long.jsonl).

![rare vs partner](../images/arith_partner.png)
![rare accuracy vs frequency](../images/arith_freq.png)

**What it means.**

- **Frequency barely matters. The partner decides.** ÷ is × by an inverse, + is − with a sign flip. Across 56 runs the rare operation's held-out accuracy equals its partner's (correlation 0.991, mean gap 0.03), from 2% to 25% of the batch.
- **Remove the partner and the rare operation is never learned.** With × (or −) taken out of the data, ÷ (or +) at 2% or 8% stays at 1–5% held-out for all 3 optimizers. With the partner in, Muon p = 0 reaches 49% (÷) and 92% (+) at 2%.
- **So the rare operation is not learned on its own.** The model learns the partner, then the rare operation's few examples only need to say "this is the partner, with a twist".
- **Muon is faster, not different.** In 3000 steps Muon p = 0 learns the partner in all 12 runs, Adam in 4. With 9000 steps Adam learns partner and rare operation too (6 of 8 settings reach 0.68–1.00). With exact SVD, 3000 Muon steps take as long as 9000 Adam steps (84 s vs 90 s).
- **Earlier result explained.** In the first table, ÷ (3%) and × (12%) were both rare, so neither had a learned partner, and neither was ever learned as a rule.

**Limits.** Seed 0 only, one task. Accuracy jumps between seeds (see the first table), so single points on the frequency chart are noisy; the partner correlation holds across all 56.

## How much rare data does the partner save?

**Question 1.** With the partner fully learned, how few distinct rare examples are enough? The rare op stays at 4% of the batch, but keeps only K of its 4704 training pairs ([few.jsonl](few.jsonl)).

**Question 2.** Without a partner, how common must an operation be to be learned? ([lone.jsonl](lone.jsonl))

![few-shot and lone](../images/arith_few.png)

| distinct rare pairs (partner known, 4% of batch) | 3–100 | 300 | 500–700 | 1000 | 1500+ |
|---|---|---|---|---|---|
| rare op held-out (4 runs: 2 orders × Adam 9000 / Muon 3000) | 0.01–0.08 | 0.10–0.19 | 0.10–0.87 | 0.43–0.99 | 0.53–1.00 |

**What it means.**

- **The partner does not make the rare op few-shot.** 3 to 300 pairs are memorized perfectly (train 1.00) and stay at 1–19% held-out. The rule appears between 700 and 1500 distinct pairs, about 15–30% of all possible pairs.
- **It is all at once, not per number.** With 300 pairs, held-out pairs whose b never appeared in training score 0.23, the same as pairs whose b appeared 3–5 times (0.21) ([per_b.py](per_b.py)). The model does not learn "the negative of 7" one number at a time.
- **Without the partner, 4% is never enough.** All 4704 pairs, 9000 steps: 1–3% held-out, and not even memorized (train 0.01–0.04). The same op with only 500 pairs is memorized (train 1.00). Many different rare examples are harder to memorize than few.
- **A lone op needs 16–33% of the batch.** + : 16% (Adam 9000 steps, 0.97). ÷ : 33% (1.00). With a partner, 2–4% was enough: the partner cuts the needed frequency 4–8×.
- **Muon memorizes rare data more, Adam generalizes it more.** At 4–8%, Muon reaches 0.19–0.89 train accuracy on the lone op, Adam 0.01–0.12. At 16–25% with 9000 steps, Adam learns + (0.97–1.00 held-out), Muon reaches 0.39–0.73.

**Limits.** Seed 0, one task, one Adam learning rate.

## How the model does it

**Question.** Does the network really compute ÷ as "× by the inverse", or does it just reuse some parts?

**Setup** ([mech.py](mech.py), results in [mech.jsonl](mech.jsonl)). Four trained models: orders A and B, each with all four ops equally common (Muon) and with the op rare at 4% (Adam, 9000 steps). For every held-out pair, three inputs that all have the same correct answer:

- **rare:** a ÷ b
- **translated:** a × b⁻¹ (the partner op, with b already inverted)
- **control:** a different pair a′ × c′ with the same answer

At each layer, the cosine between the hidden state of "rare" and "translated", and between "rare" and "control" (centered: the mean hidden state is subtracted first). Same for + against − with −b.

| model | op | first layer | second layer (h1) | third layer (h2) |
|---|---|---|---|---|
| A, all equal, Muon | ÷ | +0.31 / −0.14 | **+0.76 / +0.11** | +0.96 / +0.66 |
| A, all equal, Muon | + | +0.48 / −0.10 | **+0.75 / +0.18** | +0.94 / +0.67 |
| A, ÷ rare, Adam | ÷ | +0.11 / −0.21 | **+0.64 / +0.06** | +0.90 / +0.86 |
| B, all equal, Muon | ÷ | +0.44 / −0.09 | **+0.95 / +0.20** | +0.98 / +0.64 |
| B, all equal, Muon | + | +0.44 / −0.12 | **+0.88 / +0.22** | +0.97 / +0.70 |
| B, + rare, Adam | + | +0.24 / −0.14 | **+0.93 / +0.27** | +0.96 / +0.95 |

(each cell: rare vs translated / rare vs control)

**What it means.**

- **By the second layer, a ÷ b looks like a × b⁻¹.** Cosine 0.64–0.95 with the translated input, 0.06–0.27 with another pair that has the same answer. So the match is not just "same answer": the network has rewritten the question into the partner's question.
- **It happens in every model, even when all four ops are equally common.** The network finds the shortcut on its own; rarity only makes it necessary.
- **By the last layer everything converges** (control also reaches 0.64–0.97), because all three inputs must give the same answer.

**Why ÷ is harder than +** ([linmap.py](linmap.py)). Fit a linear map from each number's embedding to the embedding of its negative (b → −b) on half the numbers, test on the other half (ridge, λ = 0.1):

| model | b → −b | b → b⁻¹ | b → random number |
|---|---|---|---|
| A, all equal, Muon | **+0.78** | +0.00 | −0.80 |
| A, ÷ rare, Adam | **+0.43** | −0.24 | −0.82 |
| B, all equal, Muon | **+0.79** | +0.00 | −0.75 |
| B, + rare, Adam | **+0.56** | −0.03 | −0.64 |

(held-out R²; 1 is perfect, 0 is no better than guessing the average)

- **Negation is (roughly) one linear map on the embeddings. Inversion is not.** Known result (Nanda et al. 2023): these networks store a number as waves cos(2πkb/97). Negation just flips the sign of the sine part, the same linear step for every number. Inversion has no such step, so the first layer must learn it as a lookup, number by number.
- This fits every earlier table: + needs less data and fewer steps than ÷, as rare op and as lone op.

**When does the rare op click?** Over 34 runs where both partner and rare op pass 50% held-out ([freq.jsonl](freq.jsonl), [adam_long.jsonl](adam_long.jsonl), [few.jsonl](few.jsonl)):

- **It clicks right after its partner.** Median lag 150 steps; 0–100 steps at high frequency, up to 4650 steps (Adam, ÷ at 4%) at low frequency or with few pairs.
- **It skips most of the memorizing.** At the moment held-out accuracy first passes 50%, how much higher is train accuracy?

| op | runs | train − held-out at that moment (median) |
|---|---|---|
| lone op (no partner) | 10 | 0.43 |
| partner op | 27 | 0.33 |
| common op | 77 | 0.26 |
| **rare op with partner** | 27 | **0.11** |

Ops learned from scratch first memorize their training pairs, then generalize (grokking). The rare op with a partner generalizes almost as soon as it fits: it reuses a rule instead of building one.

**Limits.** Four models, seed 0. Cosines are an average over held-out pairs; they show the rewrite, not the circuit that does it.
