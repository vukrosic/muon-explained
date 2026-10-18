# Character model: why does Adam lose to Muon?

**Question.** In the arithmetic task ([arith](../arith/README.md)), Adam loses units to loss spikes and then cannot learn a rare op. Does the same happen in the speedrun's character model, and how much of Muon's lead over Adam does it explain?

**Setup** ([lmspike.py](lmspike.py)): same data and model as the [speedrun](../speedrun/README.md) (16 characters in, three 512-wide ReLU layers, batch 512, 1000 steps, lr decays over the last 85%). Hidden matrices use Muon (2 rounds, lr 0.02) or AdamW. The embedding and output layers always use AdamW, lr 0.003 unless noted. A unit is dead when it is off on all 4096 fixed validation inputs. 3 seeds per setting; validation loss is the mean over 3 seeds. Logs: lmspike.jsonl, lmwarm.jsonl, lmswitch.jsonl, lmemb.jsonl. Chart: [plot_lmspike.py](plot_lmspike.py).

![dead units and Muon start](../images/lm_start.png)

## 1. Adam kills a quarter to three quarters of layer 2 in the first 50 steps; Muon kills none

| hidden optimizer | layer-2 units dead at the end | validation loss |
|---|---|---|
| Muon, lr 0.02 | **0%** | **1.686** |
| Adam lr 0.001, no warmup | 13–18% | 1.800 |
| Adam lr 0.003, no warmup | 46–48% | 1.774 |
| Adam lr 0.01, no warmup | 74–80% | 1.862 |
| Adam lr 0.003, warmup 100 | 12–16% | 1.719 |
| Adam lr 0.01, warmup 100 | 28–33% | 1.752 |

- **The deaths happen early, then stop.** With Adam lr 0.003 and no warmup, the dead share goes 0% → 10% at step 2 → 46% at step 16 and stays at 44–49% until step 1000. At lr 0.01 the very first Adam step sends the loss from 4.56 to 15.2.
- **No later loss spikes.** No run has a loss jump of more than 1.0 above the recent median after step 50. The arithmetic task's mid-training spikes do not appear here in 1000 steps.
- **Gradient clipping at 1 does not prevent it** (lr 0.003: 44–51% dead). Unlike in arithmetic, the damage comes from many medium steps, not one huge one.
- **Rare characters do not tell anything at this length.** Loss on the rarest characters (2% of the text) is 5.6–6.1 for every optimizer, worse than guessing uniformly (4.56). None of them learns rare characters in 1000 steps.

## 2. Dead units here cannot be revived, not even by Muon

Switching an Adam run to Muon at step 25, 50, 100 or 300 leaves 45–52% of layer 2 dead (validation 1.769–1.795). In arithmetic 10 Muon steps revived 99% of units. The difference: the arithmetic units were off only on × inputs and still got a gradient from + and −. Here the dead units are off on every input, so their row of the gradient is exactly zero. Newton–Schulz maps a zero row to a zero row (checked: max 0.0), so Muon cannot move them either. **Muon prevents the deaths; it cannot undo them.**

Switching Muon → Adam at step 25 or 100 without warmup kills 27–68% of units, with loss spikes right after the switch (steps 101, 102): the same cold-restart death as in arithmetic. With a 50-step lr warmup after the switch, only 5–10% die.

## 3. Muon's lead comes from the first 25 steps

Muon for the first N steps, then Adam (lr 0.003, fresh state, 50-step warmup after the switch):

| N | 0 (Adam, warmup 50) | 5 | 10 | 25 | 100 | 300 | 1000 (Muon only) |
|---|---|---|---|---|---|---|---|
| validation loss | 1.726 | 1.714 | 1.706 | **1.693** | 1.693 | 1.690 | 1.686 |

25 Muon steps (2.5% of training) give 82% of the way from Adam to all-Muon. Steps 25 to 1000 add 0.007.

## 4. It is not Muon's flat update. It is how small the first steps are

What was used for those first 25 steps, then Adam (same switch and 50-step warmup):

| first 25 steps on the hidden matrices | validation loss |
|---|---|
| nothing (hidden matrices frozen) | 1.723 |
| Muon | 1.693 |
| gradient divided by its size, same step size as Muon | 1.695 |
| same, 4× smaller steps | 1.705 |
| same, 4× larger steps | 1.697 |
| **Adam's own direction**, scaled to Muon's step size | 1.700 |
| **plain Adam at a constant lr 0.0002** | 1.700 (1.697 with fresh Adam state at the switch) |

- **Any small gradient-following start works**, including Adam's own direction. Doing nothing does not.
- **Muon's early steps are tiny.** With 2 rounds, the update's size is only 1.7–5.4 (the gradient is close to low rank, so 2 rounds leave most strengths near 0). Times lr 0.02 that is 0.03–0.1 per step against weights of size 13. Adam at lr 0.003 moves each weight matrix by about 1.2 per step, 15–30× more. The matching Adam lr is 0.0001–0.0002.
- **beta2 is not the cause.** Adam warmup 50 with beta2 0.99 gives 1.723, with 0.95 1.734 (0.999: 1.726).

## 5. The usable fix for Adam: warm up only the hidden layers

In the warmup runs above, the embedding and output layers warmed up too. Warming only the hidden matrices:

| warmup steps | all layers warm up | only hidden layers warm up |
|---|---|---|
| 50 | 1.726 | 1.715 |
| 100 | 1.719 | 1.705 |
| 200 | – | **1.702** |
| 300 | 1.729 | – |

With the speedrun's embedding/output lr 0.036:

| | validation loss |
|---|---|
| Muon | **1.654** |
| Adam lr 0.003, all layers warm up (100) | 1.686 |
| Adam lr 0.003, hidden layers warm up (200) | 1.673 |
| Adam lr 0.002, hidden layers warm up (200) | 1.669 |
| Adam lr 0.003, hidden layers warm up (400) | 1.677 |
| Adam lr 0.005, hidden layers warm up (200) | 1.700 |

The Adam-to-Muon gap goes from 0.032 to 0.015: about half of Muon's lead in this model is the start, and an Adam schedule can get it.

**Limits (sections 1–5).** One tiny model, 1000 steps, 3 seeds (seed-to-seed spread about 0.01-0.02). Every start in section 4 except frozen beats Adam warmup 50 on all 3 seeds; differences under 0.01 between those starts are ties. Adam's lr and warmup were tuned on a coarse grid only. Why frozen hidden layers lose to tiny steps is not explained yet.

## 6. With GELU, Adam no longer kills units and the gap halves

ReLU outputs exactly 0 below zero, so a unit that is never positive gets no gradient and stays dead. GELU leaks a small gradient below zero. If dead units are Adam's problem, GELU should remove most of Adam's loss and leave Muon almost unchanged.

Embedding/output lr 0.01, each optimizer's hidden lr tuned on a grid, 3 seeds (dead shares are 3-seed means), data in [lmact.jsonl](lmact.jsonl):

| | ReLU | GELU |
|---|---|---|
| Muon, best lr (ReLU 0.02, GELU 0.03) | **1.649** | **1.627** |
| Adam lr 0.003, no warmup | 1.740 (47% of layer-2 units dead) | 1.637 (0.1% dead) |
| Adam, best warmup found | 1.672 (hidden warmup 200, 12% dead) | 1.637 (no warmup is best) |
| Adam minus Muon | 0.023 | **0.010** |

- **GELU fixes Adam's start by itself.** Plain Adam goes from 1.740 to 1.637. Hidden-layer warmup no longer helps: GELU Adam with warmup 50/100/200 gives 1.642/1.641/1.648.
- **GELU helps Muon much less** (1.649 → 1.627, 0.022) than plain Adam (0.103). Muon never had dead units to lose.
- **Muon still wins by 0.010** with GELU. That part of its lead is not about dead units.
- **Untuned lr can flip the result.** At embedding lr 0.003 with Muon lr 0.02, GELU Adam (1.656) beat Muon (1.668). Muon lr 0.04 gives 1.660 there, and embedding lr 0.01 gives Muon 1.627. Comparing optimizers at one lr per optimizer is not enough.

Other GELU grid points: Adam lr 0.002 / 0.004 with no warmup 1.649 / 1.663; Muon lr 0.02 / 0.04 at embedding lr 0.01 1.644 / 1.650. Leaky ReLU (slope 0.1) at embedding lr 0.003: Muon 1.692, Adam 1.690 to 1.694, untuned.

## 7. With GELU, Muon's lead is lost in Adam's first 10 steps

With no dead units left, what is the remaining 0.010? Same setup as section 6 (GELU, embedding lr 0.01, Muon lr 0.03, Adam lr 0.003), 3 seeds, data in [lmgswitch.jsonl](lmgswitch.jsonl) and [lmgstat.jsonl](lmgstat.jsonl), chart from [plot_gelu.py](plot_gelu.py).

![stable rank and switch curve](../images/lm_gelu.png)

| first hidden steps | then | validation loss |
|---|---|---|
| – | Muon all the way | **1.627** |
| 1 Adam step | Muon | 1.629 |
| 3 Adam steps | Muon | 1.629 |
| 10 Adam steps | Muon | 1.636 |
| 25 Adam steps | Muon | 1.637 |
| 100 Adam steps | Muon | 1.638 |
| – | Adam all the way | 1.637 |
| 10 Adam steps at 1/30 of the lr | Muon | 1.630 |
| 10 Muon steps at 30× the lr | Muon | 1.661 |

- **Ten Adam steps at the start cost 90% of Muon's lead**, and 900 Muon steps afterwards do not win it back. Worse than Muon on all 3 seeds (by 0.012, 0.009, 0.006).
- **It is the size of those steps, not their direction.** The same 10 Adam directions at 1/30 of the lr keep the lead (better than full-size on all 3 seeds). Muon's own directions at 30× the lr for 10 steps are worse than Adam all the way.
- **Why Adam's first step is so big** ([first_step.py](first_step.py)). The first gradient of each hidden matrix is nearly rank 1: 84–94% of its energy is in one direction. Adam's first update is lr × sign(gradient), and the sign of a rank-1 matrix is again rank 1 with every entry ±1, so its strength in that one direction is 512 × 0.003 ≈ 1.0–1.2. The whole starting matrix has top strength 1.15. Muon's first update has strength 0.036 (2 Newton–Schulz rounds leave a near-rank-1 input small).
- **What it does to the weights.** After 10 Adam steps the last hidden matrix's stable rank (how many directions carry its size) drops from 131 to 10. Muon keeps 123 at step 10 and still 27 at the end; Adam ends at 7. Switching to Muon after the 10 steps lifts it only to 19 by step 300.
- **Muon's directions still matter after the start.** Muon's step size each step with Adam's direction (best of lr 0.015/0.03/0.06/0.12) gives 1.653; with the raw gradient's direction 1.644. Both lose to Muon on all 3 seeds.
- **Late Adam costs less.** 25 / 100 / 500 Muon steps, then Adam: 1.637 / 1.633 / 1.629.
- **Capping the strengths does not fix it.** Clipping every hidden matrix's strengths to at most 3 after each step (exact SVD) makes both worse: Adam 1.646, Muon 1.645 (seed 0).

**Reading.** In this model Muon's early updates are small because the early gradient is close to rank 1, which works like a warmup that ends by itself when the gradient spreads. A run that takes a few full-size steps early gets stuck in a lower-rank state. An Adam warmup does not reproduce this (warmup 50/100/200 gives 1.642/1.641/1.648), likely because Adam's later directions are also worse. Untested: whether the same holds in a larger model or a longer run.

## 8. Muon's start is a little too small: 3× lr for the first 10 steps

If 30× Muon's lr for 10 steps hurts and Adam-sized steps hurt, is Muon's own start the best size? Muon with its lr multiplied for the first N steps (GELU, embedding lr 0.01, Muon lr 0.03, 3 seeds):

| first steps | validation loss |
|---|---|
| 10 at 0.1× | 1.631 |
| plain Muon | 1.627 |
| 25 at 2× | 1.627 |
| **10 at 3×** | **1.622** |
| 10 at 5× | 1.626 |
| 10 at 10× | 1.628 |
| 25 at 3× | 1.628 |
| 50 at 3× | 1.633 |
| 10 at 30× | 1.661 |

10 steps at 3× is better than plain Muon on 6 of 6 seeds (seeds 3–5 added: 1.6272 / 1.6213 / 1.6282 vs 1.6274 / 1.6316 / 1.6301). 6-seed means: 1.6237 vs 1.6285, a gain of 0.005. Small, but every seed agrees. Longer boosts (25 or 50 steps) lose it.

On the speedrun's ReLU record settings the same boost gives 4 of 5 seeds better, mean 0.0036 (see [speedrun README](../speedrun/README.md#early-muon-lr-boost)).

## 9. The rank-1 first gradient is the model learning character frequencies; put them in the bias and Adam's start stops hurting

At the start the model predicts nearly uniform characters (first loss 4.56). The cheapest improvement is to predict common characters more often, and that one shared push is the same for every example, which is why the first gradient is nearly rank 1. Starting the output bias at the log character frequencies (one line, `OUTBIAS=1` in [lmspike.py](lmspike.py)) gives that for free:

- First loss 4.56 → 3.27. Stable rank of the first hidden gradients 1.0–1.4 → 2.3–2.6. Strength of Adam's first step in its top direction 1.2 → 0.75 (seed 0, ReLU).

GELU, embedding lr 0.01, 3 seeds, data in [lmbias.jsonl](lmbias.jsonl):

| | default init | output bias = log frequencies |
|---|---|---|
| Adam (lr 0.003; 0.002 / 0.004 are worse with both inits) | 1.637 | **1.630** |
| Muon (lr 0.03; 0.02 / 0.04 are worse with both inits) | 1.627 | 1.627 |
| Adam minus Muon | 0.010 | **0.003** |
| 10 Adam steps, then Muon | 1.636 | 1.627 |
| Muon, 3× lr for the first 10 steps | 1.622 | **1.620** |
| last hidden matrix's stable rank after 10 Adam steps (seed 0) | 10 | 41 |

- **Adam gains 0.007, Muon gains nothing.** Seed by seed, Adam minus Muon is +0.006, +0.003, −0.001: no longer a consistent gap.
- **10 Adam steps at the start no longer cost anything** (1.627, same as Muon all the way).
- **The early boost still helps Muon** (better on all 3 seeds), so it is not only undoing the rank-1 start.
- **With ReLU it is not enough.** Adam still leaves 40% of layer 2 dead and ends at 1.716 (from 1.740), Muon 1.648.

The bias init is not allowed in the speedrun (only the optimizer may change there).
