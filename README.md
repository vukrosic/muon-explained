# How Muon turns a lopsided update into a fair one, step by step

[Muon](https://kellerjordan.github.io/posts/muon/) is an optimizer for the weight matrices of neural networks. Its core is one short formula, applied 5 times:

```
X  ←  3.4445·X  −  4.775·(XXᵀ)X  +  2.0315·(XXᵀ)²X
```

An optimizer is the part of training that decides how to change the weights after each batch of data. For years the default has been AdamW. Muon replaces it for the big weight matrices inside the network. The first and last layers and the small vectors still use AdamW.

![Timeline: Schulz 1933, NanoGPT record Oct 2024, blog post Dec 2024, Moonlight Feb 2025, Kimi K2 Jul 2025](images/muon_timeline.png)

Some facts:

- **The name is an acronym.** Muon stands for MomentUm Orthogonalized by Newton–Schulz.
- **The math is 91 years old.** Schulz published the core trick in 1933, as a way to invert a matrix using only multiplications. Muon uses a tuned version of it.
- **It started as a speed contest.** In October 2024 Muon set a record in the Modded NanoGPT speedrun, a public race to train a small GPT-2 model to a fixed quality as fast as possible. That training became 1.35 times faster.
- **It now trains giant models.** Moonshot AI trained Moonlight (16B parameters, 5.7 trillion tokens) with Muon in February 2025. Kimi K2, with 1 trillion parameters and 15.5 trillion tokens, was trained with MuonClip, a Muon variant, in July 2025.
- **It uses less memory than AdamW.** Muon remembers 1 number per weight, AdamW remembers 2.
- **It costs a little more compute per step than AdamW.** Each training step already does about 6 operations per weight per token for the forward and backward pass. With 524,288 tokens per step in the NanoGPT speedrun, that is about 3 million operations per weight. AdamW adds about 15 per weight, so its extra cost is close to 0%. Muon's 5 rounds add about 0.7%. For a model the size of Llama 405B the estimate is 0.5%. So Muon pays off whenever it needs at least 0.7% fewer steps than AdamW. Every measurement below says it needs more than that.

![Muon vs AdamW: CIFAR-10 record 3.3 to 2.6 s, 1.5B model 13.3 to 10 h, compute to the same loss 52% (Moonshot) or 91% (Wen et al.), half the optimizer memory, 0.7% extra cost vs 9–48% saving](images/muon_numbers.png)

The first two numbers come from Keller Jordan's post. All numbers are reported by their authors, not measured here.

How much compute Muon saves is disputed:

- **Moonshot AI says about 2×.** They fit scaling curves on models from 0.4B to 1.5B parameters. Muon needed about 52% of AdamW's compute to reach the same loss. AdamW's settings were found by grid search. Muon reused AdamW's settings.
- **Wen et al. at Stanford say much less.** They compared 10 optimizers on models from 0.1B to 1.2B parameters, and tuned every optimizer's settings separately. The best matrix-based optimizers, Muon among them, were about 1.3–1.4× faster than AdamW at 0.1B, and only about 1.1× faster at 1.2B. They found that many earlier speedups came from a weak AdamW baseline. In one example, tuning only AdamW's learning rate in a standard recipe made it about 2× faster on a 0.1B model.

Both studies agree that Muon beats AdamW. They disagree on how much, and the second one finds the gain shrinking as models grow. Which number holds for very large models is not settled. Kimi K2, the largest Muon-trained model, has no AdamW run of the same size to compare against.

This post explains what the formula does, using one small 2×2 matrix and pictures. Every number in the pictures from section 1 on comes from the scripts in this repo.

## 1. A matrix stretches space

During training, each weight matrix gets an update matrix G. In Muon, G is the gradient averaged over recent steps (momentum).

A matrix can be pictured by what it does to a circle. Multiply every point of a circle by G and you get an ellipse. The ellipse has a long arrow and a short arrow. Their lengths are the matrix's **strengths**. The textbook name for them is singular values.

![A circle multiplied by G becomes an ellipse with strengths 1.00 and 0.20](images/step1.png)

Here G stretches one direction by 1.00 and the other by 0.20. A plain update with this G moves the weights 5 times more along the blue arrow than along the red one.

The arrows are not the x and y axes. The long arrow is the longest radius of the ellipse, and the short arrow is the shortest. Every other radius is somewhere in between. The two arrows are always at 90° to each other. Our G stretches first and then turns everything by 30°, so its arrows point diagonally. Another matrix puts them at another angle.

![Three matrices: arrows on the axes, turned 30°, turned 17°. Always the longest and shortest radius.](images/step1b_directions.png)

A 2×2 matrix has 2 strengths. A 3×3 matrix turns a ball into an egg shape with 3 arrows. A 768×768 weight matrix has 768 strengths, one per arrow, all at 90° to each other.

In real networks this imbalance is far worse. Muon's author reports that updates for transformer weight matrices are dominated by a few directions, and the rest are almost zero. Muon's goal is to make every strength equal to 1. The arrows keep their directions. Only their lengths change. The ellipse becomes a circle, so the update moves every direction by a similar amount.

### Why this helps

A plain update spends almost its whole step on the few big directions. The small directions barely move, step after step. Muon's author guesses that some of these small directions carry rare signals that still matter for learning. Setting every strength to 1 gives them the same share of the step as the big ones. This is his explanation of why training gets faster, not a proof. The proof he offers is the speed records.

### Why it took until 2024

The idea of setting every strength to 1 is older than Muon:

- **2015–2016.** Carlson and colleagues proposed *stochastic spectral descent*, which did exactly this with an exact SVD. Their methods had no momentum. Muon's author found momentum necessary.
- **2018.** The Shampoo optimizer came out. In 2024 Bernstein and Newhouse showed that Shampoo, with its running averages removed, gives this same update.
- **2022.** Tuddenham and colleagues orthogonalized gradients with an SVD and added momentum afterwards. Their best setup lost to well-tuned SGD with momentum, and the paper was mostly overlooked.

Three things changed with Muon:

1. **Speed.** An exact SVD of every weight matrix at every step is far too slow on a GPU. Newton–Schulz uses only matrix multiplications, and it stays stable in bfloat16, the fast low-precision number format of modern GPUs.
2. **Order.** Muon averages the gradients first (momentum) and then sets the strengths to 1. Muon's author found that this works better than the reverse.
3. **A fair race.** Many optimizer papers beat an AdamW that was not tuned well. The NanoGPT speedrun is a public race against a heavily tuned baseline, with a fixed target and a timer. Muon won there, which made people believe it.

## 2. The plan, and step 0: divide G by its size

### What Muon wants

Every matrix can be written as three moves: a turn, a stretch, and another turn.

```
G     =  U · S · Vᵀ          Vᵀ turns, S stretches by the strengths, U turns
goal  =  U · Vᵀ              the same two turns, with the stretch removed
```

Our G is built exactly this way: a stretch by 1.00 and 0.20, then a 30° turn. Removing the stretch means setting every strength to 1. In the picture, the ellipse becomes a circle again, and the arrows keep their directions.

### Why not compute U, S and V directly

The textbook tool that finds U, S and V is called the singular value decomposition (SVD). Muon's author tried it: it is far too slow to run on every weight matrix at every training step. GPUs are fastest at one job, multiplying matrices. So the question becomes: can multiplications alone change the strengths, without touching the arrows?

They can. X·Xᵀ·X has the same arrows as X, with every strength cubed. Section 3 shows why. Adding a few such terms gives a curve applied to every strength at once:

```
f(s) = 3.4445·s − 4.775·s³ + 2.0315·s⁵
```

The plan is to pick a curve that pushes every strength toward 1, and to apply it 5 times.

### Why the strengths must start at 1 or less

A curve with s⁵ in it shoots up for big inputs. It pulls toward 1 only for inputs from 0 to about 1.2. Put 1.5 in and 4.48 comes out. Put 3 in and 375 comes out. Real gradients can have strengths of 0.001 or 40. So before the rounds, every strength must be moved into the safe range.

![Dividing G by its size, and why: the curve explodes for inputs above about 1.2](images/step0_size.png)

### Why divide by the size

The ideal divisor would be the biggest strength. The biggest strength would then become exactly 1. But finding the biggest strength needs the SVD again. So Muon divides by a number it can get from the entries alone, the size:

1. Square every entry of G and add them up. Here 0.750 + 0.010 + 0.250 + 0.030 = 1.040.
2. Take the square root. √1.040 = 1.020.
3. Divide every entry by 1.020.

Here is why this works. The sum of the squared entries always equals the sum of the squared strengths:

```
size² = (sum of every entry²) = s₁² + s₂² + s₃² + …
```

For our G: 1.00² + 0.20² = 1.040, the same total as the 4 entries. The reason is that turning never changes lengths, so only the stretch counts. The size² contains the biggest strength squared plus other terms that are never negative. So the size is never smaller than the biggest strength. After dividing, every strength is 1 or less. For our G, 1.00 becomes 0.981 and 0.20 becomes 0.196.

![Size² equals the sum of squared strengths; the biggest strength after dividing; the 768 case lifted in 3 rounds](images/step0_why.png)

### The price

When many strengths are big, the size is much bigger than the biggest strength. Take a 768×768 matrix with all 768 strengths equal to 1. Its size is √768 = 27.7, so every strength becomes 0.036. That is far below 1, and it is why the curve's start matters. Near 0 it multiplies by 3.4445, so 0.036 becomes 0.124, then 0.418, then 1.117. Three rounds pay the price back.

## 3. Step 1: one round is one curve

Here is the key fact. The matrix formula does the same thing as feeding each strength into this curve:

```
f(s) = 3.4445·s − 4.775·s³ + 2.0315·s⁵
```

Why? Write X with its arrows and strengths as X = U·S·Vᵀ, where S holds the strengths. Then XXᵀ = U·S²·Uᵀ, so (XXᵀ)X = U·S³·Vᵀ. Multiplying by XXᵀ cubes every strength and leaves the arrows alone. Doing it twice gives the fifth power. So the whole formula gives U·f(S)·Vᵀ: the same arrows, with each strength replaced by f(strength).

You never compute the arrows or the strengths. The formula only uses matrix multiplications, which GPUs are fast at. The strengths change anyway.

![One round: read each strength off the curve. 0.196 becomes 0.640, 0.981 becomes 0.717](images/step2.png)

For our matrix:

| strength | into the curve | out of the curve |
|---|---|---|
| weak | 0.196 | 0.640 |
| strong | 0.981 | 0.717 |

The weak strength more than triples. Near 0 the curve's slope is 3.4445, so small strengths get multiplied by about 3.4 each round. The strong one is pulled down. After a single round, weak ÷ strong goes from 0.20 to 0.89.

## 4. Repeat 5 times

Each round's output is the next round's input. On the plot, go up to the curve (one round), then across to the dotted line (feed the result back in), then up again.

![Five rounds as a zig-zag path on the curve, plus a table of strengths per round](images/step3.png)

| round | strong | weak |
|---|---|---|
| 0 | 0.981 | 0.196 |
| 1 | 0.717 | 0.640 |
| 2 | 1.171 | 1.094 |
| 3 | 0.839 | 0.700 |
| 4 | 1.115 | 0.915 |
| 5 | 0.797 | 0.722 |

The weak strength passes 1 after two rounds. After that, both strengths bounce between about 0.7 and 1.2. They never settle exactly on 1. That is on purpose. The coefficients were picked to make small strengths grow as fast as possible, and the price is this wobble. Muon's author reports that not reaching exactly 1 did not hurt training.

Here is the same matrix after each round:

![The ellipse after each round becomes almost a circle](images/step4.png)

The arrows never turn. The thin ellipse becomes a rounder shape after one round and stays round.

## 5. Quintic vs cubic

The formula above is called **quintic** because its curve goes up to s⁵. It is Muon's default. The older textbook version is **cubic** Newton–Schulz:

```
X  ←  1.5·X  −  0.5·(XXᵀ)X        curve: f(s) = 1.5·s − 0.5·s³
```

![Quintic vs cubic after 1 round and after 5 rounds](images/poly_plot.png)

| | quintic | cubic |
|---|---|---|
| slope near 0 | 3.4445 | 1.5 |
| what happens to s = 1 | goes to 0.70 | stays exactly 1 |
| after 5 rounds | anything from 0.005 to 1.2 lands between 0.68 and 1.15 | anything above 0.3 lands within 0.004 of 1, small ones stay small |
| matrix multiplications per round | 3 | 2 |

The cubic is exact but slow to lift small strengths. It multiplies them by only 1.5 per round, so after 5 rounds a strength of 0.01 reaches only 0.076. The quintic is rough but lifts everything.

On an 8×8 matrix with strengths 1, 0.6, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01 (run `python3 muon_orth.py`):

| method | multiplications | strengths after | error vs exact answer |
|---|---|---|---|
| quintic × 4 rounds | 12 | 0.68 to 1.17 | 0.211 |
| quintic × 5 rounds | 15 | 0.70 to 1.13 | 0.210 |
| cubic × 5 rounds | 10 | 0.06 to 1.00 | 0.565 |

The exact answer is the matrix with the same arrows and every strength equal to 1. The error is the distance from that answer, divided by its size.

## 6. The code

```python
import numpy as np

QUINTIC = (3.4445, -4.775, 2.0315)
CUBIC = (1.5, -0.5)

def muon_orth(G, steps=5, kind="quintic", eps=1e-7):
    X = G / (np.linalg.norm(G) + eps)   # step 0: every strength now ≤ 1
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T                          # keep XXᵀ the smaller square matrix
    for _ in range(steps):
        A = X @ X.T
        if kind == "quintic":
            a, b, c = QUINTIC
            X = a * X + (b * A + c * A @ A) @ X
        else:
            a, b = CUBIC
            X = a * X + b * A @ X
    return X.T if tall else X
```

Real Muon runs the same loop in bfloat16 on the GPU.

## Run it

```bash
pip install -r requirements.txt
python3 muon_orth.py      # the 8×8 table above
python3 teach_size.py     # images/step0_size.png
python3 teach_why_size.py  # images/step0_why.png
python3 teach.py          # images/step1.png … step4.png
python3 teach_directions.py  # images/step1b_directions.png
python3 plot_poly.py      # images/poly_plot.png
python3 intro_plots.py    # images/muon_timeline.png, muon_numbers.png
```

## Sources

- Muon, its coefficients 3.4445, −4.775, 2.0315, the speedrun, CIFAR-10, 1.5B-model and compute-overhead numbers: Keller Jordan, [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) (December 2024).
- Moonlight and the 2× compute result: Moonshot AI, [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) (February 2025).
- Earlier work: Carlson et al., stochastic spectral descent (2015–2016); Gupta, Koren, Singer, [Shampoo](https://arxiv.org/abs/1802.09568) (2018); Tuddenham, Prügel-Bennett, Hare, [Orthogonalising gradients to speed up neural network optimisation](https://arxiv.org/abs/2202.07052) (2022); Bernstein, Newhouse, [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325) (2024).
- The smaller speedup over well-tuned AdamW: Kaiyue Wen, David Hall, Tengyu Ma, Percy Liang, [Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.02046) (September 2025, ICLR 2026).
- Kimi K2 and MuonClip: Moonshot AI, Kimi K2 technical report (July 2025), as covered by [HPCwire](https://www.hpcwire.com/aiwire/2025/07/14/chinas-moonshot-ai-releases-trillion-parameter-model-kimi-k2/).
- Schulz, *Iterative Berechnung der reziproken Matrix*, Zeitschrift für Angewandte Mathematik und Mechanik 13 (1933), 57–59.
