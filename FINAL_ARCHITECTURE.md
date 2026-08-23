# ⚠ STALE — written 15:40, and the project moved after it. `report.md` is authoritative.

**Do not read this as the current state.** It is kept, not deleted, because this project's rule is
retraction-with-the-superseded-claim-visible. But it was written **before** three things it does not
know about, and its own preamble ("if something is not in this document, treat it as something we are
not doing") is exactly the sentence that makes it dangerous to leave unmarked:

1. **§3.5's DEEP HALF WAS WITHDRAWN at 17:23.** This document's table still says the useful band is
   *"near 40 loops"* at `U[32,48]` (line ~165). The full-budget artifact `tlab-deep-full` (30.0M
   tokens) returned plateau midpoint **22.6** against a pre-registered trigger of "near 22", and
   `mid/μ_rec = 0.57` sits in the **dense** range, not terminal-only. **The recommended schedule is
   `U[4,32]`, not `U[32,48]`.**
2. **Operator diversity became this project's first replicated CE positive** (§4.21): loop-cycled
   LoRA at rank ≥ 4, four in-job pairs across three platforms and two seeds, mean −0.0857, 95%
   t-interval excluding zero — **and the useful band does not move in any of the five pairs.**
3. **The learned depth gate was measured unable to express its own hypothesis** (§4.22): its logits
   are `w·h_t` on the raw state, so the softmax saturates to a hard argmax (effective loops mixed
   **1.0 of r**). The per-token headroom is therefore **untested** by that instrument, not refuted.

**Read `report.md` §0 (abstract), §3.5 (the method) and §6.0 (all 34 errors) instead.**
This file is retained as the 15:40 snapshot only.

---

# The final architecture — state as of 2026-08-23 ~15:40

**One-off report.** Written because the task asks for it directly and the answer was scattered across
`report.md`. Everything the project is actually doing should be in here; if something is not in this
document, treat it as something we are not doing.

---

## 0. What the task asks

From `task_at_full.md` (the brief, verbatim on the point):

> «Сделайте претрен лупд трансформера на предсказание следующего токена на Fineweb. Модель должна
> быть небольшой (**до 10M параметров**), как и бюджет токенов в обучении (**до 100M токенов**); за
> основу можно взять архитектуру **Qwen3**… **Количество лупов — чем больше, тем лучше.** Можно также
> реализовать ранний выход из лупов… Ваша задача получить **как можно более низкую перплексию на
> валидации за счет большого количества лупов**.»

and on the deliverable:

> «…отчет должен содержать подробное описание всех экспериментов и **финальной архитектуры с анализом
> того, почему именно такой вид дает лучшие результаты**. Кроме того, желательно обосновать, **почему
> ваш метод будет работать хорошо и на большем скейле**.»

Three things are being asked, and they are not the same thing:
1. **an architecture** — described, with reasons;
2. **low validation perplexity obtained *by* many loops** — not low perplexity by any means;
3. **an argument that the mechanism survives scaling**, the brief's own counter-example being a large
   learned value-embedding table whose benefit dies as parameters grow.

The brief also says explicitly: **«отсутствие положительного результата при хорошем анализе всех
негативных — хороший результат.»** A well-analysed negative counts. Much of §6 below is that.

---

## 1. Terminology used in this project

Defined here because these terms run through `report.md` and are ours, not standard.

| term | meaning |
|---|---|
| **loop / recurrence `t`** | one application of the shared block. The model applies **one** block `r` times |
| **`r`, loop count** | how many times the block is applied on a forward pass. Sampled per training step; swept at eval |
| **μ_rec** | *mean recurrence* — mean of the training-time loop-count distribution. `U[4,32]` → μ_rec = 18; `U[32,48]` → μ_rec = 40. Shorthand for "how deep this model was trained to run" |
| **loop curve** | validation CE as a function of eval-time loop count `r`. The central object; nearly every claim is a statement about its shape |
| **loop gain** | `CE@1 − CE@best`. What the looping buys over a single application |
| **plateau `[lo,hi]`** | contiguous band of swept depths within `tol` (default 0.01 nats) of the curve minimum. **Replaces argmin**, which was unusable here: 63 of 82 curves had argmins decided by margins of 1e-4 to 3e-3 against a noise floor two orders larger. `src/plateau.py` |
| **plateau midpoint / onset** | geometric midpoint of that band; and the *shallowest* depth already within `tol` of best — the deployment-relevant "how few loops can I get away with" |
| **grid-conditional** | a plateau is a set of *evaluated* depths, so it depends on which `r` were swept. Same checkpoint, dense 1..64 vs sparse {1,2,4,8,12,16,24,32}: midpoint 8.4 vs 9.8, a 17% swing. **A plateau quoted without its grid is not a number** |
| **supervision density `k`** | how many loops get a loss term per step (final loop always included, plus `k−1` sampled). `k=5` = "dense"; `k=1` = "terminal-only" |
| **supervision annealing** | dense for most of training, then terminal-only for the last fraction. **`sw90`** = switch at 90% of steps. This is the method |
| **`rev50`** | control: the same k=1 exposure placed *first* (k=1 → dense). Does nothing. Order matters |
| **in-job / paired** | both arms trained in one invocation: same seed, same data order, same loop draws, so the difference is the intervention. **Cross-job comparisons here have been wrong twice**; only in-job ones decide anything |
| **Δgain decomposition** | `Δgain = ΔCE@1 − ΔCE_best`. Separates "the ceiling improved" from "loop 1 got worse". Raising loop gain by *damaging loop 1* is not the same result. `src/gain_decomp.py` |
| **replicate floor** | measured spread between same-config runs. **CUDA dense 0.0150, CUDA terminal 0.0541, MPS 0.031–0.068.** An effect smaller than the applicable floor is not a result |
| **the anchor / decodability anchor** | interpretive framing (§4.18): dense per-loop supervision forces every supervised intermediate state to stay decodable by the tied readout, constraining how far the trajectory may travel. **Post-hoc, proposed by an external reviewer, labelled as such, and not the graded idea** |

---

## 2. The architecture, ground up

A **weight-tied looped transformer**: one block of three Qwen3-style decoder layers, applied `r`
times. Shipped checkpoint `full_control90_kaggle` — **9,064,608 parameters** (cap 10M),
**89,999,360 training tokens** (cap 100M), step 43,944.

```
h ← h0 + e                                     # e = embed(x); h0 a learned 448-vector
repeat r times:
    h ← block(h + e)                           # SAME weights every iteration; additive re-injection
logits ← tied_head( RMSNorm(h) )               # read out at EVERY loop, not just the last
```

**The block** (`src/model.py`, transcribed from the installed `transformers` Qwen3 reference and
pinned against it at `max|diff| = 2.38e-07` by `test_model.py` check [2]): pre-norm RMSNorm with fp32
upcast, per-head QK-norm on `head_dim` before RoPE, GQA, SwiGLU, no biases.

| | value | |
|---|---|---|
| hidden size | 448 | |
| layers per loop | 3 | one block = 3 distinct layers applied as a unit |
| heads / KV heads / head_dim | 4 / 2 / 112 | GQA |
| MLP intermediate | 1344 | ratio 3.0 |
| vocab | 4096 | own BPE; **tied** embedding ↔ LM head |
| prelude / coda | **0 / 0** | no unshared layers |
| `state_renorm` | **False** | no normalisation of the carried state between loops |
| `inject_mode` | additive | `h + e` at every `t > 0`; the `t = 0` injection is unconditional |
| `depth_init` | on | output projections scaled by `1/√(2·24)` at init |
| readout | RMSNorm → tied head | scale-invariant: **the readout sees direction only** |

**Where the parameters go — this is the design tension.** One decoder layer at H=448 is 2,409,568
params. The reusable block is **79.7%** of the budget; the tied embedding 1,835,008 (**20.2%**);
`h0` + `final_norm` 896. A prelude or coda is near-free at 730M params and is *not* free here — it
must be paid for out of the very block that loops multiply.

### Why each choice, with the measurement that decided it

| choice | rejected alternative | evidence |
|---|---|---|
| **no inter-loop norm** | RMSNorm between loops (Huginn's sphere confinement) | **−0.744 nats**, the largest single effect in the project. The normalised variant contracts (ρ ≈ 0.82 at every depth) and goes inert. External mechanism, verified from source: normalising the residual stream suppresses *massive activations*, and zeroing the MLP output that produces them **causally** removes stages of inference in a looped model (arXiv 2604.11791) |
| **no prelude/coda** | the sandwich every reference implementation uses | at a fixed 10M budget a prelude buys 0.355 nats **and makes the model depth-inert over the entire swept range [1,96]**. It wins the metric by removing the reason to iterate |
| **tied 4096 vocab** | a larger vocab | keeps ~80% of the budget in the reusable block. Cost: token perplexity is not comparable across submissions, so **bits/byte is the comparable figure** (3.3358 bytes/token, measured over the full val shard) |
| **additive re-injection** | concat + adapter; no re-injection | concat costs `2H²`; `inject_none` is the *worst* arm on that axis at training time — while being numerically inert at inference (`‖e‖/‖h‖` 3.2e-3 → 1.3e-4). Both are true and §4.3 reconciles them |
| **read out every loop** | final-loop-only readout | makes "how much does each loop help" directly measurable, and is what makes the supervision schedule a lever at all |

---

## 3. The method — what is actually being proposed

**Supervision annealing.** Train with a bounded per-loop loss (`k=5`) for the first 90% of steps, then
supervise **only the final loop** for the last 10%. `supervise_switch_frac = 0.90`.

**Zero parameters. Zero extra FLOPs** — it supervises *fewer* loops, not more. It is a property of the
loss, not of the model.

### Why a loss schedule and not a dynamics intervention

Four attempts to change how the state *traverses*, all nulls on the ceiling:

| intervention | result |
|---|---|
| inference-time radial clamp to `{‖h₁‖,‖h₈‖,‖h₁₆‖}` | optimum relocates 5 → 15 → 24; **best CE invariant to 0.006 nats** |
| convex gate `h ← (1−g)h + g·block(h)`, learned and fixed-`g` swept | null |
| `ε = λ/(N√L)` residual scaling | relocates *nothing* once argmin is replaced by the plateau statistic |
| training-time norm penalty | improves CE, but **88% of its loop-gain advantage is loop-1 damage** |

**Four convergent nulls are what license the positive claim:** the ceiling belongs to the path, and
the loss decides where along it you stop. The only lever that changed the *shape* rather than the
position was **where the loss is applied**.

### What the schedule does, measured

Reading the annealing arms as *how long the trajectory trains unanchored at the end* (2.5M tokens,
seed 0, one eval grid):

| terminal-only for | plateau | **mid** | onset | **CE_best** |
|---|---|---|---|---|
| 0% (dense) | [8,16] | 11.3 | 8 | 5.3418 |
| **10% (`sw90`)** | [8,24] | 13.9 | 8 | **5.2659** |
| 25% (`sw75`) | [12,24] | 17.0 | 12 | 5.3061 |
| 50% (`sw50`) | [12,24] | 17.0 | 12 | 5.3711 |
| 50% **placed first** (`rev50`) | [8,16] | **11.3** | 8 | 5.5957 |

Band depth is **monotone non-decreasing** in unanchored duration; CE is **U-shaped with an interior
optimum at 10%**; the order control returns *exactly* to dense's band. That last row is what makes
this a mechanism rather than a correlation.

---

## 4. The numbers

Protocol-matched local eval of both 90M checkpoints (`src/eval.py`, dense every-integer 1..64 grid):

| run | tokens | CE | **val ppl** | bits/byte | plateau *(dense 1..64)* | loop gain |
|---|---|---|---|---|---|---|
| **90M control** — the config §2 describes | 90.0M | **3.6599** | **38.86** | **1.5829** | [6,17] | 0.3047 |
| 90M + norm penalty — best perplexity | 90.0M | **3.6250** | **37.52** | **1.5678** | [6,14] | 0.5611 |
| previous headline | 46.0M | 4.0071 | 54.99 | 1.7330 | [5,14] | 0.2509 |

**The honest tension, stated rather than resolved.** The brief asks for low perplexity **by exploiting
many loops**. Those are two targets, and this architecture serves them at two settings:

| target | schedule | supervision | result |
|---|---|---|---|
| **lowest perplexity** | `U[4,32]` | dense | **ppl 37.52–38.86**, useful band [6,17] |
| **most useful loops** | `U[32,48]` | annealed | useful band near **40 loops**, still within 0.01 nats of best at **64 loops** — 1.33× beyond anything it trained on |

Same architecture; only the loop schedule and the loss schedule differ.

### The scale argument, and its one weak joint

**Strong half.** Annealing adds **no parameters**. The brief's counter-example fails because a fixed
lookup table's share shrinks as the model widens; there is no table here, no fixed capacity, nothing
diluted by width, and the cost is negative in FLOPs.

**Weak half, and the report says so.** The rule is stated as a *fraction* while the mechanism is keyed
to *tokens*. Loop gain emerges with training and saturates at **~10–15M tokens** — an absolute
quantity. So `sw90` switches at 2.25M in a 2.5M run (**before gain has emerged at all**) and at 81M in
a 90M run (**long after saturation**). Those are not the same intervention. The token-keyed form —
*dense until loop gain flattens* — means terminal-only for the last **~83%** of a 90M run, a different
recipe. **§3.5 defends the mechanism and explicitly not the parameterisation.** The discriminating run
is in flight (§5).

---

## 5. What is running right now

| stream | what it tests | state |
|---|---|---|
| **DS `tlab-anchor-tokenkey`** (`bt1hp97su48dc6096sqn`) | (a) **§4.18's own falsifier**: `sw90` at k=5/3/2 + in-job dense control. The anchor account says band depth depends on unanchored *duration*, **not** on dense-phase density — all three should give the same band. If they order with k, §4.18 is withdrawn. (b) **token-keyed vs fraction-keyed** annealing at 10M — decides §4's weak joint | EXECUTING |
| **Kaggle `tlab-seed-extension`** | the (sw90 − dense) paired difference at **seeds 2 and 3**. Seeds 0,1 give −0.0811, −0.0609; n=4 gives an interval on the number §3 rests on. **Pre-registered: if the four straddle 0, or their mean falls inside the 0.0541 floor, the annealing recommendation is withdrawn to "not resolved at this budget"** | pushed |
| **DS `tlab-deep-full`** | the deep artifact at μ_rec=40. Returns curves only (its config predates the `outputs:` fix) | EXECUTING, ~step 8100/19531. **It is `sw75`, which §3 has since narrowed away from** — pre-registered read in `RUNS.md` |
| **local: gated injection** | the **third cell on the normalisation axis**. §2 tested hard RMSNorm vs *nothing*; the field's actual choice is neither — a soft learned per-channel decay `α⊙z + δ⊙W_in·v` (Parcae, Done Right). ~896 params for the gates | running |

---

## 6. Work that is not architecture — and most of the project is this

The brief says a well-analysed negative is a good result. These are the negatives, and they are not
side-quests: three of them are *why* §3's method is a loss schedule rather than a dynamics one.

**Per-token depth demand is real and unreachable.** Oracle headroom 0.2008–0.3084 nats — larger than
the entire loop gain. Depth demand is reliable (split-half **0.866**) and large (46% of tokens want
more than 8 loops, 28% more than 32). **Five signal families fail** (entropy, top-1 margin, `dnorm`,
KL, cumulative-`dnorm`), in threshold and bucket form; position explains 0.06%, loop-1 entropy 0.71%.
**And it survives the literature's own explanation.** The published account is that dense supervision
pins every loop to the output manifold so confidence signals saturate — so an *annealed* model should
be readable. Matched pair, measured: band moves **1.70× deeper**, CE improves 0.061, and **oracle
headroom is unchanged (0.2008 → 0.2032) and still 0.0–0.1% capturable**. That closes the negative
rather than caveating it.

**The map does not converge, measured directly.** `u_t = h_t/‖h_t‖` is the entire input to both the
block and the readout. Fitting `‖u_t − u_T‖`: log-drift `C·ln(T/t)` wins at **R² 0.986 with one
parameter** against a convergent power law's **0.748 with two**, and 0.18 rad of angular motion still
accumulates over loops 129→384. There is **no fixed point in `u`**. Separately ρ, the Jacobian's
spectral radius, is **1.6227 at loop 2** — far outside the estimator's ~9% bias.

**We are in Blayney's degenerate branch.** Pre-norm without effective injection collapses all layers
onto one direction; measured min cross-layer cosine **1.0000 by loop 32**. A mechanism for the inert
prelude and the flat `inject_none` curve.

**Hyperparameters buy loss, not depth.** LR/weight-decay screen, six arms: every Δgain within ±0.02,
and **onset = 8 for all five well-trained arms**. Reassurance that the depth results are not an
artifact of an inherited config.

**Two ideas tested and killed today.** The *scale clock* (feed `log‖h‖` back into the block, +448
params, zero-init) was proposed on the premise that `u` converges to a fixed point. **The premise was
checked first and is false**; built and run anyway on a weaker motivation, and it is a decisive
negative — CE_best **6.7845** and **7.0170** against an in-job control's **5.4202**, i.e. **+1.36 and
+1.60 nats**, ~20× the floor, with `‖w‖` = 1.34 and 1.02, so the model *took* the parameter and got
worse. And the *arc/chord angular budget*, retracted twice, whose surviving content is an observation
with its interpretation withdrawn.

**Instrument failures are logged as results.** §6.0 is a 31-row failure table. Three from today alone:
a function named `sigma_max` had been computing **ρ** for the entire project; a column headed
`‖Δh‖/‖h‖` was actually the clean-vs-perturbed separation, off by **94×** from what the header
implied; and ρ's published `1.7019` came from an **unseeded** power iteration whose three fresh runs
(1.6578 / 1.6741 / 1.6979) bracket a value the report did not sit inside.

---

## 7. What we are NOT doing, and why

| not doing | reason |
|---|---|
| **MoE / sparse capacity** | the strongest live objection is not the brief's table example but that **our capacity is dense**, and the FLOPs-per-parameter argument is made against a dense baseline. Unaffordable under a 10M cap. Recorded as a limitation, not closed |
| **Q-exit / jointly-trained early exit** | the exit *rules* are measured and fail; a trained exiter is a different project, and the negative is the finding |
| **Muon, layer duplication, MLA** | Muon needs an LR sweep we cannot afford; layer duplication is post-hoc surgery on frozen weights and bears on no claim we make |
| **more than one model size or sequence length** | everything is 9.06M at seq 256. This bounds every claim here and is stated |

## 8. Known gaps, stated plainly

1. **The annealed arm's own noise floor was never measured.** We have floors for a dense arm (0.0150)
   and a terminal arm (0.0541); annealing is dense for 90% of training and terminal for 10%. Its
   −0.0710 is **4.7× the dense floor and 1.3× the terminal floor**. The Kaggle run addresses this by
   getting more paired estimates rather than a floor.
2. **Token-keyed vs fraction-keyed is open**, and is the weak joint in the scale argument.
3. **~20 DataSphere checkpoints were discarded** by a config defect (`outputs:` did not list the
   weights), so several analyses can never be extended to those arms.
4. **Gradient checkpointing is unused**, which is why the μ_rec=56 and 44 arms OOM'd — a documented,
   free route to exactly the deep schedules the brief is about.
5. **§1 of `report.md` — the idea narrative — is empty**, and is the user's to write: the task grades
   it separately and asks that it not come from an LLM.
