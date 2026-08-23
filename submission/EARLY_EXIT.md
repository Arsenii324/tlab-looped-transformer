# Early loop-exit: implemented, measured, and it does not pay — with the reason underneath

*The brief lists early loop-exit as an optional component: «опционально реализовать ранний выход из
лупа». It was implemented and measured rather than skipped, and the outcome is a **negative with a
mechanism**, which the brief rates explicitly as a good result. This document is the whole case: that
the demand is real, that eight rules across five instrument classes cannot reach it, why, and what the
one test built to overturn that explanation returned.*

**The short version.** Per-token depth demand in this model is large (**0.3084 nats** of oracle
headroom on the calibration/test split; 0.3083 on the full set), reliable (split-half **+0.866** against a null of **+0.0007**), and **unreachable**: the
best of eight exit rules captures **0.1%** of it. The reason is measured, not assumed — **the depths a
rule would have to tell apart span an effective rank of ~1.6 out of 32.** There is almost nothing to
discriminate between.

---

## 1. The demand is real, and it is not an artifact of the measurement

This has to come first, because "early exit doesn't help" is a much weaker statement if there was
nothing to capture. `min_k E[CE]` is a statement about the *average* depth; the per-token argmin is a
distribution, and Jensen guarantees `E[min_k CE] < min_k E[CE]` strictly unless every token wants the
same depth. So saturation at loop 8 does **not** imply loops stop paying at 8 — it implies one global
depth cannot extract what is present.

*Instrument:* `src/exit_dump.py` / `src/exit_rules.py`, 2,048 frozen sequences, **524,288 scored
tokens**, calibration/test split **by sequence** so tokens sharing a context cannot leak across it.

| quantity | value |
|---|---|
| best fixed depth (chosen on calibration) | k = 8, test CE **3.9277** |
| oracle `E[min_k CE]` (label-using upper bound) | **3.6193** |
| **headroom** | **0.3084 nats** |
| per-token argmin depth, deciles | **[1, 2, 7, 43, 64]** |
| fraction wanting depth 1 / > 8 / > 32 | 0.216 / **0.464** / **0.279** |

**And the headroom is concentrated, not spread thin:**

| argmin depth | share of tokens | mean gain (CE@1 − CE@argmin) |
|---|---|---|
| 1 | 21.6% | 0.0000 |
| 2–8 | 32.0% | 0.3827 |
| 9–32 | 18.5% | 0.8746 |
| **33–64** | **27.9%** | **1.0059** |

**Three checks that the demand is not measurement noise**, because a per-token argmin is exactly the
statistic this project has been burned by before (§4.15 retired `argmin` for depth claims):

| check | result |
|---|---|
| **split-half reliability** — correlate each token's argmin over odd vs even loop indices | **+0.866**; median \|difference\| **1.0 loop**; 95.3% agree within 2 |
| the same statistic with halves paired across **different** tokens | **+0.0007** (the null) |
| **coarse grid** — 7 candidate depths instead of 64 | headroom 0.3062 vs 0.3086, **99.2% retained** — not a fine-grid artifact |
| **null A/B** — circular-shift or permute each token's residual | null headroom **0.3877 / 0.4110**, i.e. *larger* than real |

That last row is the important one and it cuts **against** the finding being interesting in the naive
way: random depth preference manufactures *more* apparent headroom than the data has. The real curves
are **4.6× smoother** than the shifted null (roughness 0.0077 vs 0.0351), which is what makes the
argmin locations meaningful — but it also means "there is headroom" is the wrong thing to be impressed
by. **What survives the nulls is that the argmin is *reproducible per token*, and that is what an exit
rule would need.**

## 2. Eight rules, five instrument classes, and the best captures 0.1%

Every rule is calibrated on the calibration split and reported on the test split. **The five classes
are: (1) confidence thresholds, (2) convergence thresholds, (3) bucketing by an early-loop value,
(4) a learned multinomial probe, (5) Q-exit heads** — eight rules in total, listed individually below
so the count is auditable against the table rather than asserted.

| rule family | test CE | vs best fixed | mean depth | oracle headroom captured |
|---|---|---|---|---|
| best fixed depth (k = 8) | 3.9277 | — | 8 | — |
| threshold on predictive **entropy** | 4.0561 | +0.128 | 50.6 | negative |
| threshold on logit **margin** | 4.0516 | +0.124 | 53.9 | negative |
| threshold on **‖Δh‖/‖h‖** | 4.0169 | +0.089 | 2.03 | negative |
| threshold on successive-output **KL** | 4.0168 | +0.089 | 2.21 | negative |
| **bucket** by early-loop value (best: KL) | 3.9275 | **−0.0002** | 8.70 | **0.1%** |
| **learned probe** (multinomial, all 4 signals, loops 1–4) | 4.0150 | +0.087 | 37.4 | **−28.3%** |
| Q-exit, **linear** head (PonderNet form) | 3.9443 | +0.0166 | 4.27 | negative |
| **Q-exit, PALBERT's `Λ([h_t, h_{t−1}])` + MLP head** | **3.9281** | **+0.0004** | 6.73 | **≈ 0%** |
| oracle (label-using upper bound) | 3.6193 | **−0.308** | — | 100% |

**Read the mean-depth column, because it is where the diagnosis is.** The *confidence* signals
(entropy, margin) exit at mean depth **50.6 and 53.9**. The *convergence* signals (‖Δh‖/‖h‖, KL) exit
at **2.03 and 2.21**. **They disagree by 25×, and both lose to a constant.** Two families of signal,
pointing in opposite directions, neither tracking the thing that matters.

**The two rules that do not lose are the two that barely act.** Bucketing by early-loop value lands at
mean depth 8.70 against the constant's 8.00, and PALBERT's Q-exit at 6.73 — both are approximately
"exit at 8 for everything", and both return approximately the constant's loss. *A rule that recovers
the constant by imitating it has not found per-token structure.*

## 3. Why — two structural reasons, and the second explains the whole family at once

**First: the trajectory-reading rules are reading a quantity with almost no cross-token variance.**

| quantity | mean | sd | **cv** |
|---|---|---|---|
| total angular distance travelled, per token | 2.8635 | 0.1952 | **0.068** |
| angular distance at each token's **own oracle depth** | 1.3680 | 1.0921 | **0.798** |

Every rule that conditions on *how far the state has travelled* is predicting a target whose
coefficient of variation is **12× larger** than the predictor's. The trajectories are nearly
identical across tokens; the optima are not.

**Second, and this is a mechanism rather than a list of nulls: a token's 32 depth keys span an
effective rank of ~1.6.** Mean pairwise cosine **0.91–0.97**; 84–86% of pairs above 0.95. The collapse
is present **at initialisation** (rank 2.45–2.73) and **training makes it worse** (→ 1.52–1.83). The
one intervention that provably applies a *different operator at every depth* — loop-cycled LoRA —
raises that rank by **0.01–0.08 out of 32**.

**So the family does not fail because eight rules were badly chosen. It fails because the
representation a weight-tied loop builds carries almost no per-depth information for any of them to
read.** `SCALE.md` §5 argues this is a property of weight tying — one `W_K` applied to a collinear
state stream — and therefore does not go away at larger width.

## 4. The test that could have overturned the explanation, and did not

§3's explanation was, until 2026-08-23, an explanation with no experiment that could kill it. The
project's own learned depth gate had failed for a *different* reason — its logits were `w·h_t` on the
**unnormalised** state, whose norm grows 1.8–4.0× within a forward pass, so the softmax saturated to a
hard argmax (effective loops mixed **1.01–1.05 of r**, 95–98% of tokens above 0.99 top-weight). **A
rank explanation cannot be tested by an instrument that never mixes.**

So a **scale-invariant** gate was built — `F.normalize` on the state before the gate head, plus a
learned temperature, **+450 parameters** — and a **joint falsifier registered before the arm existed**:

| gate mixes? | CE gain? | what it decides |
|---|---|---|
| yes | **no** | the rank collapse is the binding constraint — §3's explanation stands |
| yes | **yes** | **the explanation is WRONG**; the per-token headroom is reachable |
| no | either | instrument failure again; **nothing is decided** |

**It mixes.** Effective loops mixed **7.58 / 8, 14.96 / 16, 29.84 / 32**, with **zero** tokens above
0.99 top-weight. *This is a genuinely working per-token soft mixture over depths, and it is the first
one this project built.*

**And it gains nothing.** Two seeds, in-job paired: **−0.0012 and +0.0023.** The sign reverses and
both are an order of magnitude inside the 0.0150 replicate floor.

**A working mixer over a representation spanning ~1.6 of 32 dimensions returns nothing — which is
what §3 predicts, and it is now the reading that survived the test designed to overturn it.**

> Its plateau reads [12,24] and [12,32], *deeper* than its control's [8,20], and that number is
> **excluded from every depth table in this submission** by a decision recorded before the arm ran:
> the gate mixes over loops `1..r`, so its plateau measures **mixture-window size, not depth.**

## 5. The inversion worth stating: early exit here is *safe* and *useless*, for the same reason

The standard objection to per-token early exit in a looped model is the **ragged KV cache** — a token
still computing at loop 32 must attend to keys and values written by neighbours that stopped at loop
2. LLA (2607.15456) reports that reusing final-loop cache "collapses GSM8K generation to zero".

**Measured here, it costs almost nothing.** With only loop `t` re-run against a depth-`k` cache, the
spread across *all seven* cache depths is **0.0052 nats at t = 8**, 0.0011 at t = 16, 0.0033 at
t = 64. The extreme off-diagonal cell (k = 64, t = 8) reads 3.9082 against a clean 3.9078.

**And oracle-depth caching — choosing the *best* cache depth per token — buys −0.0096**, about 5.5×
*below* the replicate floor, reversing sign by query depth 24.

**These are the same fact as §3, seen as a benefit instead of a limitation.** The depths are close to
interchangeable. That makes a ragged cache safe — and it is exactly why there is nothing for an exit
rule to select on. **A model whose loops mattered more would have a more dangerous cache.** The one
catastrophic region in the grid is the `t = 1` **column** (serving any deep cache to a depth-1 query
costs ~3.5 nats), which is a property of the *query* depth before the state has been through the block
at all — not of ragged exit depths among ordinary tokens.

*This is the opposite of the direction the literature's concern points, and it is stated here because
a reader who expects the ragged-cache failure will otherwise assume it was simply not tested.*

## 6. What this means for a design that wants early exit to work

Stated as engineering consequences rather than as further nulls, since the negative is only useful if
it constrains the next attempt:

1. **Do not spend the parameter budget on the exit head.** Eight rules including two learned heads and
   a PALBERT-form Q-exit all land within 0.02 nats of a constant. The signal is not in the readout, in
   the state-update magnitude, or in successive-output agreement. **A better head reads a
   representation that does not distinguish the depths.**
2. **Fix the representation first, and the lever is weight tying, not width.** The depth keys collapse
   at initialisation, before any gradient. `SCALE.md` §5 measures the asymmetry: distinct per-layer
   projections manufacture a near-orthogonal key set out of a *collinear* state stream for free, and a
   tied loop has one `W_K` and cannot buy that at any width. **Partial untying — even one unshared
   `W_K` per loop-index bucket — is the cheapest intervention this evidence points at**, and it is
   untested here (§8 below).
3. **Any exit gate must be scale-invariant.** The failure in §4 is not exotic: the readout is
   deliberately scale-invariant, ‖h‖ grows ~10³ over training, and a gate on the raw state saturates.
   This cost the project one whole instrument and a section that had to be rewritten as an instrument
   failure.
4. **Exiting early is cheap to serve.** §5 means a ragged cache is not the obstacle. If a future
   design *does* find the signal, the serving side is already measured as near-free here.

## 7. Scope, and what is not claimed

- **Not claimed:** that early exit cannot work in looped transformers. It is measured not to work
  **here** — 9.06M parameters, 3-layer tied block, 46M-token checkpoint, teacher-forced.
- **Teacher-forced throughout.** The rules are evaluated on next-token CE against a frozen context,
  not in generation. A generating exiter would pay a *compounding* version of §5's substitution, which
  this measurement bounds from below rather than settles.
- **The oracle is a label-using upper bound**, not an achievable target. Its role is to establish that
  headroom exists, and the nulls in §1 are what make that non-vacuous.
- **`dg_norm` is two seeds at 2.5M tokens.** A null at screening scale is not a null at 90M, and
  §4.24 of the report states the budget scope condition on every result of this shape.

---

*Full measurement, protocol and every intermediate table: `../report.md` §4.7, §4.7a–e, §4.8, §4.8a–b,
§4.22, §4.23, §4.24.*
