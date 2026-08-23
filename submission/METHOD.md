# The final architecture and method

*Answers the brief's «описание… финальной архитектуры с анализом того, почему именно такой вид дает
лучшие результаты». Every line cites the section of `../report.md` that measured it. Where a claim was
withdrawn, the withdrawal is stated here, not only there.*

---

## 1. The architecture, and it has not changed in weeks

One **Qwen3-style decoder block of 3 layers**, weight-tied, applied `r` times.

| | |
|---|---|
| hidden size | 448 |
| attention heads / kv heads | 4 / 2 (GQA) |
| head dim | 112 |
| MLP intermediate (SwiGLU) | 1344 |
| layers per loop | 3 |
| vocabulary (tied embedding) | 4096, byte-level BPE trained on FineWeb |
| **total parameters** | **9,064,608** ≤ 10M ✓ |
| — in the reused block | 7,228,704 (79.7%) |
| training tokens | **90.0M** ≤ 100M ✓ |

Verified against the real `transformers` Qwen3 reference at **max\|diff\| = 2.4e-07** before any
training compute was spent (`src/test_model.py`, 13 checks).

**Four choices are load-bearing, each with the measurement that decided it:**

| choice | rejected alternative | why, measured |
|---|---|---|
| **no inter-loop normalisation** (`state_renorm=False`) | RMSNorm between loops | **≈ −0.68 nats token-corrected**, the largest single effect in the project. The normalised variant *contracts to a fixed point by loop ~16* and never accrues loop gain at all (§4.1, §4.3, §4.12) |
| **no prelude / no coda** | the sandwich every reference implementation uses | at a fixed 10M budget a prelude buys 0.355 nats **and makes the model depth-inert over [1,96]** — it wins the metric by removing the reason to iterate (§4.5) |
| **additive re-injection** of the embedded input at every loop | none / concat-adapter | `inject_none` is the worst arm and uniquely shows no benefit from depth (§4.1). *Precisely: it removes **re**-injection; `h = h₀ + e` is unconditional, so every arm sees the input once* |
| **`1/√(2·n_loop_eff)` output scaling at init** | off | a real, moderate margin (§4.1). **Known defect:** `n_loop_eff` is fixed at 24 while schedules ran at mean 18 and 40 — in-job pairs cancel it exactly, cross-schedule comparisons carry it (§6.0b) |

**The block is deliberately the simplest member of its family, and that is a result rather than an
omission.** Ten interventions from the literature were tested on it — inter-loop norm, gated
(diagonal state-space) injection at the field's own α, a scale clock, radial clamping, convex gating,
ε = λ/(N√L) residual scaling, a norm penalty, loop-cycled LoRA, exclusive self attention, duo-causal
attention. **Two lower the loss; none widens the useful band.** Elaborating the block was measured and
did not pay in the way the brief asks for.

## 2. The training recipe, which is where the report's positive results live

- **Loop schedule:** every step samples `r ~ U[4,32]` (μ_rec = 18).
- **Supervision:** loss on a sparse subset of loops (`k = 5`) for most of training, then **terminal
  loop only for the final ~10% of steps** — *supervision annealing*, `supervise_switch_frac = 0.90`.
- **Zero added parameters.** It is a schedule on which loop indices receive gradient.

**Why the loss schedule rather than the dynamics.** Three independent interventions on how the state
*traverses* — inference-time radial clamping (§4.6), a learned convex gate plus a fixed-`g` sweep
(§4.10), and ε-residual scaling (§5.0) — **all relocate the optimum without raising the ceiling**, and
the third relocates nothing once argmin is replaced by a statistic that can bear weight. Convergent
nulls across three mechanisms are what license the positive claim: *the ceiling belongs to the path,
and the loss decides where along it you stop.*

> ### ⚠ What annealing does and does not do — the CE half is WITHDRAWN
>
> **Withdrawn at n = 4** by a criterion registered before the data existed. Seeds 0–3 give ΔCE_best
> −0.0811 / −0.0609 / **+0.0482** / −0.0902; mean −0.0460 sits inside the 0.0541 floor; the paired
> t-interval **[−0.1478, +0.0558] covers zero.**
>
> **What survives, at 4 of 4 seeds:** the useful band widens — **+2.5 / +2.5 / +2.5 / +7.2** grid
> points — **including at seed 2, the seed that reverses the CE claim.** Decomposed into band edges
> (§4.15), at μ_rec = 18 the **onset does not move at all** (8 → 8) while the **end** goes 16 → 24: the
> model does not improve further, it **degrades later**.
>
> **So the recommendation is: anneal if you want the useful band deeper; do not expect it to lower the
> loss.**

**A second recipe-level lead, and it is n = 1.** Keying the switch to an absolute token count rather
than a step fraction wins by **−0.2208** at 10M — ~4× the floor and the largest supervision effect
measured here. The fraction rule was validated at 2.5M where it switches *before* loop gain emerges,
and extrapolated to budgets where it switches deep post-saturation (§3.5). **One seed. It is a lead,
not a recommendation**, and replication at ≥3 seeds was not run.

## 3. Why this should keep working at a larger scale

Full argument in `SCALE.md`. In one paragraph: **the recommended mechanism adds no parameters**, so
the brief's own disqualifying example — a fixed trainable table whose benefit stops mattering as
parameters grow — does not apply; there is no table to outgrow. The **honest weak joint** is that the
rule is stated as a *fraction of steps* while the mechanism is keyed to an *absolute token count*
(§3.5), and those coincide at exactly one budget.

## 4. The released artifact, and how it differs from the recommendation

**Shipped:** the 90M **control** — `U[4,32]`, `supervise_k=5`, **no annealing**.

**Stated rather than left to be discovered:** the released checkpoint shares the architecture and
demonstrates both budgets, but the annealing schedule was established *after* the full-budget runs
launched, so **no full-budget checkpoint of the annealed recipe exists**. The annealing result rests
on in-job paired comparisons at 2.5M–30M. A run producing the recommended configuration's own weights
was launched at 18:33 (`tlab-recmethod-s2`, 2 arms × 10M).

**Why the control and not the norm-penalty arm** (which wins perplexity 37.52 vs 38.86): 88% of the
penalty arm's loop-gain advantage is **loop-1 damage**, its band narrows [6,17] → [6,14], it is the
only arm whose map **converges** — the regime §2 argues against — and it carries a clipping confound
the artifacts cannot resolve (§4.6b). The control is the arm with no confound on either axis.

## 5. Reproduce

```bash
python src/test_model.py                # 13 correctness checks, must pass first
python src/data.py                      # streams + packs FineWeb shards
python src/train.py                     # the center config
python src/eval.py checkpoints/<ckpt> --max-loops 64
# verify a released checkpoint against the tokenizer that produced it:
python src/check_tokenizer_identity.py checkpoints/full_control90_kaggle --expect-ce1 3.9622
```
