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
| **total parameters** | **9,064,608** ≤ 10M ✓ — `sum(p.numel() for p in model.parameters())`. **A `state_dict` sum gives 10,899,616**, double-counting the tied embedding (`4096 × 448 = 1,835,008`); see `README.md` |
| — in the reused block | 7,228,704 (79.7%) |
| training tokens | **90.0M** ≤ 100M ✓ |

Verified against the real `transformers` Qwen3 reference at **max\|diff\| = 2.4e-07** before any
training compute was spent (`src/test_model.py`, 13 checks).



> **One trap for anyone who clones and instantiates.** `LoopedTransformer(Config())` — the *bare
> default* — reports **9,065,056** parameters, not 9,064,608. The 448 is `loop_norm.weight`, because
> **`Config()`'s default is `state_renorm=True`**: the inter-loop RMSNorm this report measures at
> **+0.744 nats** and rejects. **The default config is the arm we argue against**, kept as the
> default only because it is the field's convention and every ablation is expressed as a delta from
> it. The released architecture is `state_renorm=False`, and the shipped checkpoint's own `model_cfg`
> carries it — load from the checkpoint rather than from `Config()` and the count is 9,064,608.
> *(The checkpoint's `state_dict` sums to 10,899,616 because the tied embedding appears under two
> keys: 9,064,608 + 4096×448 = 10,899,616. Nothing is untied.)*

**Four choices are load-bearing, each with the measurement that decided it:**

| choice | rejected alternative | why, measured |
|---|---|---|
| **no inter-loop normalisation** (`state_renorm=False`) | RMSNorm between loops | **≈ −0.68 nats token-corrected**, the largest single effect in the project. The normalised variant *contracts to a fixed point by loop ~16* and never accrues loop gain at all (§4.1, §4.3, §4.12) |
| **no prelude / no coda** | the sandwich every reference implementation uses | at a fixed 10M budget a prelude buys 0.355 nats **and makes the model depth-inert over [1,96]** — it wins the metric by removing the reason to iterate (§4.5) |
| **additive re-injection** of the embedded input at every loop | none / concat-adapter | `inject_none` is the worst arm and uniquely shows no benefit from depth (§4.1). *Precisely: it removes **re**-injection; `h = h₀ + e` is unconditional, so every arm sees the input once* |
| **`1/√(2·n_loop_eff)` output scaling at init** | off | a real, moderate margin (§4.1). **Known defect:** `n_loop_eff` is fixed at 24 while schedules ran at mean 18 and 40 — in-job pairs cancel it exactly, cross-schedule comparisons carry it (§6.0b) |

**The block is deliberately the simplest member of its family, and that is a result rather than an
omission.** **Eleven mechanisms** were tested on it — inter-loop norm, gated (diagonal state-space)
injection at the field's own α, a scale clock, radial clamping, convex gating, ε = λ/(N√L) residual
scaling, a norm penalty, loop-cycled LoRA, exclusive self attention, duo-causal attention, and a
per-token depth-mixture gate — plus one lever on the loss schedule. **Five lower the loss; none
widens the useful band; at `tol = 0.01` three of the five narrow it, a direction that does not
survive halving the tolerance (§4.25).** And **four of the five deliver 78–101% of that
gain at a single loop where their own mechanism is inert**, so they improve the block rather than the
looping (`RESULTS.md` §2). Elaborating the block was measured and did not pay in the way the brief
asks for.

## 2. The training recipe, and exactly what it is recommended *on*

- **Loop schedule:** every step samples `r ~ U[4,32]` (μ_rec = 18).
- **Supervision:** loss on a sparse subset of loops (`k = 5`) for most of training, then **terminal
  loop only for the final ~10% of steps** — *supervision annealing*, `supervise_switch_frac = 0.90`.
- **Zero added parameters.** It is a schedule on which loop indices receive gradient.

> **The recommendation in one qualified sentence, placed here rather than after the argument for it.**
> **Anneal if you want the useful depth band deeper. Do not expect it to lower the loss.** The band
> half holds at **5 of 5 seeds** with the same edge decomposition at 2.5M and at 10M tokens; **the
> ceiling half was withdrawn at n = 4 and a fifth point at 4× the budget made it worse** (+0.1119,
> §4.23e). **And the released weights are the *dense* control, not this recipe** — §4 below says why,
> and that choice is now evidenced at the recipe's own budget rather than inherited from launch order.
> *This is a recommendation about where depth stays useful. It is not a recommendation for lowest
> perplexity, and at 10M in-job it costs 0.11 nats.*

**Why the loss schedule rather than the dynamics.** Three independent interventions on how the state
*traverses* — inference-time radial clamping (§4.6), a learned convex gate plus a fixed-`g` sweep
(§4.10), and ε-residual scaling (§5.0) — **all relocate the optimum without raising the ceiling**, and
the third relocates nothing once argmin is replaced by a statistic that can bear weight.

**What those convergent nulls do and do not license, stated carefully because the next paragraph
withdraws half of this recipe's claim.** They support a *negative*: the ceiling belongs to the learned
path, and no intervention on how the state traverses it raised that ceiling. They are **not** evidence
that annealing lowers the loss — that claim was withdrawn at n = 4 and is not reinstated here. What
they license is the weaker and still useful statement that **the loss schedule is the only axis on
which anything moved the depth curve's *shape* rather than the model's position along it** — and what
moved was the band, not the ceiling.

> ### ⚠ What annealing does and does not do — the CE half is WITHDRAWN `[WITHDRAWN-ANNEAL-CE]`
>
> **Withdrawn at n = 4** by a criterion registered before the data existed, and **a fifth point at 4×
> the budget has since made it worse**: ΔCE_best is −0.0811 / −0.0609 / **+0.0482** / −0.0902 /
> **+0.1119**, the last at **10M tokens** in-job (§4.23e). Mean **−0.0144**, inside the 0.0541 floor,
> spread −0.09 to +0.11.
>
> **What survives, at 5 of 5 seeds:** the useful band widens — and **the decomposition is identical at
> 2.5M and at 10M**: **onset 8 → 8 (unchanged), end 16 → 24, midpoint 11.3 → 13.9** (§4.15, §4.23e).
> The model does not improve further, it **degrades later**. That the same two edges move the same way
> across a 4× budget range is the strongest form this claim has.
>
> **So the recommendation is: anneal if you want the useful band deeper; do not expect it to lower the
> loss.**
>
> **And one qualifier on the word "in-job paired" as it applies to these arms specifically.** The
> kernel's `sample_supervise_idx` consumes **zero** RNG draws at `k = 1`, so an annealed arm stops
> advancing the shared random stream at its switch point and sees **different batches** from its dense
> control for the final ~10% of training (§4.26). Not a bias — both draw i.i.d. from the same shard —
> **but it is added variance on precisely the comparison whose CE claim was withdrawn for variance.**
> It does not touch the band result, which reproduces the same two edges at five seeds and two
> budgets. *Every other paired comparison in this submission is batch-identical throughout.*

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

**And read that argument knowing what it is an argument *for*.** It defends a mechanism whose **CE
claim is withdrawn** `[WITHDRAWN-ANNEAL-CE]` — what scales here is *the band effect at zero parameter
cost*, not a perplexity gain. A scale argument attached to the withdrawn half would be worthless, and
this project has made that exact mistake once already (`SCALE.md` §1 defended annealing for two hours
without stating the withdrawal). **The architecture's own scale argument is separate and does not
depend on annealing at all:** the reused block is ~80% of the budget and scales the ordinary way,
and the tied vocabulary's share *shrinks* as the block widens.

## 4. The released artifact, and how it differs from the recommendation

**Shipped:** the 90M **control** — `U[4,32]`, `supervise_k=5`, **no annealing**.

**Stated rather than left to be discovered:** the annealing schedule was established *after* the
full-budget runs launched, so **no 90M checkpoint of the annealed recipe exists**. The recommended
configuration now has weights of its own at **10M** (`rec_sw90_s2_last.pt`, §4.23e), and that run is
also the recipe's first direct test at its own budget. **It confirms both halves of the
recommendation, including the unflattering one:** the band widens exactly as at 2.5M (onset 8 → 8,
end 16 → 24), and CE is **0.1119 nats worse** than its in-job dense control. **So the choice to ship
the dense control is now evidenced, not merely inherited from launch order** — and there is no case
for spending remaining quota on a 90M annealed run.

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
# the three consistency gates, all of which must pass before anything here is quoted:
python src/headline.py check            # headline numbers vs their artifacts
python src/check_caveats.py --strict    # every deflated claim carries its caveat
python src/check_crossref.py --strict   # every figure in submission/ appears in report.md
```
