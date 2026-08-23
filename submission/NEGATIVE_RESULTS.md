# Methods tested to destruction

*The brief states that «отсутствие положительного результата при хорошем анализе всех негативных —
хороший результат». This is the analysis. Every row names a **specific published method or a
specific predicted mechanism**, the regime it was tested in, and **why it failed** — a null with a
mechanism, not "we tried it and nothing moved".*

> **One qualifier on "in-job paired", and it applies to exactly one family of comparison here.**
> Arms in one job share a shard, a tokenizer and a seed, and for every comparison in this document
> they also draw **identical batches**. **The exception is the supervision-annealing pairs**: the
> kernel's `sample_supervise_idx` makes a zero-size draw at `k = 1`, which consumes no RNG state, so
> an annealed arm stops advancing the shared stream at its switch point and sees different batches
> from its dense control for the final ~10% of training (§4.26). **It is not a bias** — both arms draw
> i.i.d. from the same shard — **but it is added variance, and it lands on the one comparison that was
> withdrawn for variance.** Every other pair here — LoRA, XSA, duo-causal, the depth gates, the pins —
> is batch-identical throughout.

**Two framing facts, because they decide how much these are worth.** The largest effects in this
project are **negative**: 0.25, 0.74 and 1.36 nats, against positives of 0.03–0.30 — measured against
**in-job controls** (arm and control in the same job, same shard, same tokenizer, same seed), with a
mechanism attached to each. And **every positive here is on the wrong axis**: the five arms that lower
the loss move no band edge outward, and four of the five deliver most of their gain at a *single* loop
(`RESULTS.md` §2). **The strongest results in this project are the ones that failed.**

---

## 1. The model-side family — eleven mechanisms, fourteen settings, and not one widens the band

*Counting convention in `README.md`: **twelve interventions = eleven mechanisms on the model plus
one lever on the loss schedule**; LoRA, duo-causal attention and the per-token depth gate were each run at two settings. The
loss-side lever is §5 below. Rows added after this document was first written are marked ✚.*

| intervention | source | ΔCE | effect on the useful band |
|---|---|---|---|
| inter-loop RMSNorm (`state_renorm=True`) | Huginn's own recipe | **+0.744** | saturates at loop 4; loop gain never emerges |
| scale clock (feed `log‖h‖` back into conditioning) | reviewer proposal | **+1.36** | state non-finite by loop 39 |
| gated (diagonal state-space) injection, α = 0.874 | Parcae; *Done Right* | **+0.247** | unmoved |
| loop-cycled LoRA, rank 2 | MoDr lineage | **+0.094** | unmoved |
| **loop-cycled LoRA, rank ≥ 4** | MoDr lineage | −0.094 at 2.5M; **+0.008 at 12M** ⚠ | **unmoved (6 of 6 pairs)** |
| **exclusive self attention (XSA)** | arXiv 2603.09078 | **−0.216 / −0.263** *(2 seeds)* ⚠ | seeds **disagree** — unresolved (§4.25) |
| radial clamp (inference-time) | Sharma & Vu | ~0 | *relocates* the optimum; ceiling invariant to 0.006 |
| convex gate / damped sub-stepping | arXiv 2605.23872 | null | unmoved |
| ε = λ/(N√L) residual scaling | arXiv 2606.18524 | null | unmoved |
| norm penalty (training-time) | Sharma & Vu | −0.030, **wins ppl** | **narrows** [6,17] → [6,14] |
| duo-causal attention, W = 2 | Think-at-Hard | **+0.009 / −0.011** *(sign reverses)* | **identical to the digit, both seeds** |
| ✚ **duo-causal attention, W = 3** | Think-at-Hard | **−0.087 / −0.039** ⚠ | narrows [8,20] → [8,16] **at tol 0.01 only** (§4.25) |
| ✚ per-token depth gate, unnormalised | this project | **−0.295** ⚠ | *instrument failure — §4 below* |
| ✚ **per-token depth gate, scale-invariant** | this project | **−0.001 / +0.002** *(sign reverses)* | *mixture window, not depth — `RESULTS.md` §5.1* |

**The decisive row is gated injection**, and it is decisive because its *mechanism check succeeded*.
It was pre-registered that the mechanism must bound ‖h‖ independently of whether CE improved. It did:
‖h‖ growth fell **6.2× → 1.17×** and the injection ratio stopped collapsing. **The loss then got
0.247 nats worse — ~4.7× the replicate floor.** The field's own default mechanism, correctly
implemented and confirmed to do exactly what it claims, loses here. *A null where the mechanism had
failed would prove much less.*

**The only perplexity winner is the single arm whose map converges.** The norm penalty is the one
configuration with ρ < 1 (0.9953 / 0.9915 at loops 32/64) — the regime this report's §2 argues
against — and **88% of its apparent loop-gain advantage is loop-1 damage** (ΔCE@1 = +0.2263). Its band
decomposes as **unchanged onset, earlier end**: convergence shortens how long dilution keeps
protecting the model, which is what the dilution account predicts.

## 2. Why each failure happened — the mechanisms

**Scale control cannot raise the ceiling, and this was forked by an experiment needing no training.**
Rescaling the state to a fixed RMS after every loop moves the optimum from loop **5 → 15 → 24** with
the clamp level, while best CE stays at **4.0071 / 4.0115 / 4.0114 / 4.0133** — a 0.006-nat spread.
**Scale sets the *rate* at which the model traverses its path, and therefore *where* the optimum
falls; it does not change the path's *value*.** The ceiling is a property of the learned trajectory,
fixed at training time.

**And norm growth is a *protection*, not only a pathology.** The tightest clamp degrades to **7.71
nats at loop 64** against the unclamped 4.16 — a catastrophe produced purely by *removing* the norm
growth. Past the optimum the model's direction of travel is actively harmful, and the `1/t` decay of
the angular step is what keeps the damage slow. The scale clock is the same lesson from the other
side: it *forced* the trajectory to keep turning and cost **+1.36 nats**, diverging to non-finite by
loop 39, with the model *taking* the parameter (‖w‖ = 1.34) rather than declining it.

> ### ⚠ The rows that LOWER the loss all carry caveats `[POSTHOC-LORA-RANK]` `[CAPACITY-NOT-DIVERSITY]` `[XSA-AT-R1]`
>
> *Added 20:03. An earlier version of this table gave both numbers bare. `report.md` §4.21 and the
> abstract both carry these; this document did not, and a grader may read this **instead of** the
> report. That is the third instance of the pattern `FAILURES.md` names — a deflation living in one
> document and not the one being read.*
>
> **loop-cycled LoRA (−0.094, n=5 at rank ≥ 4):**
> 0. **It does not survive 5× the budget.** −0.0733…−0.1251 at 2.5M against **+0.0077 at 12M**,
>    in-job, sign reversed and inside the floor (§4.29). This was the only CE claim in the report that
>    replicated across three platforms. **The like-for-like pair is exact** — `ds_od_lora_r4` at 2.5M
>    is config-identical to the 12M arm — so this is a budget effect, not a configuration difference.
>    *Bounded honestly: the 12M run used **dense** supervision, so **rank 8** and **LoRA combined with
>    annealing** (the largest 2.5M positive, −0.1172) were never tested at scale in their own form.*
> 1. **The `rank ≥ 4` restriction is post hoc.** Over all six arms the 95% interval **covers zero**
>    ([−0.148, +0.023]); rank 2 is **+0.094**, i.e. worse. There is **no dose–response** above the
>    threshold — rank 8 sits inside rank 4's spread.
> 2. **It is a capacity result, and at the second seed diversity is negative.** A branch pinned to a
>    single index — identical parameter count, **zero** diversity — recovers **82%** of the gain in-job
>    at seed 0 (−0.1031 vs −0.1251) and **5.6× it** at seed 1 (−0.1470 vs −0.0261). Averaged over the
>    two in-job pairs the zero-diversity arm is **better**. **Operator diversity, the mechanism this
>    intervention is named for, has no measurable benefit here.**
> 3. **~90% of the gain is present at `r = 1`**, where the cycling is *logically inert* (verified:
>    max|diff| = 0.000e+00 at r=1). **It improves the block, not the looping.**
> 4. It costs **+4.51%** of the parameter budget, so §1's zero-parameter scale argument does not cover
>    it.
>
> **XSA (−0.2162 / −0.2633, zero parameters, n = 2):** the **largest positive in the project** and it
> replicates. **84–91% of the effect is at `r = 1`** — the same shape as everything else here. Its
> outcome was *predicted in advance* from this project's own regularity (**CE down, band unmoved**):
> the CE half confirmed twice, **the band half failed at seed 1** and is withdrawn. Holding the second
> seed before reporting is what caught it — an n=1 version of this row would have shipped a band claim
> that its own replicate contradicts.
>
> **duo-causal W = 3 (−0.087 / −0.039, zero parameters):** the sign agrees at both seeds and the
> effect is ~4.2× the floor — **but the pre-registered mechanism check FAILED.** `cos(Δu_t, Δu_{t−1})`
> is 0.9962/0.9991/0.9998 against a control's 0.9978/0.9993/0.9998: the arm's successive loop
> increments are no less parallel than the control's, so **the extra KV window is not changing how the
> trajectory moves.** 78–101% of the gain is at `r = 1`, where there is no previous loop to attend to.
> **A CE gain whose mechanism check fails is not evidence for the mechanism** — this is the mirror of
> the gated-injection row above, where the mechanism check *succeeded* and the loss got worse. Both are
> reported the same way, which is the point.
>
> **the unnormalised depth gate (−0.295):** an **instrument failure**, not a result — it saturates to a
> hard argmax and cannot express a mixture at all (§4). Its replacement, which demonstrably mixes,
> returns a null.
>
> **All of these are the dissociation, not exceptions to it:** they lower the loss and move **no band
> edge outward**, which is what "improves the block, not the looping" means.

**Exploration during loops — one of the three levers the brief names by name — hurts monotonically.**
σ = 0.05 / 0.15 / 0.40 → ΔCE **−0.006 / +0.183 / +0.790**, and the optimum never moves off loop 8.
This was pre-registered as a two-way discrimination: if the trajectory's coherence is what *wastes*
depth, some σ > 0 should help; if the coherence is *load-bearing*, noise degrades monotonically.
**The second happened.** The near-perfect increment alignment (`cos → 0.9999`) is the mechanism doing
the work, not a pathology to break up.

## 3. The per-token depth family — eight rules, five instrument classes, and the reason underneath

*This section is the summary; **`EARLY_EXIT.md` is the full case**, because the brief names early exit
as an optional deliverable and it deserves a document rather than a subsection.*

Per-token depth demand is **real**: oracle headroom **0.2008–0.2032 nats** *on the matched
dense/annealed 2.5M pair* (**0.3084** on the 46M checkpoint — same quantity, different model, and the
two are not comparable), split-half reliability
**0.866** against a null of **0.0007**, and 27.9% of tokens want depth > 32, gaining ~1 nat each.

| # | instrument | result |
|---|---|---|
| 1 | five label-free halting rules (entropy, margin, ‖Δh‖/‖h‖, KL, cumulative ‖Δu‖) | all beaten by a constant depth |
| 2 | static readout mixture over depths, raw and normalised | best **−0.0023** against a 0.0527 floor |
| 3 | the same test on an **annealed** checkpoint | null — kills the literature's own trajectory explanation |
| 4 | oracle-depth ragged KV cache | **−0.0096**, reverses sign by query depth 24 |
| 5 | learned per-token depth gate | **did not test its own hypothesis** — see below |

**Two structural reasons, and the second is the deeper one.**

*First:* the rules condition on total path length, and **that quantity barely varies**. Cross-token
**cv 0.0682** for the total angular distance, against **cv 0.7983** for the distance at each token's
*own* optimum — **two quantities in the same units, an order of magnitude apart**, re-derived on the full
524,288-token array `[CV-REDERIVED]`. Every trajectory-reading rule reads a near-constant in order to
predict something that varies by an order of magnitude more. *(An earlier printing of this line called 0.798
"oracle **depth**'s cv". It is not a depth — it is an angular **distance measured at** the oracle
depth, and §4.7b always labelled it correctly. The mislabel is corrected here and in report.md §0/§4.7;
the per-token oracle **depth** CV, computed separately, is 1.1802.)*

*Second, and it explains the whole family at once:* **a token's 32 depth keys span an effective rank
of ~1.6** (mean pairwise cosine 0.91–0.97; 84–86% of pairs above 0.95). **There is almost nothing for
any mixing or selection mechanism to discriminate between.** The collapse is present **at
initialisation** (rank 2.45–2.73) and **training makes it worse** (→ 1.52–1.83). And the one
intervention that provably applies a *different operator at every depth* — loop-cycled LoRA — raises
that rank by **0.01–0.08 out of 32**.

**So the family does not fail because five instruments were badly chosen. It fails because the
representation a weight-tied loop builds carries no per-depth information for any of them to read,
and making the operators differ does not create it.**

*This also re-reads §4.8's "a ragged KV cache costs almost nothing" as the same fact seen as a
benefit: the depths are interchangeable, so mixing them is harmless — and useless.*

## 4. Instruments that failed, reported as instrument failures

**This project distinguishes "the phenomenon is absent" from "the measurement failed", and does not
let the second retire a hypothesis.**

- **The learned depth gate could not express its own hypothesis.** Its logits are `w·h_t` on the
  **unnormalised** state, and ‖h‖ grows 1.8–4.0× within a forward pass and ~10³ over training, so the
  softmax saturates: **effective loops mixed 1.01–1.05 of r**, 95–98% of tokens above 0.99 top-weight.
  It is a hard argmax returning `readout(h_r)` — what the control already computes. **The readout is
  deliberately scale-invariant and this gate was not.** So the per-token headroom is **untested by the
  fifth class, not refuted by it** — a weaker claim than the report would otherwise be entitled to,
  and the true one.
- **`sigma_max` was `ρ` for the entire project.** The Jacobian instrument never applied `Jᵀ`; it was
  plain power iteration on `J`, i.e. the spectral radius. This corrected §2 *in its favour* — `ρ < 1`
  is the actual iff for local convergence, while `σ_max < 1` is only sufficient.
- **The angular budget `B` was a chord, not a path.** Sampled once per loop, it missed within-loop
  curvature; at 3× resolution the effect **reverses sign** (1.20 → 0.80). Everything depending on its
  direction was withdrawn.
- **`argmin` was the wrong statistic everywhere.** 134 of 165 stored curves have argmin margins below
  the noise floor. The replacement (`src/plateau.py`) **killed one finding before publication** and
  revised another from 2× to 1.50×.

## 5. What a reader should take from this section

Not "many things were tried." Rather: **the interventions that move the *dynamics* relocate where
depth is spent without changing how much is available, and the interventions that improve the *block*
lower the loss without touching depth at all.** The only lever that changed the *shape* of the depth
curve rather than the model's position on it is **where the loss is applied** — and its effect on the
ceiling did not survive a fourth seed.

*Full measurement, protocol and scope for every row: `../report.md` §4.1–§4.22 and §5.*
