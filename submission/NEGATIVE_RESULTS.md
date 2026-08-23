# Methods tested to destruction

*The brief states that «отсутствие положительного результата при хорошем анализе всех негативных —
хороший результат». This is the analysis. Every row names a **specific published method or a
specific predicted mechanism**, the regime it was tested in, and **why it failed** — a null with a
mechanism, not "we tried it and nothing moved".*

**Two framing facts, because they decide how much these are worth.** This report's positive claim is
**~0.07–0.09 nats**. Its negatives are **0.25, 0.45 and 1.36 nats** — one to two orders of magnitude
larger, measured against **in-job controls** (arm and control in the same job, same shard, same
tokenizer, same seed), with mechanisms attached. **The strongest results in this project are the ones
that failed.**

---

## 1. The dynamics family — nine interventions, one lowers the loss, none widens the band

| intervention | source | ΔCE | effect on the useful band |
|---|---|---|---|
| inter-loop RMSNorm (`state_renorm=True`) | Huginn's own recipe | **+0.744** | saturates at loop 4; loop gain never emerges |
| scale clock (feed `log‖h‖` back into conditioning) | reviewer proposal | **+1.36** | state non-finite by loop 39 |
| gated (diagonal state-space) injection, α = 0.874 | Parcae; *Done Right* | **+0.247** | unmoved |
| loop-cycled LoRA, rank 2 | MoDr lineage | **+0.094** | unmoved |
| **loop-cycled LoRA, rank ≥ 4** | MoDr lineage | **−0.086** | **unmoved (5 of 5 pairs)** |
| radial clamp (inference-time) | Sharma & Vu | ~0 | *relocates* the optimum; ceiling invariant to 0.006 |
| convex gate / damped sub-stepping | arXiv 2605.23872 | null | unmoved |
| ε = λ/(N√L) residual scaling | arXiv 2606.18524 | null | unmoved |
| norm penalty (training-time) | Sharma & Vu | −0.030, **wins ppl** | **narrows** [6,17] → [6,14] |
| duo-causal attention, W = 2 | Think-at-Hard | **+0.009 / −0.011** *(sign reverses)* | **identical to the digit, both seeds** |

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

**Exploration during loops — one of the three levers the brief names by name — hurts monotonically.**
σ = 0.05 / 0.15 / 0.40 → ΔCE **−0.006 / +0.183 / +0.790**, and the optimum never moves off loop 8.
This was pre-registered as a two-way discrimination: if the trajectory's coherence is what *wastes*
depth, some σ > 0 should help; if the coherence is *load-bearing*, noise degrades monotonically.
**The second happened.** The near-perfect increment alignment (`cos → 0.9999`) is the mechanism doing
the work, not a pathology to break up.

## 3. The per-token depth family — five instrument classes, and then the reason underneath

Per-token depth demand is **real**: oracle headroom **0.2008–0.2032 nats**, split-half reliability
**0.866** against a null of **0.0007**, and 27.9% of tokens want depth > 32, gaining ~1 nat each.

| # | instrument | result |
|---|---|---|
| 1 | five label-free halting rules (entropy, margin, ‖Δh‖/‖h‖, KL, cumulative ‖Δu‖) | all beaten by a constant depth |
| 2 | static readout mixture over depths, raw and normalised | best **−0.0023** against a 0.0527 floor |
| 3 | the same test on an **annealed** checkpoint | null — kills the literature's own trajectory explanation |
| 4 | oracle-depth ragged KV cache | **−0.0096**, reverses sign by query depth 24 |
| 5 | learned per-token depth gate | **did not test its own hypothesis** — see below |

**Two structural reasons, and the second is the deeper one.**

*First:* the rules condition on total path length, whose cross-token **cv is 0.068**, while oracle
depth's **cv is 0.798**. Every trajectory-reading rule is reading a quantity with almost no
cross-token variance in order to predict one with an order of magnitude more.

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
