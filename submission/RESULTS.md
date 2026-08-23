# Results

*Headline figures, every intervention with its effect, and what is still landing. Numbers are
`src/eval.py` post-hoc sweeps on a shared grid; `src/headline.py check` verifies this file's headline
against the artifact it came from. Cells still running are marked and dated rather than omitted.*

---

## 1. Headline

| run | tokens | CE | **val ppl** | bits/byte | useful band *(dense 1..64)* | CE@1 | loop gain |
|---|---|---|---|---|---|---|---|
| previous headline | 46.0M | 4.0071 | 54.99 | 1.7330 | [5,14] | 4.2580 | 0.2509 |
| **90M control — the released artifact** | 90.0M | **3.6599** | **38.86** | **1.5829** | **[6,17]** | 3.9622 | 0.3023 |
| 90M + norm penalty | 90.0M | 3.6250 | **37.52** | 1.5678 | [6,14] | 4.1803 | 0.5553 |

**9,064,608 parameters (≤10M ✓) · 90.0M training tokens (≤100M ✓).**

**Perplexity is tokenizer-dependent** (4096-vocab); **bits/byte is the only externally comparable
figure**, and even that is against FineWeb, not FineWeb-Edu (§2.1).

**Finishing the token budget bought 0.39–0.42 nats. Every architectural intervention bought
0.002–0.22.** That is the report's most practically useful sentence.

**Why the control ships and not the 37.52 arm:** see `METHOD.md` §4 — 88% loop-1 damage, a narrower
band, the only arm that converges, and an unresolvable clipping confound.

## 2. Every intervention, and the pattern across them

ΔCE_best against each arm's **own in-job control**. Negative = better.

| intervention | ΔCE_best | band | params |
|---|---|---|---|
| inter-loop RMSNorm | **+0.744** | saturates at 4 | 0 |
| scale clock | **+1.36** *(non-finite by loop 39)* | onset 8→4 | +448 |
| gated (diagonal state-space) injection, α = 0.874 | **+0.247** | unmoved | +896 |
| loop-cycled LoRA, **rank 2** | **+0.094** | unmoved | +204k |
| radial clamp | ~0 *(ceiling invariant to 0.006)* | relocates | 0 |
| convex gate / fixed-`g` sweep | null | unmoved | +66k |
| ε = λ/(N√L) residual scaling | null *(its "shift" was a 0.0001-nat argmin flip)* | unmoved | 0 |
| **duo-causal attention, W = 2** | **+0.0093 / −0.0115** *(sign reverses, 2 seeds)* | **unmoved, to the digit** | **0** |
| norm penalty λ=0.01 (90M) | −0.030 *(wins ppl)* | **narrows** [6,17]→[6,14] | 0 |
| **loop-cycled LoRA, rank ≥ 4** | **−0.094** ⚠ | **unmoved (5 of 5)** | +409k (+4.51%) |
| **exclusive self attention (XSA)** | **−0.216** ⚠ | **unmoved** [8,16] | **0** |
| **supervision annealing** | CE **withdrawn at n=4** | **band widens 4/4 seeds** | **0** |

> **⚠ The two rows that lower the loss both carry caveats, and both improve the block rather than the
> looping.** **LoRA:** the `rank ≥ 4` restriction is **post hoc** — over all six arms the interval
> **covers zero** — there is no dose–response above the threshold, **~90% of the gain sits at `r = 1`**
> where the cycling is *logically inert*, and a branch pinned to one index (identical params, zero
> diversity) recovers **82%** of it in-job. **XSA:** **one seed**, 2.5M tokens, second seed running;
> **84% of its effect is at `r = 1`**. Full detail in `NEGATIVE_RESULTS.md`.

**The pattern, which is the report's central finding.** **Twelve interventions. Two lower the loss.
Not one widens the useful band.** And both loss-lowering ones deliver 84–90% of their gain at a
*single* loop — they improve the **block**, not the **looping**. The one lever that moves the band
(supervision annealing, 4/4 seeds, zero parameters) **does not lower the loss.**

*That dissociation is the answer to the brief's sentence: low perplexity and «за счет большого
количества лупов» come apart under measurement.*

## 3. Depth: what a looped model here actually does

- **Saturation without convergence.** The unit state drifts **logarithmically** (log-drift R² 0.986 with
  one parameter vs a convergent power law's 0.748 with two); 0.18 rad still accumulates between loops
  129 and 384; ρ = 1.6227 at loop 2. **Yet CE stops improving at loop ~8.** The task's DEQ premise —
  fast convergence makes further compute pointless — **does not explain the ceiling here**, because
  convergence never happens. *Scope: an untrained model drifts faster, so non-convergence is
  architectural.*
- **Late motion is fully visible to the readout and still useless.** Readout gain flat 207 → 226 while
  ‖Δu‖ falls **117×**; `E`'s condition number is 132, so there is no null subspace to hide in.
- **Graceful past the trained range.** Best at loop 8, still better than loop 1 at **loop 105**,
  crossing at 106. The deepest genuinely-useful band measured is **[16,32]** on a 30M-token model
  trained at `U[32,48]`.

## 4. Per-token depth demand: real, large, and unreachable — with a measured reason

Oracle headroom **0.2008–0.2032 nats**, split-half reliability **0.866** against a null of **0.0007**;
**27.9%** of tokens want depth > 32. **Five instrument classes fail to capture more than 0.1%.**

**Why**, and this is the part that is a mechanism rather than a list of nulls: **a token's 32 depth
keys span an effective rank of ~1.6**, mean pairwise cosine 0.91–0.97. There is almost nothing for any
mixing or selection mechanism to discriminate *between*. The collapse is present **at initialisation**
and **training makes it worse** (2.73 → 1.83), and the one arm that applies a genuinely different
operator per depth raises it by **0.01–0.08 out of 32**.

**Holding scale fixed and varying only weight tying**, an untied stack's *keys* are near-orthogonal
(31.83/33) while the tied loop's are 2.73/33 — **but most of that gap is per-layer projection
randomness**: at the *representation* level the untied stack is 4.36/33 against the tied 1.40/33, a
3.1× gap, and both streams are highly collinear. **The mechanism is that distinct per-layer
projections decorrelate a collinear state stream for free, and a tied loop has one `W_K` and cannot.**
(§4.7e — corrected 20:01 after the confound was raised.)

## 5. Still landing tonight

Marked rather than omitted; this file is updated as each lands.

| arm | what it decides | ETA |
|---|---|---|
| `dg_norm` (scale-invariant depth gate) | **the highest-stakes cell.** Registered joint test: mixing engaging with no gain confirms the rank collapse as the binding constraint; a real gain means §4.7e is **wrong** | ~20:30 |
| duo-causal **W = 3** | dose-response against W = 1, 2 | ~20:30 |
| `tlab-divx-s1` | capacity-vs-diversity, all three arms **in one job**, seed 1 | ~20:45 |
| `tlab-xsa-s1` | second seed for the −0.216 result | ~20:45 |
| `tlab-recmethod-s2` | the recommended configuration's own weights | ~20:50 |
| Kaggle `tlab-lora-scaleup` | does the LoRA positive survive ~5× budget (12M/arm) | ~21:45 |

## 6. Verification

```
src/test_model.py                  13 checks — ALL PASS
src/test_plateau.py                8 checks incl. a deliberate falsification probe — ALL PASS
src/headline.py check              every headline number matches its artifact — consistent
src/check_tokenizer_identity.py    on the SHIPPED checkpoint — PASS, |diff| 0.0020 vs chance 8.3178
src/make_inventory.py --check      113 trained arms accounted for
```
