# THE WHOLE STATE — 2026-08-23 ~17:35

**Read this one if you read nothing else.** Replies 00–15 were written incrementally and several of
their claims have since been *withdrawn by later measurements in this same project*. This document
supersedes them wherever they disagree. It is self-contained: no access to `report.md` assumed.

At the end there is a **self-check**. If you cannot answer those questions from this document, ask
rather than assuming — several of them are things that earlier replies of mine got wrong.

---

## 1. What the artifact is

A weight-tied looped transformer. One 3-layer Qwen3-style block applied `r` times, **9,064,608
parameters** (cap: 10M), trained on **90M FineWeb tokens** (cap: 100M). Vocab 4096, tied embedding,
H=448, GQA 4q/2kv, SwiGLU. No prelude, no coda, no inter-loop normalisation, additive re-injection of
the embedded input at every loop, learned `h0`.

**Headline numbers**, both from a dense every-integer 1..64 eval sweep on the same protocol:

| checkpoint | val CE | val ppl | bits/byte | useful-depth plateau |
|---|---|---|---|---|
| **90M control** (the config recommended) | 3.6599 | **38.86** | 1.5829 | [6, 17] |
| 90M + norm penalty | 3.6250 | **37.52** | 1.5678 | [6, 14] |

Perplexity is tokenizer-dependent (4096-token vocab); **bits/byte is the only comparable figure**.
Against Parameter Golf's ~1.058 bpb this model loses — but PG entries train on ~7B tokens, ~70× this
budget, so that comparison is 100M-token vs ~7B-token, not architecture vs architecture.

## 2. The spine — what the report actually claims

Ordered by how much survives scrutiny.

**(a) The trajectory never converges, and this is measured three ways.** The unit state `u = h/‖h‖`
drifts **logarithmically**: a log-drift model fits at R² 0.986 with *one* free parameter against a
convergent power law's 0.748 with two, and 0.18 rad of real angular motion still accumulates between
loops 129 and 384. The mechanism is two of our own measurements composed — per-loop step decays as
`1/t` while consecutive steps stay aligned at `cos → 0.9999`, so `Σ C/s ≈ C·ln(T/t)`, which diverges.
`ρ(∂F/∂h) > 1` at every measured depth corroborates it (1.6227 at loop 2, seeded and reproducible).

**Yet CE stops improving at loop ~8–10.** So: *saturation without convergence.* That directly
contradicts the premise the task itself advances (DEQ-style: fast convergence makes further compute
pointless). Here convergence never happens and the compute stops paying anyway.

**Important scope, added today:** an untrained model with the same config *also* drifts
logarithmically, and faster (C = 0.3084 vs 0.1549). So non-convergence is **architectural**, present
before any gradient. Training slows the drift; it does not create or remove it.

**(b) Late-loop motion is fully visible to the readout and still useless.** Readout gain
(‖Δlogits‖/‖Δu‖) is flat across depth (207 → 226) while ‖Δu‖ falls 117×. There is no null subspace
for the trajectory to hide in. The problem is not that the model computes something the head cannot
read — it is that the direction it keeps moving in stops being an improvement.

**(c) Seven interventions on the dynamics; zero raise the ceiling.**

| intervention | effect on CE | effect on band |
|---|---|---|
| hard inter-loop norm (`state_renorm=True`) | **+0.744** | saturates at 4 |
| scale clock (feed `log‖h‖` back into conditioning) | **+1.36** | diverges to non-finite by loop 39 |
| gated injection, α=0.874 (the field's own choice) | **+0.247** | unmoved |
| loop-cycled LoRA r=2 (operator diversity) | **+0.094** | unmoved |
| radial clamp | ~0 | relocates optimum, ceiling invariant to 0.006 |
| convex gate / damped sub-stepping | null | — |
| ε = λ/(N√L) residual scaling | null | its apparent effect was a 0.0001-nat argmin artifact |
| norm penalty (via the loss) | −0.030, **wins ppl** | **narrows** [6,17]→[6,14]; only arm with ρ<1 |

**The only perplexity winner is the single arm that converges** — the regime (a) argues against. And
the gated-injection row is the decisive one: its *primary pre-registered mechanism check succeeded*
(‖h‖ growth 6.2× → 1.17×, injection ratio stops collapsing), and the loss got worse anyway. That is
the field's default mechanism, working exactly as designed, and losing.

**(d) Per-token depth demand is real and unreachable by four independent instrument classes.** Oracle
headroom 0.2008–0.2032 nats, split-half reliability **0.866** against a null of 0.0007 — the signal
exists. It is not reachable by: five label-free rule families (§4.7); a static readout mixture over
depths, raw or normalized (best −0.0023 against a 0.0527 floor); the same test on an *annealed*
checkpoint, which kills the literature's own explanation; or an oracle-depth ragged KV cache
(−0.0096, reverses sign by query depth 24). The structural reason: rules condition on total path
length, whose cross-token **cv is 0.068**, while oracle depth's **cv is 0.798**.

## 3. What was withdrawn today — the part earlier replies get wrong

**(i) The annealing CE claim is WITHDRAWN.** Replies 12–13 and everything before them report that
supervision annealing (`sw90`) beats an in-job dense control at −0.0811 and −0.0609. That was **n=2**.
Extended to seeds 2 and 3:

| seed | 0 | 1 | **2** | 3 |
|---|---|---|---|---|
| ΔCE_best | −0.0811 | −0.0609 | **+0.0482** | −0.0902 |

Mean −0.0460, sd 0.0640. Both pre-registered withdrawal triggers fired (values straddle zero; mean
inside the 0.0541 floor). A reviewer objected that 0.0541 is an *unpaired* floor and proposed a
paired t-interval instead — correct in principle, so I ran it: **[−0.1478, +0.0558]**, covers zero,
**same verdict**. Withdrawn to "not resolved at this budget."

**(ii) But the band widens at ALL FOUR seeds — and this is the better result.**

| seed | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Δ band midpoint | +2.5 | +2.5 | **+2.5** | +7.2 |

Never negative. **Seed 2 — the seed that reverses the CE claim — shows the same +2.5 band shift.** So
the surviving claim is: *annealing relocates the useful band robustly (4/4) and does not move the
ceiling.* That is the CE-vs-loop-utility disjointness this report documents four times elsewhere, now
demonstrated **on its own recommended intervention**, at n=4, with the ceiling half withdrawn by a
pre-registration written before the data existed.

**(iii) The "degenerate collapse" is largely a measurement artifact.** Reply 13 (and my own summary to
you) called this a headline: min cross-layer cosine → **1.0000** by loop 32, present at initialisation.
The number is real; the interpretation was wrong. It compares layer **outputs**, and in a pre-norm
stack every output is `x + branch(x)` — all three share the same residual. Comparing each layer's
**own contribution**:

| loop | cos(outputs) | **cos(increments)** | ‖increment‖/‖h‖ |
|---|---|---|---|
| 8 | 0.9994 | **0.1806** | 0.0348 |
| 32 | 1.0000 | **0.1536** | 0.0096 |
| 64 | 1.0000 | **0.1387** | 0.0053 |

The layers are **not** doing the same thing. They agree because each moves the state by 0.5–3.5% of
its norm. Confirmed two ways: per-loop *scalar* diversity (σ up to 1.0) and per-loop *operator*
diversity (LoRA branches, 168 tensors randomized) both leave cos@64 unchanged. **This closes the
entire conditioning / branch-diversity family without a training run** — and the `od_lora_r2` arm
that was already running confirms it empirically (+0.0941 CE, band unmoved).

**(iv) A published number of mine was false precision.** ρ = 1.7019 came from an *unseeded* power
iteration; three fresh runs gave 1.6578 / 1.6741 / 1.6979, with the published value outside all
three. Seeded and re-measured: **1.6227**, reproducible. The claim survives (1.62 is far outside the
estimator's ~9% bias); the four decimals did not.

**(v) An instrument had been mislabelled for the whole project.** `jacobian_spec.py::sigma_max` never
applied `J^T` — it is plain power iteration on `J`, so it always computed the **spectral radius ρ**,
not the largest singular value. This *helped*: `ρ < 1` is the actual convergence criterion, while
`σ_max < 1` is only sufficient. §2 had been hedging against a claim it already had.

## 4. Two genuine positives found today

**Token-keyed annealing beats fraction-keyed by −0.2208 nats** (~4× the floor; the largest
supervision effect in this project). The recommendation had been "switch at 90% of *steps*", validated
at 2.5M where that lands *before* loop gain emerges — and extrapolated to 10M where it lands deep
*post*-saturation. Keying the switch to an absolute token count instead wins by 0.22. Band identical
in both arms, so it buys **ceiling, not depth**; ΔCE@1 = +0.0923, so it is a trade.

**§4.18's anchor-account falsifier ran and partially failed** — honest outcome. `sw90` at k=5/3/2 was
predicted to give identical bands. **Onset is 12 in all three** (predicted, and a real shift from the
dense control's 8), but band midpoints are 17.0 / 17.0 / **19.6**. So the account is right about the
shallow edge and wrong as stated about the band; it is downgraded accordingly, with two readings left
live rather than picking the flattering one.

## 5. Running now / not running

| stream | state |
|---|---|
| DS `tlab-deep-full` (μ_rec=40, `sw75`) | **EXECUTING**, ~12k/19.5k steps. Read as a §4.16b replication (does the band still track μ_rec=40 under full training?), **not** as an annealing test — the `sw75` axis is withdrawn. Only ≥32-loop artifact |
| local `run_operator_diversity` | arm 3/4. **Motivation retired by 3(iii)**; finishing because it is already paid for |
| DS `tlab-anchor-tokenkey`, Kaggle `tlab-seed-extension` | **DONE, harvested** — sections 3(i)–(ii) and 4 |
| DS `tlab-operator-diversity` | **ERRORED** at 4 min (`ModuleNotFoundError: tokenizers`; its `cmd` installs it but the install never ran). Not relaunched |

**Open, ranked:** learned depth gate (`gate_scalar` +32 params / `gate_state` +14,336) — the only
remaining candidate for a *positive*; gradient checkpointing to recover the μ_rec=56/44 arms that
OOM'd; §2's DEQ reframe (DEQ's headline claim is *constant memory*, not compute, so the task's
objection attacks an axis DEQ isn't optimising); the three-instance scaling regularity.

**Known limitation, verified today:** `depth_init` scales by `1/√(2·n_loop_eff)` with `n_loop_eff`
**fixed at 24** in every checkpoint, while schedules ran at mean depth 18 and 40 (ratios 1.15×,
0.77×). **In-job pairs are unaffected** — both arms share the same wrong init — so no paired result
here is confounded. Cross-schedule comparisons carry it.

**Still the user's, not mine:** §1 (the idea narrative — the task grades it separately and explicitly
warns about LLM-sourced ideation) and D3 (which checkpoint ships; my recommendation is the control,
since the penalty arm is the only one that converges and would ship an artifact contradicting §2).

---

## 6. Self-check

If you cannot answer these from the document above, please ask rather than assume — each one is
something an earlier reply of mine stated incorrectly, or a place where two of our numbers look
contradictory but are not.

1. **Is the supervision-annealing result positive or negative?** *(Both. The CE claim is withdrawn at
   n=4; the band claim holds at 4/4. Any summary that says only one of those is wrong.)*
2. **Does this model's hidden state converge?** *(No — and it doesn't in an untrained model either.
   If your answer didn't include "architectural, present at init", the scope is missing.)*
3. **What does `cos → 1.0000` across layers mean here?** *(Mostly that the layers share a large
   residual. Their own contributions sit at cos ≈ 0.14–0.18. If you took it as degeneracy, that's the
   claim I retracted today.)*
4. **Why is the report's perplexity worse than Parameter Golf's?** *(Primarily ~70× fewer training
   tokens, not architecture. Quoting the bpb gap without the token ratio misrepresents both.)*
5. **Name the four instrument classes that fail to reach the per-token depth headroom.** *(Label-free
   rules; static readout mixture; annealed-checkpoint retest; oracle-depth KV cache. If you can only
   name two, you have an older reply.)*
6. **Which is the decisive row in the seven-intervention table, and why?** *(Gated injection — its
   pre-registered mechanism check **succeeded** and the loss got worse anyway. A null where the
   mechanism failed would prove much less.)*
7. **Is `tlab-deep-full` testing annealing?** *(No. Its `sw75` axis is withdrawn. It is a §4.16b band
   replication.)*
8. **Does the `n_loop_eff = 24` bug invalidate the annealing comparisons?** *(No — in-job pairs share
   the same wrong init. It affects cross-schedule comparisons only.)*
9. **What is the strongest claim in this report?** *(Reasonable people could differ, but I would say
   §2/§4.3: saturation without convergence, measured three ways, contradicting the task's own premise
   — because it needs no seeds and survives every withdrawal above. It is **not** the annealing result
   and no longer the degenerate collapse.)*
