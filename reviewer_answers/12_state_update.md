# State update — 2026-08-23 ~15:00

Since reply 11. Four new measurements, one decisive negative on your proposal, one premise of yours
refuted before it cost anything, and an audit that found the report grading itself on two different
scales.

---

## 1. Your scale-clock premise is false, and I checked it before building

You derived the clock from `u_t → u*` being a fixed point of `G(u) = F(u)/‖F(u)‖`, with
`‖u_t − u*‖ = O(1/t)` and "measured exponent −1.082". **That −1.082 is the consecutive-step
exponent, not the distance-to-limit exponent.** They are different quantities and only the second
bears on convergence. Measured (`src/angular_convergence.py`, new):

| model | power law `A·t^−b` *(convergent, 2 params)* | `C·ln(T/t)` *(log-drift, 1 param)* | motion over loops 129→384 |
|---|---|---|---|
| 90M control | t^−0.657, R² **0.748** | C = 0.1539, R² **0.986** | **0.1836 rad** |
| 46M no-renorm | t^−0.651, R² **0.744** | C = 0.1508, R² **0.983** | **0.1816 rad** |
| 90M norm-penalty | t^−0.771, R² 0.754 | C = 0.1021, R² **0.994** | 0.1083 rad |

**Log-drift wins with half the parameters.** The mechanism is §4.3's own two measurements composed:
step `~1/t` **and** consecutive steps aligned at `cos → 0.9999`, so accumulated motion is
`Σ C/s ≈ C·ln(T/t)`, which **diverges**. The state travels a fixed angular distance per *doubling* of
loop count, forever. **There is no `u*`.** Any intervention motivated as "break the fixed point" is
aimed at something that is not there.

This is also the strongest form of §2's claim we have — non-convergence *by construction* rather than
from a marginal `ρ ≳ 1` reading.

## 2. I built and ran the clock anyway, on the surviving weaker motivation. It fails badly.

Motivation after the correction: the block cannot see `‖h‖`, so it cannot behave differently late
than early. Implementation: `h_in *= 1 + w·(log rms(h_t) − log rms(h_1))`, per token, **+448 params
(0.0049%)**, `w` zero-init so it is bit-identical to the current model at step 0 (verified,
max|diff| = 0.0). Three in-job arms at 2.5M, seed 0, config derived from the reference checkpoint.

| arm | CE@1 | CE_best | plateau | mid | **‖w‖** |
|---|---|---|---|---|---|
| `sc_ctrl` | 5.5129 | **5.4202** @8 | [8,16] | 11.3 | — |
| `sc_clock` | 6.8802 | **6.7845** @8 | [4,16] | 8.0 | **1.3412** |

**ΔCE_best = +1.3643 nats — about 20× the MPS replicate floor.** And `‖w‖ = 1.34`, so the model did
**not** decline the parameter: it took it and got much worse. **This is the failure mode you named in
advance** — *"a large `w` is a feedback loop from the block's output into its own conditioning, which
could destabilize. Zero-init mitigates but doesn't eliminate it."* It does not eliminate it.

I pre-registered "read geometry first, not CE, because 448 params against a 0.031–0.068 floor cannot
be resolved on loss." That pre-registration is now moot in the honest direction: a +1.36-nat
regression is not a subtle geometry question, and the arms are no longer comparable. The third arm
(clock + `sw90`) is still running; I will report it, but I do not expect it to rescue this.

## 3. Two more instruments run, both against your suggestions

**SCSE's anchor response** (`src/anchor_response.py`, construction verified from
`papers/sources/2607.27656/lvr.tex` §131–137). **All three of my pre-registered predictions failed
and the registered falsifier fired.** `‖b‖` tracks the *state* scale (1715 / 429 / 4.7), not `‖e‖`
(~1.5 in all three arms), and `R` is comparable across a **380× norm spread**. What survives is
better than what I predicted: the anchor sits **299–775× from being a one-step fixed point**, and the
constant forcing bias **exceeds the model's own per-step motion from ~loop 2**, reaching 2.7–5.3× by
loop 64 — scale-invariantly, which their single-model study could not show.

**Blayney's degenerate branch — we are in it.** From 2604.11791 line 279 (verified, macros resolved):
pre-norm without input injection reaches a *degenerate* fixed point where "each layer converges to
the **same** fixed point", diagnosed by the lowest cross-layer cosine → 1. Measured on ours:

| checkpoint | loop 8 | 32 | 64 | 128 |
|---|---|---|---|---|
| 90M control (min cos) | 0.9994 | **1.0000** | **1.0000** | **1.0000** |
| 90M norm-penalty | 0.9958 | 0.9995 | 0.9998 | 0.9999 |

At `‖e‖/‖h‖ ≈ 1e-3` we are effectively un-injected at inference, and all three layers collapse onto
one direction by loop 32. Caveats I am carrying: their experiment is **randomly initialized 12-layer**
models (line 277), ours is trained with 3; and cos→1 in *direction* is not a state fixed point, which
§1 above shows does not exist.

## 4. R45 answered — and it closes §4.7 rather than caveating it

The annealed checkpoint you ranked highest now exists (trained locally; every DS job discarded its
weights). Matched pair, same dump protocol, 524,288 scored tokens, split **by sequence**:

| | dense control | annealed | |
|---|---|---|---|
| best fixed depth | 10 | **17** | 1.70× deeper |
| CE at that depth | 5.5011 | **5.4404** | −0.0607 |
| **oracle headroom** | **0.2008** | **0.2032** | **+1.2% — unchanged** |
| best label-free rule | −0.0001 | −0.0003 | **0.0% / 0.1% captured** |

**The result is the invariance.** Annealing demonstrably changes the trajectory — 1.70× deeper band,
better ceiling — and leaves exploitable per-token headroom **identical and still unreadable**. The
literature's trajectory explanation for §4.7's negative (dense supervision pins every loop to the
output manifold, so confidence signals saturate; unpin them and a signal appears) has now been
**tested against its own prediction and failed**. §4.7 goes from "you may have measured the wrong
model" to closed.

## 5. An audit found the report grading itself on two different scales

A full correction-propagation pass (fork, read end to end, every finding verified against artifacts
before I applied it) returned 11 findings. The two that matter to you:

**The report used the conservative noise floor where annealing looked bad and the permissive one
where it looked good.** §3.5 judged μ_rec=40 against the CUDA **terminal** floor (0.0541); §8
described the μ_rec=18 result as "4–5× the measured floor", which is the **dense** floor (0.0150).
Same class of arm, two standards. Underneath is a real gap: **§4.15 never established a floor for an
annealed arm** — dense for 90% of training, terminal for 10%. Both floors are now stated at both
sites, and §4.15 says plainly that annealing's −0.0710 is **4.7× the dense floor and 1.3× the
terminal floor, and we never measured the floor of the arm we actually ran.**

**ρ = 1.7019 was false precision.** The power-iteration start vector was unseeded. Three fresh runs
gave 1.6578 / 1.6741 / 1.6979 — the published value lay *outside all three*. Seeded and re-measured:
**1.6227 / 1.0460 / 1.0049 / 1.0013**, identical across runs. The claim survives (1.62 at loop 2 is
far outside the ~9% estimator bias); the four decimals did not, and the error flattered the result.

Others: §4.16c's **heading** still asserted the claim its own body withdrew; the retracted 1.4× was
load-bearing inside the novelty claim; §3's architecture table reported 9,065,056 params
(`state_renorm=True`) for a model that ships at 9,064,608. **The pattern the audit named is the
useful part: corrections were landing in prose and not propagating to headings, summary boxes and the
load-bearing-choices table — which is what a grader reads first.**

## 6. Running now

| stream | what | state |
|---|---|---|
| DS `tlab-deep-full` | deep artifact, μ_rec=40 **sw75** | EXECUTING, step 8100/19531, 1296 tok/s. Curves only. **Note: sw75, which §3.5 has now narrowed away from** |
| DS `tlab-anchor-tokenkey` | §4.18's falsifier (`sw90` at k=5/3/2 + in-job dense) then token-vs-fraction at 10M | EXECUTING, cheapest arms first so an early cancel still answers the falsifier |
| Kaggle `tlab-seed-extension` | (sw90 − dense) paired difference at seeds **2 and 3** | pushed |
| Local MPS | scale clock, arm 3/3 | running |

The Kaggle job is the direct answer to §5's gap: rather than measure "a floor", get **n=4 paired
estimates** of the quantity §3.5 actually claims. Pre-registered: if the four straddle 0, or the mean
falls inside 0.0541, the annealing recommendation is withdrawn to "not resolved at this budget".

## 7. What I did not do, and why

- **Layer Duplication (2510.25741).** Skipped. The case for it was "buys credibility", and its own
  reconciliation — post-hoc surgery on frozen weights measures which layers *tolerate* repetition,
  not which *learn to be* repeated — means it bears on no claim we make.
- **Sparse Layers (2605.09165).** Not obtainable here, so §3.3 states the density objection **on its
  own logic with no number quoted from them.** After the 73% retraction, a relayed number is not
  usable.
- **2606.20075.** Now verified from the tarball (it arrived mid-session): `OS-Latent` **9.8/18.3** vs
  `OS-No-CoT` **18.7** — an unsupervised chain is no better than no chain. Genuine opposite sign to
  §4.14/§4.17, with the scope stated (their steps carry *distinct rationale content*; our loops share
  one target). The convergence is the better half: their `OS-GR` arm *is* a decodability anchor,
  independently isolated.
- **Gated injection** (`α⊙z + δ⊙W_in·v`) — the third cell on the normalisation axis, which §4.1 never
  tested. In progress.

**Still open and still yours:** §1 is empty, and the checkpoint choice (D3) is unmade.

---

## 8. Every suggestion you have made, with a disposition. Nothing left silent.

*Added because silence on a suggestion is indistinguishable from having missed it.*

### Acted on

| suggestion | what happened |
|---|---|
| **Massive-activations ablation (2604.11791)** as a mechanism for `state_renorm` | **Done, §4.1.** Verified from the tarball, macros resolved. **And your flagged line-49-vs-104 contradiction does not hold** — line 49's contrast is with the *retrofitted* models ("which lack this norm"), not with Huginn. Both Huginn and Ouro normalise; the retrofitted series does not, which is why it is the arm that shows stages and the one they ablate. Your caution was right in principle; the specific tension did not survive reading both lines |
| **SCSE `R_t` anchor-response**, pre-committed to §4.3 as explanation not §3.5 as evidence | **Done**, and I honoured the placement. All three of my predictions failed; §3 above |
| **MixerLoop over-ranked; FullLoop has the lower NLL at 15M** | **Already in the report** before your message — §8 carries 2.995 / 2.946 / 2.936 and an explicit *"This corrects an earlier draft of this section"* |
| **`sw90` second-seed floor** — the one run you'd spend compute on | **Done better, I think.** Rather than a same-config replicate ("a floor"), the Kaggle job extends the **paired** (sw90 − dense) difference to seeds 2 and 3, giving **n=4 estimates of the quantity §3.5 actually claims** instead of a floor to compare against. Pre-registered withdrawal condition stated |
| **§4.10 as damped-Euler-in-the-wrong-regime** (2605.23872) | **Done, §4.10.** Verified `method.tex:75`: their goal is *"not to advance integration to t=K, but to better approximate the same endpoint x(t=1)"*, on a **frozen checkpoint**. In pretraining there is no trained one-step endpoint to return to, so the null is the mechanism's prediction, not a disappointment |
| **§3.3's density threat** | **Done, §3.3** — stated on its own logic, with Sparse Layers logged SECOND-HAND and **no number quoted from it** (not obtainable here). Added the distinction that density threatens the *architecture's competitiveness*, not *annealing's transferability* |
| **Blayney's degenerate branch / layer fixed-point separation test** | **Run** — §3 above. We are in it: min cross-layer cos **1.0000 by loop 32** |
| **The layer-band / §8 question** | Superseded: the degenerate-fixed-point result is the sharper statement |

### Declined, with the reason

| suggestion | why not |
|---|---|
| **Layer Duplication (contradiction in §8)** | Your own reconciliation is the reason: post-hoc surgery on frozen weights measures which layers *tolerate* repetition, not which *learn to be* repeated. It bears on no claim this report makes, and "buys credibility" is not a result. **Skipped deliberately, not overlooked** |
| **"Don't touch the architecture"** | **I did**, twice — the scale clock and (in progress) gated injection. Reason: the task grades idea-*testing*, and the user asked for it explicitly. Both are zero-init/off-by-default, pre-registered on geometry, and the scale clock has already returned a clean **+1.36-nat negative** that is worth more than not having run it |
| **"Don't run `R_t`, prefer the floor measurement"** | I ran both. `R_t` cost one forward pass and falsified three of my own predictions; the floor question got the better instrument (n=4 paired). Neither displaced writing |

### Owed and not yet done — named so they are not silently dropped

| suggestion | state |
|---|---|
| **Gated injection** (`α⊙z_t + δ⊙W_in·v`, learned per-channel decay) — the **third cell** on the normalisation axis, which §4.1 never tested (hard RMSNorm vs *nothing*, never the soft decay both reference implementations use) | **in progress**, ranked #1 of the remaining architectural gaps |
| **Gradient checkpointing → recover the μ_rec=56 / 44 arms that OOM'd** | **not started.** Documented in §6.0b as free and unused; it is the memory route to the deep schedules the task is actually about |
| **§2's DEQ reframe** — DEQ's headline is *constant memory*, not compute, so the task's "convergence wastes compute" objection attacks an axis DEQ isn't optimising | **not started.** I have the tarball. The sentence I'd write is neither theirs nor the task's: *usefulness ends at loop ~8 while the trajectory runs past 128, so the problem is not that convergence wastes compute but that usefulness ends long before convergence does* |
| **The three-instance scaling regularity** (our norm penalty 12×, our annealing's schedule-axis reversal, exact-ZOH's 14× shrinkage) as a methodological finding about the subfield | **not started.** Caveat I'd state: two of the three are ours, and the blogs + Done Right are one author, so it is not three independent groups |
| **Blayney's stability inversion** — models that reach a fixed point keep enacting stable stages at arbitrary test-time depth; those that don't, don't | **partially.** I have the degenerate-fixed-point half; the "Stability to Unseen Numbers of Recurrences" quote is **unverified** and I will not write it until I have read it in the source |
| **Sparse Layers' early-exit finding** (loop boundaries are the superior exit point; the benefit comes from looping, not sparsity) — would close an escape route on §4.7 | **blocked.** Not obtainable here. It would be a relayed number, and this project retracted a claim for exactly that (§6.0 row 22) |
| **Finish 2606.20075** | **done** since you wrote that — verified from the tarball, §4.18 |

### On the §1 argument

You are right that the report never says what I thought *before* I measured, and that the four dated
turns are in `LOG.md`. **I am not going to write §1** — the task grades it separately and explicitly
asks it not come from an LLM, and that constraint is the whole point of the section. What I can do
and have done is make the raw material findable: the dated reversals are in `LOG.md`, the retractions
are §6.0's 31 rows, and `reviewer_answers/08` lists the unknown-knowns. The narrative is the user's.
