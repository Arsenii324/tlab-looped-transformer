> **Dated working record — not a maintained document.** Last committed 2026-08-23; the text itself may be older. Kept intact rather than rewritten, because this project's rule is that superseded statements stay visible with their corrections. **Current numbers are in [`submission/`](submission/) and [`report.md`](report.md); where this file disagrees with them, they win.** See the repository README's *How to read this repository* table.

# How to design an experiment here so it does not need a rerun

STATIC doc. Derived from this project's own measured noise, not from general advice.

## The measured noise floor (know these before designing anything)

| source | size | removable by |
|---|---|---|
| training seed, same comparison | **0.25 nats** (0.746 vs 0.496) | pairing (same init + data order), NOT by averaging |
| eval sample, same checkpoint | **0.065 nats** | a frozen common-random-numbers eval set |
| in-training 6-batch eval vs post-hoc dense | up to **0.061 nats** | never use in-training numbers for claims |
| fp32 reduction order | ~1e-7 | ignore |

**Four of the five screening effects in §4.1 were smaller than the first row.** That sweep could not
have worked. Check this table before running anything.

## The ladder — always go down it in order

**Tier 0 — post-hoc on an existing checkpoint. Zero training variance, minutes.**
If a hypothesis can be tested by intervening at inference, it costs no training noise at all and
usually no GPU-hours worth counting. Everything in this tier landed first-time in this project:
radial clamp, per-token exit dump, cross-depth KV grid, state dynamics.
*Ask first: can I test this without training?* Most dynamics questions can be.

**Tier 1 — paired training arms, ONE parameter swept over 3-5 levels.**
Same init seed, same data order, token-matched (never wall-clock — see §4.1's confound). Evaluate on
the frozen set; bootstrap the paired delta over sequences.

**Tier 2 — scale up only what Tier 0/1 showed is live.**

## The five rules that decide whether a result survives

1. **Pair, don't average.** Pairing removes a variance source; extra seeds only estimate it better.
   Same frozen eval set, same init, same data order.
2. **Sweep levels and read the ORDERING.** The clamp's CE differences were 0.006 nats — unusable as
   a pairwise claim — but the optimum moved 5 -> 15 -> 24 monotonically. Noise does not produce
   monotonicity across 4 points by accident (~1/12 by chance). **An experiment readable only as a
   single pairwise difference is the last resort, not the default.**
3. **Pick the highest-SNR outcome.** CE at one loop is noisy; loop gain (CE@1 - CE@best) is a
   within-model difference so level shifts cancel; argmin location is ordinal and immune to them.
4. **Every instrument needs a control reproducing a KNOWN number.** Unclamped -> published curve at
   1.9e-07; cross-depth diagonal -> clean per-loop curve; n_prelude=0 -> bit-identical to the
   pre-sandwich model. A failed control tells you it is the instrument, and saves the "was that
   real?" rerun.
5. **Pre-register the prediction and a threshold.** Written before the run, with the number that
   would falsify it. Prevents reading noise as confirmation.

## The most expensive mistake in this project

**A negative result obtained on flaky hardware is not a result until it is reproduced elsewhere.**
§4.4 reported that an 81M untied stack could not be trained at all, across six configurations, and a
mechanistic account (weight tying as an implicit regulariser) was built on top of it and used to
explain other results. Every attempt had run on MPS -- the backend this project separately documents
as producing silent zeros and fake NaNs under load. On CUDA, the same architecture at the same three
learning rates trained to completion with no NaN, **including the exact LR that had NaN'd at step 13**.
The negative was the hardware. It survived a full session because it was convenient and because
nothing forced a second backend. Cost: one wrong section plus the reasoning that leaned on it.

## Anti-patterns this project actually committed
- Wall-clock-budgeted arms (hands more data to the cheaper config; flipped 2 of 5 axes).
- Single-seed A/B on an effect smaller than the seed spread.
- Reading a "best loop" off a coarse grid that did not contain the true optimum (loop 11 / loop 7).
- A verdict line using a bare `<`, which reported a win on a 0.0001-nat margin.
- Bucketing on a signal that is identically zero at the bucketing point (dnorm/kl at loop 1).

## Fixed seed is not a replicate (measured 2026-08-23)
Two runs of an identical config at an identical seed, 3.5h apart under different drivers, ended
**0.031 and 0.068 nats** apart (`sup_uniform4_32_s{0,1}` vs `sd_dense_k5_s{0,1}` — nominally the same
arm; neither was intended as a replicate). A 30-step probe with no chunking shows **CPU bit-identical
(0.000e+00)** and **MPS nondeterministic (9.5e-07 over 30 steps)**; that per-step noise is amplified
by the optimisation over 1,219 steps into the figures above. A second, structural contributor:
`chunked_runner.py` cuts training into 240s **wall-clock** chunks and rebuilds the optimiser at each
one, so momentum resets land at load-dependent steps.
**Consequences.** (1) Any single-arm difference under ~0.05 nats is not a result. (2) argmin over a
loop curve is unusable — 52/71 stored curves have argmin margins under 0.005 nats; use
`src/plateau.py` (band within tol of the minimum) and report a plateau, never a point. (3) "Paired,
same seed" does NOT mean low-variance on MPS; state device alongside any noise claim, because the
measured floor is MPS-only and the CUDA floor is inferred, not measured.
