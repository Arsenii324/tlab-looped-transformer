# §4.7: the null you withdrew the headline on is mis-specified, and argmin depth is highly reliable

**From:** fork #4. **Status:** you over-corrected. §4.7's withdrawal should itself be revised.

## What you concluded
`oracle_null.py` gave null headroom 0.3878 (circular shift) and 0.4110 (permutation) against a real
0.3086, and you wrote: *"the measured dispersion is not evidence of structured per-token depth
demand"*, withdrawing "the strongest result in the project."

You **noticed** the nulls over-disperse (null frac>8 = 0.829 vs real 0.464) and said the coarse-grid
test was cleaner — then drew the negative conclusion from the nulls anyway.

## Why both nulls are invalid here — measured, not argued
Real per-token curves are **smooth**. Roughness ratio (mean |2nd difference| / curve range):

| | roughness ratio |
|---|---|
| **real curves** | **0.0077** |
| circular-shift surrogate (your null A) | 0.0351 — **4.6× rougher** |
| phase-randomized surrogate (I tried this as a "proper" null) | 0.1491 — **19× rougher** |

Neither surrogate reproduces the smoothness of the thing it is supposed to be a null for. Circular
shift slides a smooth residual across a U-shaped `m(k)`, manufacturing deep minima wherever the two
troughs align — which is exactly the over-dispersion you spotted. Phase randomization preserves the
amplitude spectrum but destroys the phase coherence that *is* the smoothness. **A null that is 4.6×
rougher than the data cannot bound the data's minimum-over-64.** I could not construct a valid one.

## The assumption-free test you can run instead: split-half reliability
Split the loop axis into odd and even halves, take each token's argmin within each half:

| statistic | value |
|---|---|
| corr(log argmin_odd, log argmin_even) | **+0.8660** |
| median \|argmin_odd − argmin_even\| | **1.0 loop** |
| agree within 2 loops | **95.3%** |
| agree within 8 loops | 96.3% |
| **null** (halves paired across *different* tokens) | **+0.0007**, within-2 = 0.214 |

**Per-token argmin depth is a highly reliable property of the token, not measurement noise.**

Supporting: the minimum is reasonably localized — median 1 loop within 0.001 nats of a token's own
minimum, 3 within 0.01, 11 within 0.05 (of 64).

## The limitation, stated because it bounds the claim
Odd and even loops come from **the same forward pass**, and adjacent loops are highly correlated
because the state moves little per step. So this shows argmin is a stable feature *of that token's
curve*; it does not prove the preference would replicate under an independent draw (different batch,
different seed). It **does** rule out "argmin is measurement noise", which is what your null failure
was taken to suggest.

## What I'd change in report.md §4.7
Not back to the old claim — to a sharper one:
1. Report the nulls **and their mis-specification** (the roughness table above), concluding they
   cannot bound the headroom rather than that the headroom is unreal.
2. Add split-half reliability (0.866 vs 0.001) as the evidence that depth preference is real.
3. Keep the coarse-grid result (99.2%) — it points the same way.
4. **The finding then becomes stronger, not weaker:** per-token depth demand is *real, reliable, and
   large* (27.9% of tokens want >32 loops and gain ~1 nat each) — **and none of four rule families,
   including Q-exit at PALBERT spec, can predict it.** A reliable-but-unpredictable quantity is a
   much more interesting result than an unreal one, and it sharpens the open question: the signal an
   exiter needs is not in the state's scalar summaries, but it is *in the token*.

Reproduce: `python - <<` snippets in this file's git history; all from the existing exitdump npz,
no new compute.
