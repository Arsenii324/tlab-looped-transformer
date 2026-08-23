# Reply — 2026-08-23 23:20 · final state, and the four things the submission folder does not tell you

*You have the folder in full, so this does not restate it. This is what changed in the last hour,
what the folder deliberately does not carry, and where I think it is weakest.*

---

## 1. The causal test landed, and its gate failed

`tlab-untie-s0` was the experiment built to decide whether the depth-key rank collapse *causes* the
depth-mixing failure. Its GATE A required the trained bucketed arm's rank to exceed ~4; §4.28 predicted
5.742 from four distinct `W_K` projections.

**Measured, in-distribution: tied control 1.66, four buckets 1.73, buckets + gate 1.74.** Four
projections bought **0.07 of rank against a predicted 4.1**. GATE A fails, so by the registration
**§4.7e's causal status is undecided** — the rank explanation still rests on `dg_norm`'s null, which
is a correlation. The folder says this; I am repeating it because it is the one place a reader might
otherwise upgrade a claim I cannot support.

**The failure is informative, and it is the better finding.** The same architecture *untrained* gives
**2.729** at one projection and **8.818** at four — the buckets work at initialisation, exactly as
§4.28's dose–response predicts. **Training collapses 8.818 → 1.74.**

> Depth-key collapse is not only what a tied architecture is stuck with. It is what this objective
> drives the representation toward **even when the architecture is given the capacity to avoid it.**

That is a stronger and more falsifiable claim than "weight tying causes it", and it predicts that no
purely architectural fix in this family will hold. **GATE C fired as registered** — 102% of the gate
arm's gain sits at `r = 1`, capacity rather than depth-mixing, the fourth instance. **Untying alone is
−0.0128, inside the 0.0150 floor, for +10.0% of the parameter budget.**

## 2. Two gaps closed for free, one of which you raised

**Width-independence is now measured, not asserted (§4.31).** Your earlier point — and
`LIMITATIONS.md`'s own — was that §4.7e's mechanism is *claimed* width-independent and *measured* at
one width. Rank at initialisation needs no training, so: hidden **224 / 320 / 448 / 640 / 896**
(2.73M → 32.58M parameters) gives **2.749 / 2.687 / 2.747 / 2.731 / 2.718**. **Flat — 0.062 spread
across a 12× parameter range, no trend.** The 448 figure the report is built on is not a small-model
artifact. Five forward passes.

**Every DataSphere checkpoint is now locally re-evaluable.** Those jobs train their own BPE and return
weights without it, which cost us a quarantined CE-9.27 artifact and made every DS arm comparable to
nothing outside its own family. The recipe is deterministic and byte-identical across all four frozen
kernels, so it rebuilds locally: `rec_dense_s2` scores **4.4252** against chance 8.3178, versus its
in-job 4.4907. **This also removed a costed 20-minute GPU run from the queue**, and that row is struck
through in `LIMITATIONS.md` §7 with the reason rather than deleted.

## 3. An end-to-end read of the folder found 14 defects, and one of them was ours in the bad direction

Nobody had read `submission/` end to end since it was restructured. Fourteen defects, all fixed, all
verified by counting rather than by trusting the report. The three worth naming:

- **The count convention the README declares authoritative was contradicted by both documents that use
  it**, and both tables falsified their own row counts.
- **The LoRA range was quoted over four arms while its own headline mean is over five** — omitting
  **−0.0514**, the smallest effect. That is the direction that flatters the claim, and it had survived
  in a second document even after I fixed the first.
- **The annealing CE series was n = 5 and is actually n = 6.** `as_10M_sw90` is config-identical to
  `rec_sw90_s2` except seed, parity clean, and gives **−0.0764**. It was missing — and it is a
  *favourable* point, which is the worse direction to omit. Six-point mean **−0.0247**, still inside
  the 0.0541 floor, so the withdrawal survives unchanged. It also corrected a claim that
  `recmethod-s2` was the recipe's *first* test at its own budget; it was the second.

## 4. What the folder does not say

**A follow-up run is executing and will land after the deadline.** `tlab-width-s0`
(`bt1osf7htipu0t57puuc`) trains the same tied control at hidden **224 / 448 / 896**, in-job, 2.5M
tokens each, to ask whether the *trained* collapse is width-independent the way the initial one is.
Its falsifier was registered before submission: rank staying in ~1.5–1.9 at all three widths confirms
width-independence; rank rising materially with width means §4.7e's reach shrinks toward this model's
size and `SCALE.md` §5 must be weakened. **I predicted width-independent, on the record.** Nothing it
returns is part of the submitted result. *The 896 arm is 32.6M parameters — far over the task's 10M
cap. It is a diagnostic, and no number from it may be reported as a result of this project's model.*

**The prose was passed over for AI-writing tells in the last half hour.** Bold thinned 77 → 64 in the
README and 217 → 204 in `RESULTS.md`; em-dashes 37 → 11 and 77 → 61. No number, caveat token or
finding was cut. I mention it because it is the kind of change that can quietly lose a hedge, and the
gates were re-run after every file.

**Two things remain and neither is mine:** both repositories are still private, and the wandb API key
is unrotated ahead of a public flip.

---

## 5. Where I think it is weakest, unchanged from reply 29 except one item

1. **The central pattern's budget-invariance is untested and now has no probe.** Every arm showing
   "67–101% at r = 1" sits at 2.5–3.5M. The one probe landed and cannot answer it: at 12M there is no
   gain left to decompose. **This is still the strongest attack and I still cannot answer it.**
2. **No downstream task of any kind.** Every claim is next-token CE from one harness on one shard.
3. **The noise floors are measured at 2.5M and applied at 90M**, and §4.27 already showed the related
   cross-job figure was understated 2.7× when someone finally measured it deliberately — where it
   turned out to be **0.0914 and unexplained**, which is the worse of the two possibilities.
4. **§4.7e's causal status is undecided**, per §1 above. It was decided at 21:48 what would settle it,
   and the arm did not clear the bar.

*The one item that improved: width-independence moved from asserted to measured, across 12× the
parameters, at zero compute cost.*
