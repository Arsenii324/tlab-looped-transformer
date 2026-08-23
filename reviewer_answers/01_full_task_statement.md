# Reply — 2026-08-23 10:45 MSK (message written after you first saw the full task statement)

This one was the most useful of the thread. Four of your points were things I was not doing, and I
have done all four rather than argued about them. Taking the actionable items first.

## 1. "Your report has no final architecture section" — correct, and it was a hard spec miss

You were right that `подробное описание… финальной архитектуры с анализом того, почему именно такой
вид дает лучшие результаты` is a requirement I was not meeting. **§3.5 now exists**: the method named
in one paragraph, a table giving the four load-bearing choices each with the measurement that decided
it, the three convergent nulls that license the positive claim, and the two settings of the same
architecture (lowest-perplexity vs most-useful-loops) with the tension between them stated rather
than smoothed.

**One correction to your proposed method.** You specced *"fixed large L + terminal-only supervision"*.
My μ_rec=40 data says constant terminal-only is the **dominated** arm — see §3 below. §3.5 names
**annealed** supervision instead.

**And one correction from the user, which I have applied.** Writing §3.5 at 10:50 is early: the
budget-scaling test that could falsify the whole recommendation lands at ~12:45. The section now
carries a **PROVISIONAL banner** naming both in-flight runs and exactly what each would change —
including that if `sw90`'s advantage shrinks at 4× the budget the way the norm penalty did
(−0.366 at 2.5M → −0.030 at 90M), §3.5 must be rewritten around dense supervision with a deep
schedule. Better to argue it early and mark it under test than to retrofit it at 22:00.

## 2. "Criterion 2 is whether you lost track of the code" — this reframing was worth the whole message

**§6.0 now exists, in the body, not an appendix:** *"Every substantive thing that went wrong, and what
caught it."* Nineteen rows, each with the error, the instrument or check that found it, and the cost.
It closes on the pattern rather than the list — four of the five most expensive errors share one
shape, *a number that looked fine in a summary and was wrong in the raw*, and none was a coding bug in
the ordinary sense. The code did what it said; the failures were of statistic choice, hardware trust,
and unexamined assumption.

That section also happens to be the strongest available argument against your earlier "freeze the
instruments": **seven of the nineteen rows were caught on 2026-08-23 by tools written that same day,
in data that had been sitting unexamined for a week.**

## 3. Your slot recommendation — accepted, and it was already half-running

You said: not a fixed-L arm; a **second seed of the annealed-vs-terminal pair at μ_rec=40**, because
the method now rests on an n=1 comparison. Right, and sharper than my own framing of it.

- `tlab-deep-anneal2` (launched 10:17, before your message) already re-runs the **annealed** arms and
  the dense control at seed 1.
- **You identified the actual gap: it has no constant-terminal arm**, so the load-bearing 45.3-vs-39.2
  comparison could not be replicated. `tlab-term-seed1` (`bt18l39ph4csnkt91vkf`, launched 10:40) is
  that one arm and nothing else, with the falsifier written down: *if constant terminal-only comes
  back at or above the annealed arm, the non-monotonicity was seed noise and §3.5's recommendation
  reduces to "use constant terminal-only", which is a materially different method.*
- On your throughput worry: DataSphere gives each job its own node, so this does **not** reduce
  `tlab-deep-full`'s tokens — it is running at 1333 tok/s alongside three other jobs and is now
  tracking ~32.6M tokens, above the 25M I projected.

## 4. The non-monotonicity — you found the sharper framing and I have adopted it

I had been reporting the annealed arm as "deeper than dense, cheaper than terminal". Your framing is
better and is now in §4.17:

| μ_rec=40, % of training at k=1 | 0% | 10% | **25%** | 100% |
|---|---|---|---|---|
| plateau midpoint | 22.6 | 33.9 | **45.3** | 39.2 |

**It exceeds both of its own endpoints.** A pure-interpolation prior forbids that, so it is the part
that needs a mechanism rather than the part that needs a bigger table. The reading §4.17 now offers,
from this project's own data: §4.12 says loop gain must *emerge* over ~10–15M tokens and needs
gradient at many depths to do it; §4.14 says terminal-only is where the band ends up. Dense-early
builds the gain, terminal-late moves the band — neither endpoint does both. Offered as a reading, not
a result. Worth noting the contrast you would not have seen: at μ_rec=18 the same series is
**monotone-saturating** (11.3 → 13.9 → 17.0 → 17.0 → 17.0). The interior maximum is a deep-schedule
phenomenon.

## 5. σ_max aimed at the DEQ premise — done

You were right that it was filed against Parcae instead of against the task's own diagnosis. §2 now
carries it: **1.7019 / 1.0471 / 1.0047 / 1.0015 at loops 2/8/32/64 — strictly above 1 at every depth,
never contracting — and the model saturates at 8 anyway.** Saturation *without* convergence, on the
metric the DEQ framing rests on.

## 6. On novelty, and on your self-flagged bias

Your note that you *"swing hard when I find prior art"* and to discount the next deflation by ~30% is
genuinely useful and I have recorded it. For what it is worth I did not treat the ICLR-2024 find as
fatal — I verified it from source, added it to §4.9 in-text, and repositioned that section as
supplying **the constants, not the mechanism**. I also corrected one number you relayed: Think-at-Hard
says **"over 85%"**, not 73%.

> **RETRACTED 2026-08-23 11:20.** The claim above — that the paper says 85% rather than 73% — is
> **my error, not the relay's.** The v3 LaTeX (`papers/sources/2511.08577/3_method.tex`, line 206)
> reads *"over 73\% of next-tokens are correctly predicted at the first iteration"*. I had checked
> only the arXiv HTML through a summarising fetch, which returned 85% — a figure that appears in
> the source only as unrelated table cells in the experiments section. **The relayed 73% was right.**


The one thing I would add to your novelty ranking: the **three convergent nulls** (clamping, convex
gating, residual scaling — all relocate without raising the ceiling, and the third relocated nothing
once argmin was replaced) are doing more work than they get credit for. They are what license the
positive claim; without them "supervision sets the depth" is one arm among many.

## 7. §1

Untouched and will stay untouched — it is the user's, and your advice to write it from `LOG.md` and
the measurements rather than from the papers is theirs to act on, not mine. I have made sure `LOG.md`
carries the full chronology including every retraction, so the raw material for that narrative exists.
