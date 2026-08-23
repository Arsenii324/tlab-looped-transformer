> **Dated working record — not a maintained document.** Last committed 2026-08-23; the text itself may be older. Kept intact rather than rewritten, because this project's rule is that superseded statements stay visible with their corrections. **Current numbers are in [`submission/`](submission/) and [`report.md`](report.md); where this file disagrees with them, they win.** See the repository README's *How to read this repository* table.

# Brief for the "find a thing I don't have a clear hold on" subagent

Written by the main agent, 2026-08-23 ~01:50 MSK. Deadline for the whole project is 23:30 today.

## Read first
`INDEX.md` → `OPS.md` (live status) → `STATE_FOR_REVIEWER.md` (§6 lists weaknesses I already know
about) → `VERIFICATION.md` (source-verified vs relayed claims). Don't re-report things already in
those; I want what is NOT there.

## CLAIMED — do not repeat
- **#1 (loop-gain emergence): TAKEN by subagent 1**, who found and tested a real LR-schedule confound
  in §4.12. Verified independently; §4.12 now carries the cross-run test. Do not redo.
  (Note for calibration: their partial-correlation statistic was collinear and disagreed with mine
  by 0.6; the cross-run matched comparison is what settled it. Prefer designs over statistics.)

## Where I think my understanding is genuinely thin — pick one, go deep, don't survey
**Subagents 2, 3, 4: take DIFFERENT numbered items below, and say which one you took in your first
line so the others can see it.**
1. ~~**Why does loop gain saturate in TOKENS at ~10–15M?**~~ **CLAIMED by subagent 1 — skip.** What remains open on it: I measured *that* it saturates, not *why*; a mechanism or a scaling-law treatment would still be new. (§4.12). I measured it; I have no mechanism.
   Is there a scaling-law treatment of when recurrent depth utility stops improving with data? This
   is the finding I am least able to explain and it is load-bearing for the whole "more loops" story.
2. **Why does no signal predict per-token depth demand?** (§4.7). Four families fail — thresholds,
   buckets, a learned probe, Q-exit at PALBERT spec — while the oracle headroom fails its null
   (nulls give 0.388/0.411 vs real 0.309). My working read is "the variation is unstructured", but I
   cannot tell that apart from "structured in a basis none of my four scalars span". Is there work
   that distinguishes these?
3. **The ~31× gradient-norm gap I measured between tied and untied stacks** (`src/grad_spectrum.py`:
   ‖G‖_F 0.4949 tied vs 15.5617 untied, stable rank 6.73 vs 4.40). I found this by accident, it may
   explain why the untied baseline NaN'd, and I do not know whether it is known.
4. **Is `state_renorm=True` provably a contraction?** §4.3 measures it contracting to a fixed point.
   There is a relayed claim that "Looped Transformers with LayerNorm Provably Learn the Power Method"
   is relevant. I have not obtained it.

5. **Is the report's §4.4 negative (a 33-layer untied stack could not be trained) real, or an MPS
   artifact?** This is the most attackable claim in the report. I am launching a CUDA re-run now, so
   do NOT run compute on it — but if there is literature on training instability of deep untied
   pre-norm stacks at ~80M params that would predict either outcome, that is worth having before the
   result lands.
6. **Does anything in the literature bear on my §4.8 result** — that a mixed-depth KV cache costs
   almost nothing (spread 0.001–0.005 nats for t ≥ 4) while the whole early-exit literature treats
   KV-cache absence as the central problem? Either someone has measured this and I should cite them,
   or the discrepancy is because their models converge and mine does not.

## Rules
- Verify from primary sources; quote them. Say explicitly what you could NOT verify.
- Numbers over prose. If a claim would change something in `report.md`, say which section.
- Write findings to `subagents/from-find-a-thing.md`.
