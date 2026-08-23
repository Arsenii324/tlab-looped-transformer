# Why this should keep working at a larger scale — and the one joint where it might not

*The brief asks to «обосновать, почему ваш метод будет работать хорошо и на большем скейле», and gives
the failure mode it has in mind: a large trainable value-embedding table, whose benefit is a fixed
lookup that stops mattering as parameters grow. This document audits the method against that test,
and then against the objection the literature raises instead — which is different and harder.*

---

## 1. The recommended mechanism costs zero parameters

**Supervision annealing** — dense supervision for most of training, terminal-only for the last ~10%
of steps — is a schedule on *which loop indices receive gradient*. It is a property of the **loss**,
not of the model.

- **No table to outgrow.** There is no fixed-size lookup whose share shrinks as the model widens, and
  nothing whose contribution is diluted by width.
- **No extra FLOPs.** It supervises *fewer* loops, not more.
- **Nothing is undefined outside its trained range.** Contrast an iteration-embedding table, which
  caps the loop count architecturally and is undefined at `t` beyond its rows — so "evaluate at 64
  loops after training at 32" is not merely worse but impossible.

Under the brief's own criterion this is a **strictly stronger** position than the alternatives: the
one scale-legitimate conditioning mechanism found in the literature (IterAdaLN, a *function* of `t`
rather than a table over it) still pays ~344k parameters, ~4.8% of this budget.

**And the architecture itself adds no capacity decoupled from the reused block's width.** The block is
~80% of the budget and scales the ordinary way; the tied vocabulary embedding scales *linearly* in
width, so its share **shrinks** as the block widens — the opposite of the failure mode being guarded
against; `h₀` and the norm weights are O(width), the same class as any bias.

## 2. The weak joint, stated plainly

**The rule is validated as a *fraction* and the mechanism is keyed to *tokens*.** Annealing was
validated at 2.5M tokens, where "switch at 90% of steps" lands at 2.25M — **before loop gain has
emerged at all** (it emerges over ~10–15M tokens). Extrapolated to 90M, the same fraction switches at
81M, **deep post-saturation**. Those are not the same intervention.

A token-keyed form — *dense until loop gain flattens, terminal-only thereafter* — would put the switch
at ~10–15M in any budget, which at 90M means terminal-only for the last **~83%** of training. That is
a different recipe, not a reparameterisation. **One 10M in-job comparison favours token-keying by
0.22 nats — at n = 1**, against a measured seed spread of 0.0640 on this class of paired difference.
It is a **lead, not a recommendation**, and it is the single place where the scale argument defends
the *mechanism* and not the *parameterisation*.

## 3. The objection the literature raises, which is not the brief's

**This model's capacity is dense.** The reused block is dense attention + dense SwiGLU, and the
argument that looping buys FLOPs-per-parameter is made *against a dense baseline* — which is not the
frontier alternative. If sparse (MoE-style) capacity scales better, then "loop a dense block many
times" could be the right answer to the wrong comparison.

**Nothing in this project tests that** — no sparse arm, no budget for one. What can be said:

- **MoE's value proposition is decoupling *stored* from *active* parameters, and this task caps
  *stored* parameters.** The arithmetic is not close: this block's MLP is `3·448·1344 = 1,806,336`
  parameters, and splitting it into four experts of `I/4` gives `4·3·448·336 = 1,806,336` — **exactly
  the same total.** It subdivides existing capacity and adds routing noise at a granularity where the
  noise plausibly dominates.
- **What subdivision *would* buy, stated so the exclusion is not attackable for omitting it:** top-k
  of E cuts per-loop FFN FLOPs by ~k/E, and the FFN is ~68% of per-loop cost, so top-1-of-4 is ≈0.49×
  per loop ≈ **2× more loops at the same wall clock**. Wall clock is the binding constraint on loop
  count here. Real benefit, on the axis the task cares about — and it does not change the conclusion,
  because the parameter cap is what MoE trades against.
- **The evidence does not bear on the axis this task is about.** Every paper in that family
  demonstrates at **2–4 loops**. Their claim is about the parameter/compute scaling *curve*, not about
  loops paying at r = 32.

**The distinction that matters: density threatens the architecture's *competitiveness*, not the
finding's *transferability*.** Annealing is orthogonal to the dense/sparse axis — it is a schedule on
gradient, and it would apply unchanged to a sparse block. Only the second is what this method claims.

## 4. What scaling would actually buy, measured rather than assumed

**Finishing the token budget dominates every architectural intervention measured here.** 46M → 90M
tokens bought **0.39–0.42 nats**; every mechanism in this report is worth 0.002–0.19.

The scaling constant has two anchors and should be quoted as a **range, 0.40–0.52 nats/e-fold**, not
a point. The lower anchor crosses a device *and* a validation-shard boundary; the cleaner pair
(46→90M, same platform, same shard, same protocol) gives the **higher** figure. **No conclusion in
this report changes anywhere in that interval** — re-derived at both ends.

**Against external numbers, the honest arithmetic.** At 90M tokens and 9.06M parameters this model
sits at **D/N ≈ 9.9**, against a Chinchilla-optimal ratio near 20 — trained at roughly *half* the
tokens its size wants, because the task caps the budget. Published entries this would be compared
against are **data-unconstrained** and train on the order of **~7B tokens, roughly 70× this budget**.
Quoting a bits/byte gap without that ratio attached misrepresents both numbers.

## 5. The scale question this project cannot answer, and it is the interesting one

**The depth-key rank collapse may or may not be scale-dependent, and nothing here tests it.**

A token's 32 depth keys span an effective rank of **~1.6**; the collapse is present at initialisation
and **training makes it worse** (2.73 → 1.83). This is the mechanism under the whole depth-mixing
family (`NEGATIVE_RESULTS.md` §3).

**If it is a property of weight tying, it does not improve with scale** — the same block applied twice
produces near-identical keys because that is what weight tying *means*, and a wider block does not
change it. On that reading, mixture-over-depths mechanisms that work in **unshared** stacks (where
each layer's keys are a genuinely different function) should be expected to keep failing in looped
models at any size, and the published positives in that family are a property of *distinct layers*.

**If instead it is a small-model artifact** — 448 hidden units, 4 heads, 112 head-dim, and only 3
distinct layers to differentiate — then a wider block might carry genuinely distinguishable depth
keys and the family reopens.

**This project cannot distinguish those**, and it is the single measurement I would most want next:
the same effective-rank probe (`src/depth_key_rank.py`, one forward pass, no training) run on a
larger weight-tied looped model. It costs almost nothing on an existing checkpoint and it decides
whether the strongest negative here is a statement about looped transformers or about *small* looped
transformers.
