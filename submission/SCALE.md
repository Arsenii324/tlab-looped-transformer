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

> **⚠ Read this section knowing the CE half of the claim was WITHDRAWN.** *Added 20:03; it was
> missing, and a reader of this document alone would have got a scale argument for a claim whose main
> half no longer stands.* Supervision annealing's **CE advantage over dense supervision was withdrawn
> at n = 4** by a criterion registered before the data existed: seeds 2 and 3 gave **+0.0482** and
> −0.0902, the four-seed mean (−0.0460) sits inside the 0.0541 floor, and the paired t-interval
> **[−0.1478, +0.0558] covers zero.** **What survives is the depth half** — the useful band widens at
> **4 of 4 seeds** (+2.5/+2.5/+2.5/+7.2), including at the seed that reverses the CE claim. So the
> mechanism defended below is one that **relocates the useful band at zero parameter cost and does not
> lower the loss**, and the scale argument should be read as being about that.

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

## 5. The scale question — ANSWERED at 19:47, and the answer is weight tying

*This section previously read "the scale question this project cannot answer". It proposed running the
effective-rank probe on a larger weight-tied checkpoint to decide whether the depth-key collapse is a
property of **tying** or of **smallness**. That question is now settled, by a cleaner route than the
one proposed: rather than varying scale and hoping nothing else moved, **hold scale fixed and vary
tying.***

A token's depth keys span an effective rank of **~1.6 of 32** in this model; the collapse is present
at initialisation and **training makes it worse** (2.73 → 1.83). It is the mechanism under the whole
depth-mixing family (`NEGATIVE_RESULTS.md` §3), so whether it generalises decides how far that
negative reaches.

**Measured. Both models untrained, identical hidden size, heads, head_dim and initialisation, 33
depths each** — so training quality cannot explain the difference and the *only* variable is
tied-vs-untied. **The table carries four rows rather than two, because the two-row version was
measured first and was substantially a confound** `[RANK-PROJECTION]`:

| what is measured | effective rank | mean pairwise cos |
|---|---|---|
| tied — depth **states**, no projection | **1.40 / 33** | +0.8044 |
| untied — depth **states**, no projection | **4.36 / 33** | +0.8815 |
| untied — the same states through **one shared** `W_K` | 4.36 / 33 | +0.8807 |
| tied — **keys** (its one `W_K`) | 2.73 / 33 | +0.8022 |
| **untied — keys, each layer's own `W_K`** | **31.83 / 33** | **+0.0000** |

**Read the rows in that order and the mechanism is visible.** At the level of the *representations*
the two architectures are **3.1×** apart and **both are highly collinear** — the untied stack's states
are, if anything, *more* cosine-correlated (0.88 vs 0.80). The near-orthogonality appears only in the
last row, and it appears **because each untied layer has its own `W_K`**: 33 independent random
projections of even an *identical* state are near-orthogonal in ℝ²²⁴ by construction. Sharing the
projection collapses 31.83 back to 4.36 immediately.

**So the asymmetry is in the projections, not in the representations — and that is the stronger form
of the argument, because depth attention reads keys.** MoD-Attention and its family attend over the
keys the model computes, so key-space rank is the right quantity for explaining their positive, and
31.83 stands there. **Distinct per-layer projections decorrelate a collinear state stream for free. A
weight-tied loop has one `W_K` and cannot buy that at any width.**

> **What this replaces.** An earlier version of this section led with **"11.7×, with smallness held
> fixed by construction"** — the ratio of the two *key* rows alone, which credits weight tying with a
> representation gap that is really a projection artifact. The confound was raised by an external
> reviewer and confirmed by a control run (hold the projection fixed) within six minutes. **The
> conclusion did not change; the mechanism behind it did, and got more defensible.** The superseded
> number is left visible rather than deleted, per this project's rule for retractions.

**So the collapse is weight tying, and the negative generalises.** Three consequences for scaling:

1. **A wider block does not fix it.** The same block applied twice produces near-identical keys
   because that is what weight tying *means*. Mixture-over-depths mechanisms should be expected to
   keep failing in weight-tied looped models **at any size** — this is no longer a statement about
   *small* looped transformers.
2. **The published positives in that family are explained rather than contradicted.** MoD-Attention
   reports +0.2 perplexity and +2.11% downstream at 1.5B on **24 and 48 unshared layers** — precisely
   the near-orthogonal key set measured above. **Their gain is a property of distinct layers**, now
   measured rather than argued.
3. **It names the real cost of weight tying, which this report had not stated — and the claim is
   about keys, not representations.** Tying buys parameter efficiency and **pays for it in the
   distinguishability of the depth *keys* a mixing mechanism actually attends over**. The narrower
   claim is the one the correction above leaves standing, and it is also the sufficient one: an
   unshared stack does **not** need diverse representations for depth attention to work, because its
   diverse projections manufacture a near-orthogonal key set out of a collinear state stream. A tied
   loop has one `W_K`. That is a structural trade, not a tuning problem, and it is the sharpest
   scaling statement this project can make.
   *Not claimed:* that an untied stack builds richer depth **representations**. Measured, it barely
   does — 4.36 vs 1.40 of 33, both collinear.

**Scope, stated:** one width and one depth count. The ratio at other widths is untested — though the
mechanism (identical weights produce identical maps) does not obviously depend on width, and the
untrained comparison removes training quality as an explanation. Reproduce with
`src/depth_key_rank.py::tied_vs_untied`, one forward pass each, no training.


