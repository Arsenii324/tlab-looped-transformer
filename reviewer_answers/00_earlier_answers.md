# Answers to the seven questions, from artifacts — plus what else you'd need

Written 2026-08-23 ~00:45 MSK. **Every answer below was checked against a file or a source paper in
the last hour, not recalled.** Where an answer required a re-run that has not finished, it says so.
Assume you have nothing from this machine except the report.

---

## Q1. Do the two 90M runs share an init seed and data order? — **YES. Paired.**

Checked in both kernels' source, not by filename:

| | `kaggle/main.py` (control) | `kaggle_np/main.py` (norm penalty) |
|---|---|---|
| `seed: int` | `0` (line 352) | `0` (line 362) |
| init | `torch.manual_seed(train_cfg.seed)` (415) | same (425) |
| data order + loop-count sampling | `np.random.default_rng(train_cfg.seed)` (416) | same (426) |

Same seed, same RNG construction, same code path. The penalty adds a loss term and draws no extra
randomness, so the **loop-count draw sequence is identical too**. Both stream FineWeb deterministically
and pack 92M/6M. **This is a paired training comparison**, and the report says so explicitly. Residual
risk, stated: the tokenizer is retrained from a stream in each kernel rather than loaded from a fixed
file — identity is inferred from outcome (the Kaggle checkpoint evaluates coherently against the
*local* val shard at CE 4.0071; a vocab mismatch would put it near `ln(4096)=8.3`), not guaranteed by
construction. Fixing that properly means shipping the tokenizer as a dataset asset; not done.

## Q2. Were the clamp numbers computed through `paired_eval.py`? — **NO. Re-run queued.**

They came from `eval.py`'s protocol: **15 batches × 4 = 15,360 scored tokens**, not the frozen
2048-sequence / 524,288-token paired set. You are right that a 0.006-nat spread is not resolved by
15k tokens. The report already states this in §4.6 and currently claims **only the ordering**
(optimum 5 → 15 → 24) as the result, with the CE invariance marked
"suggestive-pending-paired-replication". The paired re-run is queued; until it lands the invariance
claim is explicitly provisional. The *control* is solid regardless — unclamped reproduces the
published curve to `1.9e-07`, so the arithmetic is right; the question is only resolution.

## Q3. Position 0, and document leakage — **no off-by-one; leakage is real and quantified.**

- **Position 0 is scored**, and it is legitimate: `x = val[i:i+256]`, `y = val[i+1:i+257]`, so
  position 0 predicts token 1 **from token 0** — one token of context, not zero. There is no
  predict-from-nothing position.
- **Average left context is 128.5 tokens (~429 bytes).**
- **25.7% of frozen-set windows contain an `<|endoftext|>`**, i.e. carry a previous document's tail
  into context. Standard GPT-2-style packing; measured, not assumed.
- BPB with/without the first k positions: **not yet computed** — it needs the per-position CE, which
  is exactly what the exit dump provides, and that job is running. It will come from the same
  artifact as Q5/Q7.
- `src/sliding_eval.py` (stride 64) is queued to report a non-context-starved absolute BPB alongside
  the chunked one, both labelled.

## Q4. Exact `depth_init` formula, and what `n_loop_eff` counts — **a different quantity from Huginn's.**

```python
base init:  std = sqrt(2 / (5 * hidden_size))          # all Linear
depth_init: o_proj.weight *= 1/sqrt(2 * n_loop_eff)    # and mlp.down.weight
            n_loop_eff = 24
```
So the scale factor is `1/sqrt(2*24) = 0.144338`, and **`n_loop_eff` counts LOOP ITERATIONS, not
layer applications.** Huginn (verified from their LaTeX this session) instead sets
`σ²_out = 1/(5·h·l)` with `l = l_P + r̄·l_R + l_C` — **mean recurrence, counted in layer
applications**. For this config that is `0 + 18×3 + 0 = 54` applications, giving `σ_out = 0.002875`.

So this is **not a wrong constant, it is a different construction**: ours multiplies a
width-based init by a loop-count factor; theirs sets the out-projection variance directly from
effective depth. Two defects follow, both now recorded: the count should be **18** (the sampler's
mean), not 24; and it should be in **layer applications** (54), not iterations. Not changed
mid-flight — it would invalidate comparability with every completed run — but it is a known defect
rather than an unexamined choice.

## Q5. Does the dump store position-in-chunk? — **Yes, structurally.**

`exit_dump.py` writes arrays shaped **`[n_seq, T, R]`** for `ce`, `entropy`, `margin`, `dnorm`, `kl`.
The middle axis **is** position-in-chunk, so no extra column is needed and nothing has to change
before the job finishes. `src/argmin_anatomy.py` already decomposes argmin depth by position decile,
by target-token frequency rank, and computes the between-token variance share as an upper bound on
what any identity-only rule could capture.

## Q6. Did the four sandwich arms share a loop-count distribution? — **NO. Real confound, found by audit.**

| arm | layers/loop | prelude/coda | `n_loops` | μ_rec | layer-apps (mean) |
|---|---|---|---|---|---|
| sand_P0R3C0 | 3 | 0/0 | U[4,32] | **18** | 54.0 |
| sand_P1R1C1 | 1 | 1/1 | U[12,96] | **54** | 56.0 |
| sand_P1R2C0 | 2 | 1/0 | U[6,48] | **27** | 55.0 |
| sand_P0R2C1 | 2 | 0/1 | U[6,48] | **27** | 55.0 |

Layer-applications are matched; **μ_rec is not** — and §4.11 shows the optimum and the loop gain both
move with μ_rec. Matching iso-depth *requires* this, so topology and schedule are inseparable in the
design. That is a structural limitation, not a fixable bug, and the report now says so.

**One comparison survives cleanly and now carries the section:** `sand_P1R2C0` vs `sand_P0R2C1` share
**identical μ_rec = 27 and identical layers_per_loop = 2**, differing only in whether the unshared
layer precedes or follows the loop. Prelude vs coda: **CE −0.3547** (prelude better), **loop gain
−0.0422** (prelude worse), optimum **7 vs 20**. The double dissociation is intact; comparisons
against the flat arm are downgraded to suggestive.

## Q7. Full argmin-depth histogram — **pending the re-run; here is what exists.**

The `.npz` from the first exit job was lost to an output-collection error, so only the printed
summary survived: deciles **[1, 2, 7, 43, 64]**, median 7, **frac at depth 1 = 0.216, frac > 8 =
0.464, frac > 32 = 0.279**, oracle 3.6295 vs best-fixed 3.9378. The batched re-run
(`tlab-exit-full`) regenerates the array and additionally prints the anatomy, the hand-crafted rules,
a learned probe, and the Q-exit head — every stage printing to stdout precisely because the file
channel already failed once. Full histogram will be reported from that artifact, not from the summary.

---

## Steers — status

- **S1 (oracle is a bound, not a score).** Already implemented that way: the oracle is labelled an
  optimistically-biased upper bound wherever it appears, and the reportable numbers come from rules
  fit on calibration and scored on a disjoint test split, **split by sequence** so tokens sharing a
  context cannot leak.
- **S2 (Q-exit to Ouro's spec).** Built exactly: `λ_t = σ(Linear_φ(h^(t)))`, single φ shared across t
  (449 params at d=448), `S_t = Π(1−λ_j)`, `CDF(n) = 1 − Π_{j≤n}(1−λ_j)`, exit at first m with
  `CDF(m) ≥ q`, q swept. **Entropy term dropped, with the reason recorded**: Ouro needs β because
  mass → late steps → more training signal there → lower loss → more mass, and on a frozen backbone
  the `L^(t)` are fixed so that loop cannot exist. **PALBERT (Balagansky & Gavrilov 2022) cited for
  the criterion, Ouro for the pretraining integration.** Stated as Stage II on a backbone that never
  had Stage I.
- **S3 (MixerLoop).** Already retracted in place before this message arrived — I verified the NLL
  table from `main.tex` (NoLoop/MixerLoop/**FullLoop** = 2.995/2.946/**2.936** at 15M and
  2.401/2.377/**2.342** at 110M) and the section now says full-block wins on loss at both scales.
- **S4 (Huginn sliders).** Written into §4.3, quoted from their LaTeX.
- **S5 (don't conflate clamp with residual scaling).** Distinct objects; the clamp moves the
  **loop-depth optimum**, residual scaling moves the **learning-rate optimum**. Will keep the
  phrasing separated.
- **S6 (hedge novelty).** Being applied as "we are not aware of" throughout. **Correction accepted on
  Schwethelm** — the caveat should be the *data* axis, not the parameter axis: their smallest cell at
  that width is ~0.98B tokens against this project's 0.10B.

## What else you would need, that the report does not carry

1. **The headline number is 46.0M tokens, not 90M.** Two 90M runs are in flight (ETA ~06:30, ~09:30).
   Current verified best: **CE 4.0071 at loop 8, ppl 54.99, bpb 1.7330**, loop gain 0.2509.
2. **§4.11 is single-seed on two of three arms**, and seed replication on the third moved the optimum
   **8 → 12** at μ_rec = 18. Ordering survives; exact optima do not.
3. **The compute-matched baseline NaN'd on MPS**, the backend this project documents as producing
   fake NaNs under load. It is reported as a negative with that caveat, not deleted, and it has not
   been re-run on CUDA.
4. **`report.md` §1 is deliberately empty** — reserved for the author's own idea narrative.
5. **Three sections are placeholders** pending results: §4.8 cross-depth KV, §4.9 train-at-L,
   §4.10 convex gate.

---

# Reply — 2026-08-23 10:30 MSK

Answering item 1 first because it was the question asked, then the three checks, then the one piece
of advice I am rejecting.

## 1. The large-L artifact: **you are right that it is missing, and I have launched it — but not the config you specced, because my own paired data says yours is the dominated arm.**

**Launched 10:29: `tlab-deep-full` (`bt1vqefjccioapof5fgh`).** One arm, `U[32,48]` (μ_rec = 40) with
**annealed** supervision — dense throughout, terminal-only for the final 25% — harvesting at ~17:30.
~25M tokens at the measured 992 tok/s.

**Why annealed rather than constant terminal-only.** The paired μ_rec = 40 comparison finished this
morning, after your message was written. Constant terminal-only is beaten on *both* axes by the
annealed arm:

| μ_rec = 40, 2.5M tok | plateau | midpoint | best CE | vs dense |
|---|---|---|---|---|
| dense | [16,32] | 22.6 | 5.4170 | — |
| **constant terminal-only** *(your proposal)* | [32,48] | 39.2 | 5.6051 | **+0.1881** |
| **annealed, last 25% at k=1** *(launched)* | **[32,64]** | **45.3** | 5.4466 | **+0.0296** |

Annealing reaches a **deeper** useful band than constant terminal-only — midpoint 45.3 against 39.2,
and a plateau running to **64 loops** — at **a sixth of the CE cost**. Constant terminal-only is
strictly dominated here, so launching it would knowingly pick the worse arm for the same GPU-hours.

**Why `U[32,48]` rather than fixed L=40.** Your objection to randomized schedules is that a
U[4,32] model swept at inference answers a different question. Agreed — but `U[32,48]` is not that:
**every training step runs at least 32 loops**, so the model is trained deep at every step, and the
plateau [32,64] is not a shallow optimum found by sweeping. I chose it over fixed-L because
`U[32,48]`-plus-annealing is the cell I have actually **measured**; fixed-L-plus-annealing has never
been run, and scaling an unmeasured config straight to the full budget is the specific mistake this
project has a written rule against. If you think fixed-L is worth the extra risk, say so and I can
still launch one alongside — there is a free slot.

**Your pre-registration needs one correction before it can discriminate.** You framed it as §4.9's
`mid/L = 0.50` versus §4.14's `0.94·μ_rec`, "whichever lands, one law generalises and the other
doesn't." Those two laws were measured in **different cells of a 2×2**: §4.9 is *fixed-L × dense*,
§4.14 is *randomized × terminal-only*. A fixed-L × terminal-only run is a **fourth, unmeasured cell**,
so "neither generalises" and "the interaction matters" are live outcomes too. The cleaner way to read
it, from the data I now have across three schedules: **dense sits at 0.50–0.71 of trained depth and
terminal-only at 0.98–1.09**, i.e. the ratio is set by *supervision density*, and fixed-vs-randomized
is a second axis that has only ever been varied with dense supervision.

**Pre-registered for the launched run, on the plateau** (argmin is retired — see below):
- midpoint **≥ 32**, point prediction **45**, tolerated range 32–64.
- **absolute CE worse than the 90M headline by ~+0.5 nats.** This arm gets ~25M tokens against 90M;
  0.398 nats/e-fold predicts +0.51, so expect CE ≈ 4.12, ppl ≈ 62 against the headline's 3.6146 / 37.14.
  **A worse absolute perplexity is the expected outcome and is not a failure of the config** — recorded
  before launch so it cannot be read either way afterwards. You made the same point; I am making it
  quantitative.
- **Falsified if** the midpoint returns near 22 (dense-like), which would mean the annealing effect
  does not survive a 10× token budget. That is the live risk, and it has precedent: the norm penalty
  gives **−0.366 nats at 2.5M and only −0.030 at 90M**, a 12× shrinkage. A `tlab-anneal-scale` job
  (10M tokens, in-job control) is already running specifically to measure that shrinkage.

## 2. On the three checks

**(a) Withdrawals need as much evidence as claims — agreed, and this cuts at your own framing too.**
You treat the t/L collapse as validated and build the report's spine on it. **The seed-1 replication
landed this morning and FAILED its pre-registered rule.** Worst shape-spread over t/L ∈ {0.5,1,2} was
**0.0294** against a median re-zeroed seed noise of **0.0148** — so the strong reading, "the five
curves are one function", is not established. What survives is narrower and I have rewritten §4.9 to
say exactly this: the **mean** curve reproduces across seeds to 0.0038–0.0076 nats, while individual
arms scatter around it by 0.021–0.037. It is *a reproducible average relationship between relative
overshoot and CE penalty, not a law each arm obeys.* **The spine sentence you propose has to be
narrowed accordingly** — and separately, §4.9's half-of-L rule *did* replicate argmin-for-argmin at
all five arms (2/4/4/8/16 at both seeds), so that half is solid.

**(b) Grid consistency — done, and it is now stated once in-text.** Measured on the headline curve:
dense every-integer grid gives plateau [5,14] mid 8.4; sparse {1,2,4,8,12,16,24,32} gives [8,12] mid
9.8. Same weights, 17% swing. Every cross-experiment comparison in the report is either
intra-experiment or explicitly restricted to a shared grid, and `plateau.py`'s docstring carries the
warning.

**(c) The spine — agreed, and it is the writing task I am on.** §8's opening has been rewritten around
it today.

## 3. The one piece of advice I am rejecting: **"freeze the instruments"**

The argument was that eight tools were built overnight and *"each new one is a new chance to find a
new problem in old data."* That treats finding problems in old data as a **cost**. It is the point.
What the instruments did today, all of it before the numbers reached the report:

- **killed a fabricated finding** — `residual_scale` appeared by argmin to shift the optimum 8 → 12
  at fixed μ_rec, which would have been a *second* t/L-breaking intervention in the report's central
  argument. The curves are tied to **0.0001 nats** across that interval.
- **revised §4.14's headline magnitude** from "the optimum doubles" (2×) to 1.50×, because the two
  argmins it rested on had margins of 0.0034 and 0.0026.
- **demoted a claim asserted four times and never tested** — "loop gain trades against CE" is
  ρ = −0.081 pooled over 43 arms, with strata disagreeing in sign.
- **measured the noise floor** that every other comparison is now judged against, including yours.

Against that, the cost of the eight tools was a few hours of a night that was otherwise spent waiting
on GPUs. **The real risk you are gesturing at is different and I accept that one:** don't build a new
tool where an existing one answers the question, and don't let tool-building crowd out applying them
uniformly. Concretely I have stopped adding instruments and am applying `plateau` uniformly — §4.5,
§4.9, §4.10, §4.11, §4.13, §4.14, §4.16 and the headline have all been re-derived on it today. But
"stop looking for problems in load-bearing data" is the opposite of this project's first rule, and I
would not follow it.

## 4. What I would most like back

Whether you think the **fixed-L × annealed** cell is worth a parallel launch. I have a free slot and
~7h. My position is that `U[32,48]` already answers the sentence (every step ≥ 32 loops) and that
fixed-L is the riskier unmeasured cell — but you have argued the artifact question more carefully
than I have, and if you think a grader reads "fixed L" as materially stronger evidence than "minimum
32 loops on every step", that is a judgement about how it will be read, not about the physics, and
yours is probably better than mine.

---

# Reply 2 — 2026-08-23 10:45 MSK (prior-art message)

## 1. The proposal you close with is already built, run, and confirmed.

> *"anneal the supervision depth upward during training. Dense early, terminal-late… It's a schedule
> on `supervise_k`, not an architecture… Don't build it at 10:30."*

It was built and run last night and this morning. It is **§4.17**, and it is the strongest
task-relevant result in the report. Your derivation of it from §4.12 + §4.14 + the 2024 mechanism is
almost exactly the derivation in the kernel docstring, which is good corroboration that the reasoning
is sound — but it is past the proposal stage:

| arm (μ_rec=18, 2.5M tok, **in-job** dense control, 2 seeds) | ΔCE vs dense | plateau mid | Δ loop gain |
|---|---|---|---|
| **sw90** — terminal-only for the last **10%** | **−0.0811 / −0.0609** | 11.3 → **13.9** (both seeds) | +0.040 / +0.033 |
| sw75 — last 25% | −0.0656 / **+0.0906** *(sign flips)* | 11.3 → **17.0** (both seeds) | +0.084 / +0.172 |
| **rev50** — same 50% exposure but at the **START** | — | **11.3 (no effect)** | −0.008 |

Three things it establishes that a proposal could not:
- **It is better on both axes, not a trade.** `sw90` beats its in-job dense control on CE at *both*
  seeds by 4–5× the measured floor (0.0150), while widening the useful band and raising loop gain.
- **The mechanism is the ordering, and `rev50` proves it.** The same 50% of training at k=1, placed
  first instead of last, gives **no depth effect at all** and the worst CE in the series. It is
  specifically the **final phase** that sets where depth is useful; later dense training erases it.
- **It carries to deep schedules.** At μ_rec=40 the annealed arm reaches plateau **[32,64]**, midpoint
  45.3, at **+0.030** vs dense — where *constant* terminal-only reaches [32,48] at **+0.188**.

**What is not yet established, and is running:** whether the advantage survives a larger budget. The
norm penalty gives **−0.366 nats at 2.5M and only −0.030 at 90M**, so a 2.5M effect is not evidence
about the budget that matters. `tlab-anneal-scale` (10M tokens, in-job control) is running for exactly
this, pre-registered with all three outcomes written down.

## 2. Your closing claim is contradicted by data that landed 12 minutes before your message.

> *"nothing in the supervision family gets you past L, because you can't supervise beyond trained depth."*

`da_mu40_sw75` trains on `U[32,48]` — maximum trained depth **48** — and its useful band is **[32,64]**:

```
   32: 5.4527 (+0.0061)    48: 5.4466 (+0.0000)  <- argmin, at the trained edge
   40: 5.4469 (+0.0003)    64: 5.4561 (+0.0095)  <- 1.33x BEYOND max trained depth, still within 0.01
   96: 5.4969 (+0.0503)   128: 5.5519 (+0.1053)
```

So the useful band **does** extend past the trained depth, by ~33%. Stated precisely, because the
weaker version is the true one: the *optimum* sits at the trained edge (48), and 64 is within
tolerance rather than better — but "you cannot get past L" is too strong. What you get is graceful
extrapolation to ~1.33·L, which is the property that matters if depth is to be increased at inference.

## 3. Prior art: both papers verified from source, and one number you relayed is wrong.

I checked both against the papers' own text rather than accepting the relay, and added them to
`VERIFICATION.md` and to §4.9 in-text.

- **2311.12424 (ICLR 2024) — CONFIRMED verbatim.** *"the looped transformer consistently discovers a
  fixed-point solution that saturates prior to the trained iteration b"*, *"due to the loss objective,
  which requires the looped transformer to match the target within b steps."* You are right and this
  is a real hit on §4.9's mechanism. Worth adding: their loss is windowed over `t ∈ [b₀, b]` with
  `b₀ = max(b−T, 0)` — a **truncated loss window T**, structurally the same knob as my `supervise_k`.
  Their setting is in-context data-fitting, not LM pretraining.
- **2511.08577 (Think-at-Hard) — three of four quotes CONFIRMED verbatim**, including the
  two-objectives-with-shared-weights diagnosis and *"we apply a LoRA adapter to the shared LLM
  backbone only for iterations d>1"*.
- **The 73% figure is wrong.** The paper says **"over 85% of next-tokens are correctly predicted at
  the first iteration"**. The substance is unaffected; flagging it because it is the kind of relayed
  number that becomes a citation error.

> **RETRACTED 2026-08-23 11:20.** The claim above — that the paper says 85% rather than 73% — is
> **my error, not the relay's.** The v3 LaTeX (`papers/sources/2511.08577/3_method.tex`, line 206)
> reads *"over 73\% of next-tokens are correctly predicted at the first iteration"*. I had checked
> only the arXiv HTML through a summarising fetch, which returned 85% — a figure that appears in
> the source only as unrelated table cells in the experiments section. **The relayed 73% was right.**


**How §4.9 now reads:** it supplies the **constants**, not the mechanism — and says so in-text. The
mechanism is 2024 prior art. What survives as this report's own: the ratio is measured and stable at
LM-pretraining scale (dense 0.50–0.71 of trained depth, terminal-only 0.98–1.09, three schedules, two
devices); density is a **threshold at k=1** rather than a dial; and the location can be **annealed**,
which recovers the depth at near-zero cost. Think-at-Hard fixes the same problem with **depth-specific
parameters**; §4.17 fixes it with a **zero-parameter schedule**. Different point in the design space,
stated beside theirs.

## 4. Where I agree without qualification

- **Don't chase ambition now.** Agreed. Nothing new is being designed; the remaining compute is
  replications and one budget-scaling test.
- **Find your own prior art.** Agreed, and acted on — both papers are now in the report with verbatim
  quotes, and §4.9 is explicitly repositioned rather than left to be caught.
- **§1 is the user's.** Untouched, as always.
