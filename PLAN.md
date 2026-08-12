# PLAN — looped-transformer pretrain on FineWeb, T-Lab test task

Written 2026-08-13, for a 12h autonomous execution window (user unavailable to respond). Everything
below is a decision I'm making now, not a menu — each entry says what I chose, what I rejected, and
what would flip it. No subagents, no workflows, per explicit instruction: everything here runs as
direct tool calls in this session, sequentially or via backgrounded local processes I monitor myself.

## 0. What's actually being graded, and how that shapes my scope

The spec (`../files/test_task_tlab.txt`) grades two things: idea quality (explicitly the user's,
explicitly *not* meant to come from an LLM), and implementation/verification quality (explicitly
where a coding agent is expected to do the work, with the human staying on top of what it did). I'm
building for the second half, and using the first half only to the extent I can ground candidate
mechanisms in *this workspace's own measured evidence* on Huginn-3.5B rather than in literature
recall — that's execution informed by prior data, not idea generation. Where I pick a design axis
myself I say so and why, so it's separable from whatever the user brings.

**Confirmed vs. not, on the task's own references** (checked now rather than assumed, per standing
instruction to verify a citable claim before leaning on it):
- Ouro = "Scaling Latent Reasoning via Looped Language Models" (arXiv 2510.25741), LoopLM
  architecture, entropy-regularized early exit, 1.4B/2.6B trained at 7.7T tokens. Confirmed.
- "Loop the Loopies!" (arXiv 2607.16051) = Loopie, MoE loops each layer twice, argues looping only
  wins under a *compute-matched* comparison, not parameter-matched. Confirmed, and this is the right
  lens for my own eval too (see §5).
- "Q-exit" — **could not confirm a specific named method.** Nearest confirmed relatives: Ouro's
  entropy-thresholded exit, and a second-order "acceleration" (hidden-state 2nd difference) exit rule
  from a 2025 paper found in the same search. I'm treating "Q-exit" as "a learned per-loop
  confidence/halting signal" generically rather than citing a source I don't actually have.
- Found and worth noting even though unsolicited: "The Readout Blind Spot in Looped Language Models"
  (arXiv 2606.24898) states directly that scale-invariant readouts (RMSNorm/LayerNorm) hide
  hidden-state scale from a dense per-loop loss, so per-loop supervision can train an early-exit
  signal fine while the recurrent state's scale explodes unchecked underneath it — and recommends
  either making scale loss-visible or removing it from the loop. This independently agrees with
  Huginn's own measured behaviour (state confined to a sphere by an explicit per-loop RMSNorm) and
  directly motivates §3's state-renorm axis. Two independently-arrived-at reasons to lean the same
  way is the strongest prior I have for any single design choice here.

## 1. Where this lives, and what it does not touch

`~/build-projs/barannikov-work/tlab-loop-transformer/` — new sibling directory, new git repo, no
GitHub remote added yet (that's the task's actual submission step; adding/pushing to a real remote is
the user's call, not mine to make silently). Nothing in `Geometry-of-Reasoning-Trajectories/` or
elsewhere in `barannikov-work/` is modified. A second Claude session is concurrently working in that
other repo on the paper's report (D203/D204, census-curve correction); coordinated once, no shared
files, no further interaction needed.

## 2. Compute (no Yandex Cloud)

Checked, not assumed:
- **Local: Apple M2 Pro, 12 cores (8P+4E), 32GB RAM, torch 2.13 with MPS.** Raw fp32 matmul benchmark:
  5.6 TFLOPS unfused at 2048x2048. Real small-model throughput will be well below that (small hidden
  dims, attention overhead, MPS kernel-launch cost) — measured directly in Phase 1, not assumed.
- **Kaggle: authenticated CLI, already used extensively by the sibling project (T4, free tier).**
  Not Yandex. Available as fallback.
- **Disk: 41GB free (95% full volume) — the binding local constraint.** FineWeb is pulled via HF
  `datasets` **streaming** only, tokenized on the fly, and packed into local `uint16` token shards;
  raw parquet is never bulk-downloaded. 100-150M tokens packed is ~200-300MB. Confirmed streaming
  works (3 docs fetched in 19s on first call; schema is `text/id/dump/url/date/file_path/language/
  language_score/token_count`, config `sample-10BT`).

**Decision: local MPS is the default compute path for everything, including the full-budget run,
unless the Phase-1 throughput measurement says otherwise.** Rejected: Kaggle-first. Reasoning: Kaggle
GPU-hours are a shared, explicitly-flagged-scarce resource under this user's standing instructions;
spending them should be a visible decision, not a default. **What would flip it:** if measured
tokens/sec on MPS implies the full 100M-token run plus screening would blow past the compute budget
by a wide margin (see §5 for the actual arithmetic once measured), scaled-up runs move to Kaggle,
capped to a stated wall-clock budget per job, logged as a deliberate spend in LOG.md.

## 3. Architecture

Base: Qwen3's decoder block, read from the installed `transformers==4.53.3` reference
(`modeling_qwen3.py`), not from memory — confirmed mechanics: pre-norm RMSNorm, **QK-Norm applied
per-head on head_dim before RoPE** (Qwen3's actual differentiator vs. Qwen2/Llama), GQA
(`num_key_value_heads < num_attention_heads`, repeated), SwiGLU MLP (`down(silu(gate(x)) * up(x))`),
no linear biases anywhere. This becomes the *inner block*; the outer wrapper is the looped/weight-tied
part, which Qwen3 itself doesn't have — that's the modification the task asks for.

**Fixed by the budget, not a design choice:** vocab must be small. A 10M param ceiling cannot carry
Qwen3's real 151,936-token vocab (would alone cost tens of millions of params even at tiny
hidden_dim) or even GPT-2's 50,257 comfortably. **Decision: train a small custom byte-level BPE
tokenizer (target ~4-8k vocab) on a FineWeb sample, locally, with the `tokenizers` library (already
installed).** Rejected: reusing an off-the-shelf tokenizer. What would flip it: if `param_budget.py`
(Phase 1) shows a somewhat larger vocab (e.g. 16k) buys more validation perplexity per param than the
hidden_dim it displaces, in the screen.

**Five ablation axes, each a toggle in `model.py`, chosen because each is grounded in a specific
prior result rather than being a generic hyperparameter:**

1. **BPTT truncation depth: full vs. last-K.** *Highest-confidence bet in this plan.* The sibling
   project's strongest concrete, causal lead on why Huginn saturates after ~10 loops is that its
   training recipe backpropagated through only the last 8 iterations — a live code path
   (`torch.no_grad()` around the early loops), not a training note — and 8 sits right at the reported
   knee, while a controlled toy/real-architecture check there found early loops are *not* starved of
   gradient by contraction itself (gradient mass roughly uniform across loops when nothing truncates
   it). At 10M params, full backprop through even 64 loops is affordable on this hardware (activation
   memory for a model this size at reasonable batch/seqlen is tiny). **Default: full BPTT.**
   Truncated-K is an explicit ablation arm reproducing Huginn's regime for direct comparison — this
   is the one experiment Huginn's own team couldn't run affordably and this task can.
2. **Explicit state renormalization each loop: on vs. off.** On Huginn, an explicit per-loop RMSNorm
   confines the recurrent state to a sphere (measured, not architectural necessity — it's a choice
   Huginn made), and the state's dynamics are then measurably rotation-dominated rather than settling
   in place. The independently-found Readout Blind Spot paper (above) gives a second, unrelated
   reason to expect this matters: without it, dense per-loop supervision can train the readout fine
   while raw state scale explodes unboundedly underneath, invisible to a scale-invariant loss.
   **Default: on** (explicit RMSNorm of the full state before re-injection each loop). Off is the
   contrasting arm (standard pre-norm residual, where only sublayer *inputs* are normed, not the
   carried state itself).
3. **Input re-injection per loop: none / additive / concat+adapter.** Huginn re-injects the prelude
   output every iteration via concat+adapter; that costs adapter params I may not be able to afford
   at 10M. Default: **additive** (cheap, still gives every loop fresh access to the input) as the
   center config; concat+adapter and none are ablation arms.
4. **Depth-aware initialization: on vs. off.** Huginn scales init std by width and output projections
   by an expected-loop-count-aware factor, specifically so that many-loop training doesn't blow up or
   vanish from init alone. Cheap to include, plausibly load-bearing for "stays useful at high loop
   count" specifically. Default: on.
5. **Train-time loop-count randomization: fixed vs. sampled range.** Huginn already does this
   (Poisson-lognormal, mean 33) and *still* saturates — so this alone is confirmed **not sufficient**,
   but removing it entirely could still make things worse (a model that only ever sees exactly R loops
   at train time has no reason to stay coherent at R+1). Default: on, sampled log-uniform up to a max
   comfortably above the eval sweep's top end.

**Deliberately deferred, not core-required:** an early-exit / halting head (the task's own wording:
"can also implement," optional). Scaffolded as a hook in `model.py` (a per-loop scalar head is cheap
to add later) but not trained in the first pass — the core ask (low perplexity *via* many useful
loops) doesn't need it, and building it before knowing whether loops 20-64 are even worth exiting
*into* is premature. **What would flip it:** if screening shows a clear falloff point R* well inside
the range I can afford to train past, implementing the exit becomes worth it both for its own sake and
for the report's required "why does this scale" argument (adaptive compute is exactly the kind of
mechanism that doesn't die at scale, unlike e.g. a large fixed value-embedding table — the task's own
example of what NOT to lean on).

## 4. Data and eval

FineWeb `sample-10BT`, streamed, custom BPE tokenizer, packed to `uint16` shards, disjoint train/val
document sets (split before tokenization, not after, so val is never seen). Token budget: ~90M train
+ ~5-10M held-out val, comfortably under the 100M training cap with room for the val set to not eat
into it.

**Primary instrument: teacher-forced per-loop validation cross-entropy, loop count swept 1..R_max at
eval time regardless of what R the run trained at** (cheap — one extra axis on an eval forward pass,
no separate runs needed). This deliberately avoids the trap the sibling project hit: that was a
*generation*-based, chat-templated, discourse-confounded readout (gold-token rank at the first
generated position, which tracked whether the model was about to write a filler word, not whether it
knew the answer). A raw next-token teacher-forced loss on plain web text has no chat template and no
"is it about to write a stock opener" failure mode to inherit — but the general lesson (know what your
per-loop instrument actually reads before trusting its shape) still applies, so:

- **Also logged every run, not just the interesting ones:** per-loop hidden-state L2 norm (catches
  the Readout Blind Spot paper's scale-explosion failure, which a scale-invariant loss alone would
  hide); per-loop predictive entropy (catches collapse-to-unigram-prior, a different degenerate
  "flat loss curve" than genuine saturation); an online contraction-rate estimate via two-trajectory
  convergence on a fixed prompt pair (this project's own D44 method, state-level, doesn't touch the
  readout at all).
- **Report the perplexity-vs-loop curve compute-matched, not just parameter-matched**, per Loopie's
  explicit framing — a small model looped 32 times spends ~32x the FLOPs/token of the same model run
  once, and the honest comparison is against a same-compute non-looped or shallow-looped baseline, not
  only against loop-count-1 of itself.

## 5. Ablation strategy and rough compute budget

Not a full 2x2x3x2x2 = 48-arm factorial — too expensive and most of those cells are uninformative.
**Screen at a reduced token budget (~8-15M tokens/arm) one-axis-at-a-time from a "center" default
config** (full BPTT, state-renorm on, additive injection, depth-init on, loop-randomization on),
varying axis 1 (truncation) and axis 2 (state-renorm) first since those carry the strongest priors and
the most riding on them, then 3-5 more cheaply. Promising configs (by the per-loop val-CE curve, not
a single number) get scaled toward the full ~90M-token budget; at most 2-3 configs reach full budget,
chosen and logged with reasoning, not run because they were next in a queue.

Rough compute check (refined empirically in Phase 1, this is the planning-time estimate): a ~10M-param
block looped ~16-32 times costs roughly `6 * loops * block_params` FLOPs/token; at block_params~1-2M
and loops~24, that's ~3-6e8 FLOPs/token, so 90M tokens ~2.7-5.4e16 FLOPs total for one full-budget run.
Against a conservatively-assumed *effective* (not peak) MPS throughput of 0.3-1.5 TFLOPS for a model
this small, that's roughly 5-50k seconds (1.5-14h) for **one** full run — wide enough that the Phase-1
measurement, not this estimate, decides whether full-budget runs stay local or move to Kaggle, and how
many configs can afford to reach full budget inside the remaining window.

## 6. What I will not do without the user

Push to any GitHub remote or Hugging Face Hub (the task's literal submission targets — his call).
Spend Kaggle GPU-hours beyond a small, explicitly logged, capped job if local proves insufficient.
Delete or modify anything outside this directory. Pick "the" final answer to report's idea-generation
section — that stays reserved for him.

## 7. Phases (driven by actual completion, not a clock — 12h is a budget, not a schedule I can watch)

0. Recon — done (this document).
1. Repo scaffold: `param_budget.py` (search hidden_dim/n_heads/vocab grid to hit ~10M), tokenizer
   training, `data.py` (stream -> tokenize -> pack, train/val split), throughput micro-benchmark on
   the actual model shape (not generic matmul) to settle §2's compute-path question empirically.
2. `model.py` (Qwen3-derived looped block, all five toggles), unit tests (param count matches budget;
   full-BPTT gradient reaches loop 1; state-renorm toggle actually bounds norm; each injection mode
   shape-checks).
3. `train.py` + `eval.py`; tiny CPU smoke run, then MPS, comparing a few forward passes for numerical
   parity; checkpoint save/load round-trip verified byte-for-byte on immediate reload.
4. Screening sweep per §5, local, background-run and monitored, LOG.md updated per arm as it finishes.
5. Full-budget run(s) of the winner(s); local unless §2's flip condition fires.
6. Best checkpoint saved locally (not yet pushed); `report.md` populated with measured sections and a
   clearly separate reserved section for the user's own idea narrative; README with exact repro
   commands.

Stretch, only if time remains after 6 with margin: early-exit head; eval-time loop sweep well past
the trained max (does it degrade gracefully or catastrophically past R_train — informative either
way); a larger-vocab ablation arm.
