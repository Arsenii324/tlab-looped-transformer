# T-Lab looped-transformer: what I built, what it showed, what you need to know

Written for you to read before doing the task yourself. Dense on purpose. Every number here was
re-verified from raw result files on 2026-08-22, not recalled.

---

## 1. Bottom line, three sentences

1. The platform works and is verified: tokenizer → data → model → train → eval, Qwen3 block matched
   to the reference implementation to 2.4e-7, 5 correctness tests, chunked/resumable training,
   46M tokens trained on the best config.
2. The strongest *measured* result: turning **state renormalization off** is worth ~0.75 nats at
   screening scale and holds at full scale — by far the largest effect of the five axes I ablated.
3. **The task's actual objective is not met.** Loop benefit saturates at ~8–12 loops, exactly the
   regime the task calls out as the problem (Huginn ~10). More training bought lower absolute loss
   but *not* more useful loop depth. Details in §3 — this is the most important thing in this doc.

---

## 2. What the task rewards, and what is yours vs. mine

From the spec, verbatim criteria: (1) you can generate good ideas, (2) you can implement and verify
them. The spec explicitly says not to use LLMs for idea generation, and warns that the second
criterion is deceptively easy — an agent will "grab the wrong tokenizer or forget to save a
checkpoint" while you lose track of what it wrote.

That warning is not hypothetical. This session hit that exact bug class repeatedly (§9). If you want
one concrete way to score well on criterion 2: take the failure list in §9 and check your own run
against it.

**The idea slot is yours.** Nothing in this repo is a novel looping mechanism. What I ran is a
one-axis-at-a-time ablation over five *existing* design choices (from Huginn and the Readout Blind
Spot paper), plus scale-ups. `report.md` §1 is deliberately empty for that reason. §8 below tells you
what my data *constrains* about any new idea — that's evidence, not ideation, and it's the honest
line I've tried to hold.

---

## 3. The honest status of the headline claim

Protocol-matched dense sweep (both runs: same eval, 15 batches, batch_size 4, loops 1–64 every
integer), same config, same seed, different token budgets:

| loop | local, 14.6M tok | Kaggle, 46.0M tok |
|---|---|---|
| 1 | 4.7114 | 4.2580 |
| 4 | 4.4992 | 4.0266 |
| 6 | 4.4762 | 4.0107 |
| **8** | 4.4672 | **4.0071** ← best |
| 10 | 4.4643 | 4.0082 |
| **11** | **4.4642** ← best | 4.0097 |
| 16 | 4.4700 | 4.0212 |
| 32 | 4.5111 | 4.0682 |
| 64 | 4.6036 | 4.1579 |

Read it carefully:

- **3.15x more tokens improved absolute CE by 0.457 nats** (4.4642 → 4.0071). Real, large.
- **Loop gain did not improve: 0.2472 → 0.2509 nats.** Flat. More training bought better loss, not
  more useful depth.
- **The optimum did not move outward.** Argmin went 11 → 8, but the basin is flat to ~0.003 nats
  across loops 6–12 in both runs, which is inside eval noise — so the defensible statement is "useful
  depth stayed in a 6–12 basin", not "it decreased". Either way it did **not** increase.
- Past the optimum, CE rises monotonically. Loop 64 beats loop 1 (4.158 < 4.258), but it is clearly
  worse than loop 8. A model that peaks at 8 and declines afterwards **has saturated at 8**.

**Correction to my own report:** `report.md` §4.2 framed "still better at loop 64 than loop 1" as
"the direct, positive answer to does this keep being useful past where prior work saturates." That
overclaims. Graceful degradation past the optimum is a weaker property than the task wants. I have
corrected that section — see the note in `report.md` §4.2 and `LOG.md` 2026-08-22.

So: treat this repo as a **verified baseline and measurement rig that reproduces the saturation
problem**, not as a solution to it. That is still useful to you — you now know the target to beat
(0.25 nats of loop gain, saturating ~8–12) with a rig that can measure it.

---

## 4. The setup that works (reuse this, it's verified)

| | |
|---|---|
| hidden size | 448 |
| heads / kv heads | 4 / 2 (GQA) |
| head dim | 112 |
| MLP intermediate | 1344 (SwiGLU) |
| layers per loop | 3 (the reused unit is 3 full decoder layers) |
| vocab | 4096, custom byte-level BPE on FineWeb (3.45 chars/token, measured) |
| total params | **9,065,056** — 79.7% in the reused block, 20.2% vocab embedding (tied) |
| seq len | 256, batch 8 |
| loop schedule | random 4–32 per step at train time |
| supervision | mean CE over a bounded random subset of loops (final + 4 others) |

Two choices worth keeping:

- **Small vocab is load-bearing for the param budget.** 4096 keeps the embedding at 20% instead of
  50%+, so most of the ≤10M goes into the block that actually gets reused. The task explicitly
  penalises fixed-size non-reused capacity; the embedding is exactly that.
- **Bounded-subset per-loop supervision, not dense-all.** Supervising every loop every step was
  measurably pathological on MPS (37–158 s/step at 16 loops vs ~6.5 s/step for a 5-loop subset) —
  every intermediate loop gets two outgoing autograd edges instead of one. Caveat: this is an
  engineering choice, never compared to dense supervision at matched cost, so it's a confound if you
  care about supervision scheme as an axis.

Bits-per-byte is in the eval output. Use it, not token perplexity — the custom 4096 vocab makes
token-level ppl incomparable to anything external.

---

## 5. Every result, one table

All measured, all re-verified from JSON on 2026-08-22.

**Screening, 7 arms, one axis at a time, same seed, ~18 min/arm (0.89–1.19M tokens).** Δ is against
center *at the same loop count* (that's how `analyze_screening.py` defines it — not best-vs-best):

| arm | tokens | best loop | best CE | loop-1 CE | Δ vs center |
|---|---|---|---|---|---|
| **no_state_renorm** | 0.99M | 8 | **6.028** | 6.081 | **−0.746** |
| fixed_loops16 | 1.19M | 4 | 6.644 | 6.700 | −0.128 |
| inject_concat | 0.89M | 2 | 6.757 | 6.797 | −0.028 |
| truncate8 | 1.19M | 4 | 6.757 | 6.829 | −0.016 |
| center | 0.89M | 4 | 6.772 | 6.790 | — |
| no_depth_init | 0.99M | 8 | 6.914 | 6.919 | +0.140 |
| inject_none | 0.99M | 1 | 6.951 | 6.951 | +0.161 |

**Scale-ups and checks:**

| run | tokens | best loop | best CE | note |
|---|---|---|---|---|
| no_state_renorm, Kaggle T4 | 46.0M | 8 | **4.007** | best model in repo; **bpb 1.733** |
| no_state_renorm, local MPS | 14.6M | 11 | 4.464 | bpb 1.931 |
| center config (2nd seed, mislabeled dir) | 5.36M | 4 | 5.671 | see §9 bug #2 |
| fixed_loops16 (corrected) | 1.98M | 7 | 6.312 | flat to loop 32 |
| seed=1 replication: no_state_renorm | 0.79M | 8 | 6.252 | gap vs center 0.496 nats |
| seed=1 replication: center | 0.99M | 8 | 6.749 | direction replicates, magnitude varies |
| compute-matched non-looped baseline | 0.30M | n/a | 6.611 | **would not train stably**, §7 |

Seed sensitivity is real: the same comparison gave 0.746 nats at seed 0 and 0.496 at seed 1. Same
sign, ~1.5x magnitude spread. Single-seed margins under ~0.15 nats in the screening table should be
treated as unresolved.

---

## 6. What each of the five axes actually bought

- **State renorm off — the only large effect.** Everything else is ≤0.16 nats at screening.
- **Injection matters, and `inject_none` is the cleanest negative in the project.** Without
  re-injecting the input each loop, the per-loop curve is *flat* (6.951 → 6.957 across 32 loops) —
  loops do literally nothing. Confirms the loop needs new information each iteration, not just more
  compute. Worth keeping in your own design as a sanity control.
- **Depth-aware init (1/sqrt(2·n_loop_eff) on residual output projections) is a real stabiliser**
  (+0.140 CE when removed) and became load-bearing later — the non-looped baseline NaN'd without it.
- **BPTT truncation (last-8) was ~neutral** (−0.016, inside noise). Note the memory story though:
  full BPTT retains activations across up to 32×3 = 96 sequential layer applications at once, which
  is what actually bounds batch size. Truncation is a memory lever more than a quality lever here.
- **Fixed loop count 16 beat randomized 4–32** (−0.128) at screening. Follow-up at 1.98M tokens
  stayed flat past loop ~4 and didn't resolve it. Unsettled — plausibly the random range spends
  training on cheap 4-loop and expensive 32-loop draws that a fixed schedule never averages over.

---

## 7. The mechanism, and the one genuinely interesting side-result

The instrument that mattered most is readout-independent: perturb `h0`, run a clean and a perturbed
trajectory on the same batch, track ‖h_clean − h_noisy‖ per loop. Ratio <1 = contracting, >1 =
expanding. This does not touch the LM head at all, which is why it survived when loss-based readings
were ambiguous.

- `state_renorm=True`: **contracts** (ratio 0.35 by loop 2), and its loss saturates by loop ~4.
- `state_renorm=False`: **never contracts** (ratio 1.31 → 1.01, always >1), and its loss keeps
  improving to loop 8–11.

The link is mechanical, not just correlational: a contraction mapping converges to a fixed point, and
once you're at the fixed point additional loops have nothing left to do. This is the same failure the
task flags in DEQ ("после нее все вычисления становятся бессмысленными"). So renormalizing the
carried state — which the design prior said would *help*, from Huginn's sphere confinement — is
plausibly the thing *causing* early saturation.

**The side-result worth your attention.** I built the compute-matched non-looped control the task's
framing implies: 33 distinct decoder layers (= 3 layers × 11 loops), no weight tying, 81.4M params,
same blocks, same data. It **could not be trained stably** — NaN across 6+ attempts (steps 13, 51,
55, 142, 411, …), surviving neither depth-init alone, nor 6x lower LR, nor tighter clipping. The
looped model at the same effective depth, with no renormalization anywhere, trains fine.

That suggests weight-tying is doing two separable jobs: parameter efficiency (the obvious one) and an
implicit regularisation that stabilises the un-renormalized, expanding regime. Flagged honestly: this
is inference over three results, not an experiment built to test it, and my LR/init search on the
baseline was shallow. Do not cite it as established.

---

## 8. What the data constrains about any new idea

Not ideas — constraints. The gap to close is: **~0.25 nats of loop gain, saturating at 8–12 loops,
unchanged by 3.15x more tokens.**

- Anything that makes the loop map contract faster will make saturation *worse*, not better. The
  contraction diagnostic is cheap and readout-independent — measure it early on whatever you build.
- Absolute-loss improvements and loop-depth improvements are decoupled here. A change that lowers CE
  is not evidence it helped the loop. **Always report loop-1 CE alongside best CE**; the interesting
  quantity is the difference, and it's easy to accidentally optimise the wrong one.
- `inject_none` proves loops need new information per iteration. Whatever you add, ask what new
  information enters at loop *k* that wasn't there at loop *k−1*.
- The eval grid matters. My own "best loop 8 vs 11" comparison was nearly a grid artifact (§9 bug #6)
  — sweep every integer, not a coarse grid, when you're comparing optima.
- The task explicitly names normalizations, looping schemes, and exploration-during-loops as fair
  game, and explicitly accepts a negative result with good analysis.

---

## 8b. Two open conventions you should settle before you build (added 2026-08-22)

**Metric: bits-per-byte, yes — and fixing it exposed a real bug.** `eval.py` already computed bpb
everywhere, so switching the headline metric costs nothing. But it was computed with
`CHARS_PER_TOKEN = 3.45`, a constant estimated from a **5-document sample**, and it counted
*characters* where bits-per-*byte* needs bytes. Measured over the full 6M-token validation shard the
true value is **3.3358 bytes/token** (3.3162 chars/token; corpus is 1.006 bytes/char). Every bpb
figure was therefore ~3.4% optimistic. Corrected everywhere on 2026-08-22:

| | old (wrong) | corrected |
|---|---|---|
| best model (Kaggle, 46M tok, loop 8) | 1.676 | **1.733** |
| local run (14.6M tok, loop 11) | 1.867 | **1.931** |
| non-looped baseline | 2.764 | **2.859** |

The lesson generalises: bpb's *only* advantage over token-perplexity is cross-tokenizer
comparability, so the bytes/token constant is load-bearing. Measure it on the exact set you report
on, not on a convenience sample.

One caveat on comparability: **arXiv 2604.21106 reports validation loss in nats, not bpb**, with a
32K Llama-2 tokenizer. So bpb will not line you up with that paper's numbers either — its nats/token
and your nats/token are over different token distributions. bpb is still the right primary metric;
just don't expect it to buy a direct comparison to that specific work.

**Parameter budget: "≤10M" is genuinely ambiguous, and it's your call.** The spec says only
"до 10M параметров" with no qualifier — the plain reading is *total*. I verified your recollection
about the paper: arXiv 2604.21106 explicitly uses "unique **non-embedding** parameter count". So
there is real precedent for the other reading.

The risk is asymmetric. Assuming non-embedding when the grader meant total = **violating a hard
constraint** with a ~11–12M-parameter model. Assuming total = merely leaving capacity unused. My
build took the conservative reading (9.07M total, of which 7.23M non-embedding).

What the other reading would buy, if you take it (computed with `src/param_budget.py`):

| config | block | embed | total | non-emb |
|---|---|---|---|---|
| current: H=448, I=1344, V=4096 | 7.23M | 1.84M | 9.07M | 7.23M |
| H=504, I=1512, V=4096 | 8.89M | 2.06M | 10.96M | **8.90M** |
| H=504, I=1680, V=4096 | 9.66M | 2.06M | 11.72M | **9.66M** |
| H=448, I=1344, **V=16384** | 7.23M | 7.34M | 14.57M | **7.23M** |

Two consequences worth more than the raw headroom:

1. **+23–34% more reused-block capacity** (7.23M → 8.9–9.7M) at H≈504. H=560 overshoots even the
   non-embedding budget (11.1M).
2. **Vocab becomes free, which changes a decision I made for the opposite reason.** I picked 4096
   *specifically* to keep the embedding from eating a total budget (§4). Under a non-embedding
   convention that rationale evaporates, and 4096 is unusually small — it costs you bytes/token
   (mine: 3.34; a 32K Llama tokenizer gets ~4+), and since bpb divides by bytes/token, a larger vocab
   is plausibly a direct bpb win. Untested here. Watch the interaction with §9 bug #4 though: the
   `[B,T,V]` per-loop logits tensor scales linearly in V, so a 16K vocab makes the eval-time OOM
   four times easier to hit.

Whichever you pick: **report both numbers prominently** (total and non-embedding). That is cheap,
removes the ambiguity for the grader, and costs nothing if you're right about the convention.

**External prior worth knowing before you design.** arXiv 2604.21106 fits a recurrence-equivalence
exponent **φ = 0.46**: looping a block r times is worth about r^0.46 unshared blocks, so r=4 buys
~1.86 blocks, not 4. They also find that at *matched training compute* each additional recurrence
**predictably increases** validation loss, monotonically over r ∈ {1,2,4,8}, with no crossover in
their compute window. That is an independent, much better-resourced result pointing the same way as
my §3 measurement, and it means the task's objective is hard on purpose — you are being asked to
beat a scaling law that currently says more recurrence is a losing trade at fixed compute.

---

## 9. Failure modes that cost real time — check your run against these

Ranked by how much they cost. The spec predicted this category; here is what actually happened.

1. **Silent GPU corruption under sustained MPS load (~700s).** All forward passes returned exact
   zeros. No exception, no NaN, loss looked plausible in the summary line. Caught only by reading raw
   values. Two fixes, both kept: a degenerate-output check that raises on `loss==0.0`/NaN/Inf/zero
   state-norm, and running training as 240s subprocess chunks that resume from checkpoint. This
   later recovered a real live failure correctly.
   *There is a second trigger:* rapid sequences of short `eval.py` invocations with no cooldown hit
   the same driver error (`kIOGPUCommandBufferCallbackError...`), once as a fake data-looking NaN,
   once as a hang needing `kill -9`. `eval.py` never got the chunking discipline `train.py` has.
2. **A config that silently didn't apply.** `run_full.py` rebuilt a full run's config from the
   screening JSON but read only `model_cfg` — and the `fixed_loops16` arm's *only* difference from
   center lives in `TrainConfig`. So a run named `fixed_loops16` trained the default random schedule
   for 1.5h. Caught by reading the checkpoint's own saved `train_cfg`, not the directory name.
   **Check what your checkpoint says it is, not what you named it.**
3. **Checkpoints that never saved.** `eval_every_tokens` worked out to 386 steps while a chunk could
   only do ~141 — so no checkpoint was ever written and every chunk restarted from zero. Wasted ~970s
   before anyone noticed. This is literally the spec's "забудут сохранить чекпойнт".
4. **Eval OOM at high loop counts.** `return_all_loops=True` materialises `[B,T,V]` logits for every
   swept loop simultaneously — 42GB at batch 32 / 64 loops. Needs its own memory guard and a *small,
   separate* eval batch size (4). Bit me once locally and once again on the T4.
5. **An unverified "bigger GPU ⇒ bigger batch" assumption.** Set batch 96 on the T4; OOM'd on step 1.
   A T4's ~14.6GB is *comparable to*, not more generous than, the local cap I'd already tuned to.
6. **Coarse eval grids hiding the answer.** See §3 — the two runs' optima were measured on different
   grids and weren't comparable until I re-ran a dense sweep today.
7. **Accidentally overwriting a result file.** `eval.py` unconditionally writes
   `eval_<dir>.json`; re-running it for a reproducibility check clobbered the file the report cited.
   Restored from an in-conversation copy, verified byte-identical. Copy result files aside before
   re-running anything that writes them.
8. **fp32 catastrophic cancellation in a diagnostic.** `contraction_dist` is a norm of a difference
   between two large (1e4–1e5), nearly-equal vectors. Reproduces to only ~7% run-to-run; the ratio is
   stabler (~2%). Qualitative claim survives, exact decimals don't.

---

## 10. Infrastructure: what to reuse, what to skip

**Reuse:**
- `src/model.py` — Qwen3-derived looped block, verified against the real `Qwen3DecoderLayer` to
  2.4e-7 with copied weights. Five toggles as plain config flags.
- `src/test_model.py` — 5 checks including exact identities (full-BPTT vs truncated forward must be
  bit-identical; no_grad windowing). Run it before spending compute.
- `src/chunked_runner.py` + the degenerate-output check in `train.py` — the MPS survival kit.
- `src/eval.py` — per-loop CE/ppl/bits-per-byte + predictive entropy + contraction estimate.
- `kaggle/main.py` — self-contained T4 kernel (script kernels can't import sibling files). This ran
  the best model. Kaggle gives ~30 GPU-h/week free, ~2400–2900 tok/s here vs ~1000–1300 local.

**Skip or redo:** `src/baseline_nonlooped.py` works but its LR/init search was shallow — if the
non-looped control matters to your argument, budget real time for it.

---

## 11. Commands and repo map

```bash
python src/train_tokenizer.py    # 4096-vocab BPE, ~15s
python src/data.py               # streams+packs FineWeb → data/{train,val}.bin, ~90s
python src/test_model.py         # 5 correctness checks — must pass first
python src/run_screening.py      # 7-arm ablation, ~2h10m total
python src/run_full.py no_state_renorm --seconds 14400
python src/eval.py checkpoints/full_no_state_renorm --max-loops 64 --n-batches 15 --batch-size 4
python src/run_second_seed.py    # seed=1 replication
kaggle kernels push -p kaggle/   # the 46M-token run; pull with `kaggle kernels output`
```

- `report.md` — the deliverable. §1 empty, reserved for your idea narrative.
- `LOG.md` — chronological ledger, every bug and fix with timestamps.
- `PLAN.md` — design rationale, rejected alternatives, what would flip each decision.
- `HANDOFF.md` — session-state snapshot.
- Best checkpoint: `checkpoints/full_no_state_renorm_kaggle/last.pt` (36MB, 0 non-finite params,
  verified). **Nothing has been pushed to GitHub or HF** — that's yours to do.

---

## 12. Known weaknesses in what I produced

State them yourself before a reviewer does.

- Core objective not met (§3). Saturation reproduced, not solved.
- Max 46% of the 100M token budget on one run; everything else far less.
- Screening arms are single-seed except the one axis I replicated. Margins <0.15 nats are unresolved.
- The two full runs of the winning config differ in hardware and data stream, not just tokens.
- `state_renorm=True` vs `False` was never run at the *same* large budget back-to-back — the cleanest
  version of the headline comparison is still missing.
- The compute-matched baseline is a trainability result, not a loss-level comparison.
- Loop count 8–12 is the useful range; "many loops" in the task's sense is not demonstrated.
