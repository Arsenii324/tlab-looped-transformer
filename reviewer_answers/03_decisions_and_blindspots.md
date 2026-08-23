# Reply — 2026-08-23 11:20 MSK · `DECISIONS.md`, D2 closed, two screens launched

Your rewritten prompt was better than the one it replaced and I built to it rather than to the
original. **`DECISIONS.md` is in the repo root** — one row per choice, provenance-tagged
MEASURED / INHERITED / ASSUMED / UNEXAMINED, values pulled programmatically from `model.py` and
`train.py` rather than recalled, plus your three questions and a fourth I added
(*which changes would invalidate comparability*).

Rather than restate it, here is what writing it **found** — the parts I did not know before the table
forced a tag into every row.

## What the forcing function actually caught

**1. §3.4 was arguing against itself, exactly as you suspected — and the fix is better than the bug.**
It read: *"this project's own architecture passes trivially and uninterestingly — it has no loop
conditioning at all."* False since §4.17. **Annealed supervision *is* loop conditioning, applied
through the loss rather than the parameters**: which loop indices receive gradient is a function of
the step, and it satisfies the function-vs-table rule **perfectly** — not by abstaining, but because
*there is no table to outgrow, since there are no parameters at all*. That is a strictly stronger
position under the task's own scale criterion than IterAdaLN's, which still pays ~344k params for a
function of `t`. Corrected in-text with the superseded sentence quoted.

**2. The three MEASURED rows I would now trust least were not the ones I expected.**
`inject_mode`, `depth_init` and `truncate_bptt` were all decided in the §4.1 screening sweep — at
~1M tokens, on a **wall-clock** budget, and under `state_renorm=True`. **None has been re-tested in
the no-renorm regime that every headline result uses.** `state_renorm` itself survives only because
−0.744 nats is an order of magnitude above the confound. That is three architecture axes resting on a
sweep the report already partially retracts.

**3. The largest unscreened capacity decision is not the LR — it is the MLP ratio.**
`intermediate_size = 1344` is ratio **3.0**, neither Qwen3's 8/3 nor the common 4.0. It was chosen by
the budget search as a round multiple and **never screened as an axis**, and it holds ~64% of block
parameters. Bigger lever than anything in §4, and I had not registered it as a decision at all.

## Both screens you proposed are running

`tlab-hyper-screen` (`bt1e9hht62prt6ah8974`), six arms at 2.5M tokens, **with an in-job reference arm
at the current values** so both series are measured against the configuration every other result used
(§4.15: cross-job drift 0.0074–0.0334, larger at deep schedules — same order as any effect expected here):

- LR ∈ {1e-3, **3e-3**, 6e-3} at wd 0.05
- wd ∈ {0, 0.01, **0.1**} at lr 3e-3

Pre-registered: *if 3e-3 is within ~0.05 of the best of {1e-3, 6e-3}, the inherited value is
defensible and §6.0b can say so with a number; if 1e-3 wins by more, every result in this report was
measured at a stated handicap* — reportable, not fatal. Your framing that a screened null strengthens
the report while a *changed* axis at hour 12 is a rewrite is exactly right, and it is written into the
kernel docstring so nobody is tempted later.

Your optimizer table settled a question I had been carrying: the AdamW/Muon split is **cultural, not
scale-driven**, and the paper being replicated most directly (Sharma & Vu) is AdamW at this exact
parameter scale. `DECISIONS.md` records AdamW as INHERITED-and-correct rather than INHERITED-and-lucky.

## D2 is now closed — and it closed better than "it ran"

The one untested link was cold `data.py` from a fresh clone. Run:

```
fresh clone -> python src/data.py
  [train] done: 92,000,000 tokens from 99,499 docs in 95s
  [val]   done:  6,000,000 tokens from  6,589 docs in  6s

sha256(val.bin)    original 2ae0968496657ba6 == cold 2ae0968496657ba6   IDENTICAL
sha256(train.bin)  original 754fbad1a5cb618a == cold 754fbad1a5cb618a   IDENTICAL
meta.json IDENTICAL
```

**Byte-identical.** The FineWeb stream is deterministic at this sample and offset, so a grader
regenerates exactly the shards every number was measured on. Full verified chain: clone → shipped
tokenizer (matching hash) → identical shards in 101s → `test_model.py` 9/9 → `test_plateau.py` 8/8 →
tokenizer-identity gate PASS on both Kaggle checkpoints.

## Your urgent fp16 item: real mechanism, not live here

Worth the check and I am glad you flagged it. At the deep schedules' observed ‖h‖ ≈ 1e5 the
per-element mean-of-squares is **2.3e7 against fp16's 65,504 ceiling**, and the risk **grows with loop
count** — so it would strike precisely the deep runs the method depends on. Not live, for two
independent reasons: **(a)** no `autocast`/`float16`/`bfloat16`/`half`/`GradScaler` anywhere in the
training path — fp32 throughout; **(b)** `RMSNorm` upcasts (`x = x.float()` before the reduction)
because it was written to match Qwen3's reference rather than hand-rolled. Verified empirically at
the deep run's actual scale: finite output, RMS exactly 1.0000. Now stated in §6.0b as a warning to
anyone reproducing this in mixed precision.

**RETRACTION (11:20): my "73% is wrong, the paper says 85%" was itself wrong.** The v3 tarball,
now in `papers/sources/2511.08577/`, reads *"over 73\% of next-tokens are correctly predicted at the
first iteration"* (`3_method.tex` line 206) — **your figure was right**. I had checked only the arXiv
HTML through a summarising fetch, which returned 85%; that number appears in the source only as
unrelated table cells. I asserted a correction to you from a summary rather than a primary source,
which is the same class of error as §6.0's worst rows and is now logged as one of them.

**Your item 5 was already fixed and is now re-verified:** `BYTES_PER_TOKEN = 3.3358` measured over the
full 6M-token shard (20,014,585 bytes / 6,000,000 tokens), discrepancy **0.0000**. Bits/byte divides
by bytes. (A 200k subsample gives 3.185 — the shard start is unrepresentative, which is a small
cautionary tale of its own.)

## Adopted from your last message

- **S1 → §4.7 now says it explicitly**: the head is `[h_t, h_{t−1}]` with a tanh hidden layer, i.e.
  **PALBERT's own best ablation row**, and a single linear on a single state is their *worst* row and
  the Ouro form. So the negative rules out the strongest published configuration of the method the
  task names.
- **D3 → your disclosure framing is better than mine and is adopted.** If the norm-penalty checkpoint
  ships, the model card and report will say in the same sentence that the penalty's effect **shrank
  12× from 2.5M to 90M (−0.366 → −0.030)**. Uploading the winner while disclosing that the margin is
  largely a small-budget artifact beats both hiding it and shipping the worse checkpoint.
- **MLA + LLA** is filed as §8 material with your framing: LLA measured the recurrent KV trajectory to
  be genuinely low-rank ("not safely collapsible to a single state", near-lossless to 32× on
  Huginn-3.5B), and §4.3 here independently shows keys near depth-invariant (25.13 → 21.36) while
  values fall 2× — so MLA's imposed structure is the structure a looped model's KV already has. Not
  tonight.
- **STARS Pre-Sandwich** and the **LoopMTP aggregation conflict** are both in `DECISIONS.md` Q3 as
  named-but-untested, with the prediction attached in the first case and the reason it is a genuine
  conflict rather than a missing ablation in the second.

## One place I would still push back

You wrote that the report should say *"we tried two of the four Sharma & Vu interventions and the two
we skipped are the ones the paper rates highest."* All four are in fact run — inter-loop
normalisation, the norm penalty, **raw readout** and **final-only readout** — the last two by
`run_scale_control.py`, which finished seed 0 while your message was in flight:

| arm | best CE | Δ vs control | loop gain |
|---|---|---|---|
| control (norm readout) | 5.3636 | — | 0.1056 |
| raw readout | 5.3380 | −0.0256 | 0.2214 |
| final-only readout | 5.2654 | −0.0982 | 0.2170 |
| norm penalty λ=0.01 | **4.9975** | **−0.3662** | **0.2522** |

Seed 1 is training. The headline caution stands and belongs beside these numbers: the norm penalty's
−0.366 at 2.5M became **−0.030 at 90M**, so all four of these are small-budget screens until shown
otherwise — which is the same warning that now governs my own annealing result.
