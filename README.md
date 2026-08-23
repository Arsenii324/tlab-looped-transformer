# tlab-loop-transformer

**T-Lab test task submission.** A looped (weight-tied) transformer pretrained from scratch on FineWeb
next-token prediction.

## ▶ Start here: **[`submission/README.md`](submission/README.md)**

**That one page is the submission.** It carries the headline numbers, the finding and its mechanism,
what worked, and a link to each of the 8 documents answering a clause of the brief. Everything else
in this repository is evidence behind it or a dated working record.

| | |
|---|---|
| **Read this first** | **[`submission/README.md`](submission/README.md)** — the submission proper; 8 documents, one per clause of the brief |
| **The full evidence base** | **[`report.md`](report.md)** — 6,700+ lines; evidence, not reading material |
| **Released weights** | **[Arsen4ikVar/tlab-looped-transformer](https://huggingface.co/Arsen4ikVar/tlab-looped-transformer)** |
| parameters | **9,064,608** (cap: 10M) |
| training tokens | **90.0M** (cap: 100M) |
| val perplexity | **38.86** (CE 3.6599 at 10 loops) |
| bits/byte | **1.5829** |
| useful-depth band | **loops 6–17** (dense 1..64 eval grid) |

Perplexity is tokenizer-dependent (this model has its own 4096-token BPE), so **bits/byte is the
figure comparable across submissions.**

> **If you read one thing, read [`submission/README.md`](submission/README.md).** The sections below
> describe how the repository is organised and which surfaces are current; they are not the
> submission.

## How to read this repository

**Two surfaces are current and maintained. Everything else is a dated working record, kept intact.**

| | what it is | trust it? |
|---|---|---|
| **`submission/`** | the readable submission — `METHOD`, `RESULTS`, `EXPERIMENTS`, `SCALE`, `NEGATIVE_RESULTS`, `FAILURES`. Start at `submission/README.md` | **yes — current** |
| **`report.md`** | the complete evidence base. §0 abstract · §3.5 final method · §4 experiments · **§6.0 every error that reached a number, how it was caught, what it cost** (the task asks for this explicitly; it is in the body, not an appendix) | **yes — authoritative** |
| `LOG.md`, `RUNS.md` | append-only. Every run, its pre-registered falsifier, and its outcome in timestamp order | **yes — as history** |
| `reviewer_answers/` | 25 numbered replies to an external reviewer, never edited after sending. Corrections get a *new* file, so an early file may state something a later one withdraws | **as dated correspondence** |
| **everything else** (`PLAN.md`, `BRIEFING.md`, `HANDOFF.md`, `QUEUE.md`, `DECISIONS.md`, `INTERVENTIONS.md`, `STATE_FOR_REVIEWER.md`, `FINAL_ARCHITECTURE.md`, `PROGRESS_REPORT.md`, `HANDOVER_CLAUDE.md`, `METHODS.md`, `VERIFICATION.md`, `REVIEW_NOTES.md`, `subagents/`) | **dated working records from the sessions that produced the work.** They are kept rather than cleaned up — several are *why* the numbers are checkable — but they were written at a point in time and **several predate the day's largest changes.** Some carry the superseded 46.0M-token headline (CE 4.0071 / ppl 54.99) rather than the current one | **no — read as history.** Where one disagrees with `report.md`, `report.md` wins |

**Why they are not rewritten.** This project's own §6.0 lists twelve retracted claims, and the rule it
adopted is that a superseded statement stays **visible** with its correction rather than being edited
away. Silently bringing twenty working documents "up to date" would destroy exactly the audit trail
that makes the retraction record meaningful. The cost is that a reader must know which surface is
current — which is what this table is for.

**Three commands settle any disagreement between documents:**

```bash
python src/headline.py check        # every headline number vs the artifact it came from
python src/make_inventory.py        # regenerates the experiment inventory FROM the stored JSON
python src/check_caveats.py --strict  # no file states a deflated claim without its caveat
```

## Layout

```
src/
  model.py            Qwen3-derived looped transformer, 5 ablation toggles
  test_model.py        correctness checks -- run this before spending any training compute
  train_tokenizer.py   trains the 4096-vocab byte-level BPE tokenizer (streamed FineWeb, ~15s)
  data.py               streams+tokenizes+packs FineWeb train/val shards (~90s)
  train.py              training loop (bounded-subset per-loop supervision, cosine LR, checkpointing)
  chunked_runner.py     runs training as short resumable subprocess chunks (MPS-stability mitigation)
  run_screening.py      runs the one-axis-at-a-time ablation sweep
  run_full.py            full-budget run of one screening arm, chosen and passed by name
  analyze_screening.py  independently re-derives the screening table from raw JSON (verification)
  eval.py                per-loop val perplexity, swept past the trained loop range; contraction estimate
  state_dynamics.py     what the loop map does to the state, measured in the space the readout can
                         see (report.md sec 4.3) -- supersedes eval.py's contraction_ratio, which is
                         confounded by linear norm growth; also probes the input-injection forcing
                         term and the per-loop ||v|| / ||norm1(h)|| early-exit question
  baseline_nonlooped.py compute-matched non-looped baseline (report.md sec 4.4) -- N distinct decoder
                         layers, no weight tying, FLOPs-matched to the winning config's best loop count
  run_second_seed.py    seed=1 replication check on the winning axis (report.md sec 4.1)
  upload_checkpoint.py  Hugging Face upload; ships tokenizer.json + model.py + a generated card,
                         and raises rather than uploading an unverifiable artifact
kaggle/
  main.py               self-contained full-budget training kernel for a Kaggle T4 (everything inlined
                         -- Kaggle script kernels can't import sibling files); actually used this
                         session to scale the winning config well past local MPS throughput
checkpoints/             saved models + JSON histories (gitignored except via explicit add)
data/                    packed token shards (gitignored)
configs/                 tokenizer.json
```

## Quickstart

> **DO NOT run `train_tokenizer.py` if you want to evaluate a released checkpoint.**
> `configs/tokenizer.json` **ships with this repo and is the exact vocabulary the released weights
> were trained with.** `train_tokenizer.py` *overwrites* that file with a freshly-trained one, and a
> BPE retrained from a live stream is not guaranteed byte-identical. A mismatched vocabulary **does
> not raise** — it silently reports cross-entropy near `ln(4096) = 8.32` instead of ≈3.6, which looks
> like a broken model rather than a broken setup. Run `train_tokenizer.py` only when reproducing the
> pipeline end-to-end from nothing, and expect different weights if you do.

```bash
# --- evaluating a released checkpoint (the normal path) ---
python src/data.py                        # uses the SHIPPED configs/tokenizer.json; does not retrain
python src/test_model.py                  # 13 correctness gates; must pass
python src/check_tokenizer_identity.py checkpoints/full_control90_kaggle --expect-ce1 3.9622
python src/eval.py checkpoints/full_control90_kaggle --max-loops 64

# --- or verify the RELEASED weights, downloaded, without trusting this repo's copy ---
huggingface-cli download Arsen4ikVar/tlab-looped-transformer --local-dir /tmp/hfcheck
python src/check_tokenizer_identity.py /tmp/hfcheck/model.pt --expect-ce1 3.9622 \
       --tok /tmp/hfcheck/tokenizer.json
#   verified 2026-08-23: PASS, |diff| = 0.0020 against a chance level of 8.3178

# --- reproducing the whole pipeline from scratch (produces a DIFFERENT tokenizer) ---
python src/train_tokenizer.py             # overwrites configs/tokenizer.json
python src/data.py
python src/test_model.py
python src/train.py                       # trains the default (center) config
python src/eval.py checkpoints/center --max-loops 64
```

`check_tokenizer_identity.py` exists because this exact trap was live in this repo until 2026-08-23:
the Kaggle training kernel trained its BPE fresh and never saved it, so the vocabulary behind the
headline checkpoints existed only as a side-effect of a run. Identity with the shipped vocab is now
**verified** (both Kaggle checkpoints pass) rather than assumed, and the kernel saves its tokenizer.

Everything above runs on CPU or Apple Silicon (MPS) by default; no GPU required for this model size,
though see `LOG.md` for measured MPS throughput and a real memory-cost finding about full
backprop-through-many-loops.

## Why it looks the way it does

Every non-obvious choice is justified in `PLAN.md` at the point it's made, with what was rejected and
what would change the decision. `LOG.md` is the append-only record of what actually happened, in
order, including the mistakes (a memory-cost misdiagnosis, a Kaggle quota exhaustion that later reset
mid-project, a config bug that silently mislabeled a full run, a compute-matched baseline that took
six-plus attempts to train stably at all) — kept rather than cleaned up, because that's what makes the
final numbers checkable.
