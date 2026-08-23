# tlab-loop-transformer

A looped (weight-tied) transformer, ≤10M parameters, pretrained on FineWeb next-token prediction in
≤100M tokens, built to keep benefiting from many loops rather than saturating after ~10. T-Lab test
task. Design rationale and decision log: `PLAN.md`, `LOG.md`. Write-up: `report.md`.

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
  upload_checkpoint.py  Hugging Face upload (not run -- reserved for the user's decision)
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
python src/test_model.py                  # 9 correctness gates; must pass
python src/check_tokenizer_identity.py checkpoints/<ckpt> --expect-ce1 <CE@1 from the model card>
python src/eval.py checkpoints/<ckpt> --max-loops 64

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
