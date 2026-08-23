"""Train a small byte-level BPE tokenizer on a slice of FineWeb, entirely in memory.

Why custom and why this small: the param budget (param_budget.py) picked vocab_size=4096 because at
a 10M total-parameter ceiling, the embedding table competes directly with the reusable looped block
for parameters, and the block is the thing this task is actually about. Byte-level BPE (GPT-2 style)
is used so there is no UNK token and no failure mode on unexpected bytes.

Streamed, never bulk-downloaded (disk is the binding local constraint, 41GB free on a 95%-full
volume): documents are pulled from the HF stream and only their tokenizer-training text is kept in
memory, nothing is written to disk except the final ~small tokenizer.json.

Known quirk, worked around rather than investigated: the `datasets` streaming Parquet reader raises
a harmless "Exception ignored" during interpreter shutdown on this environment (a generator
finalizer racing teardown), which can leave the process hanging past a wrapping `timeout`. All real
work here finishes and is flushed to disk before that point, so the script force-exits at the end.
"""

from __future__ import annotations

import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "tokenizer.json"
VOCAB_SIZE = 4096
CHAR_BUDGET = 60_000_000  # ~60MB of raw text is ample for a 4096-vocab BPE trainer
SPECIAL_TOKENS = ["<|endoftext|>"]


def collect_text(char_budget: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    texts, total = [], 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        t = ex["text"]
        texts.append(t)
        total += len(t)
        if i % 2000 == 0:
            print(f"  {i} docs, {total/1e6:.1f}M chars, {time.time()-t0:.0f}s", file=sys.stderr)
        if total >= char_budget:
            break
    print(f"collected {len(texts)} docs, {total/1e6:.1f}M chars in {time.time()-t0:.0f}s",
          file=sys.stderr)
    return texts


def main() -> int:
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    texts = collect_text(CHAR_BUDGET)

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS,
                                   show_progress=True)
    tok.train_from_iterator(texts, trainer=trainer)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(OUT))
    print(f"saved tokenizer to {OUT}, vocab_size={tok.get_vocab_size()}")

    # sanity: compression ratio on a held-out slice of the same pool (last 5 docs, not used to train
    # the trainer differently -- fine, this is just a compression sanity check, not a leakage-
    # sensitive split)
    sample = "\n".join(texts[-5:])
    ids = tok.encode(sample).ids
    print(f"compression check: {len(sample)} chars -> {len(ids)} tokens "
          f"({len(sample)/max(len(ids),1):.2f} chars/token)")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(rc)  # sidesteps a benign but hang-prone datasets/pyarrow finalizer at interpreter exit
