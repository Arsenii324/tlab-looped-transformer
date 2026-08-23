"""Stream FineWeb, tokenize with the trained BPE tokenizer, pack into uint16 train/val shards.

Streamed only -- disk is the binding local constraint (41GB free on a 95%-full volume). Nothing
raw ever touches disk; only the packed token arrays do (~2 bytes/token, so ~95M tokens is ~190MB).

Train/val split is by STREAM POSITION, not by shuffling after the fact: the tokenizer was trained on
the first ~19,319 documents of the stream (train_tokenizer.py), so packing starts well past that
offset, then continues the same iterator uninterrupted from train into val -- documents are disjoint
by construction (each document is consumed exactly once, in order), no bookkeeping needed to avoid
train/val overlap.

Documents are separated by the tokenizer's <|endoftext|> id (0) when packed, standard practice for
concatenated-document LM pretraining.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "configs" / "tokenizer.json"
DATA_DIR = ROOT / "data"

SKIP_DOCS = 20_000          # past the tokenizer-training slice (19,319 docs), rounded up
TRAIN_TOKENS = 92_000_000
VAL_TOKENS = 6_000_000
EOS_ID = 0


def pack(char_budget_hint_ratio: float = 3.4) -> None:
    from datasets import load_dataset
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    assert tok.token_to_id("<|endoftext|>") == EOS_ID, "EOS id assumption broke, fix EOS_ID"

    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    it = iter(ds)

    print(f"skipping {SKIP_DOCS} docs already used for tokenizer training...", file=sys.stderr)
    t0 = time.time()
    for _ in range(SKIP_DOCS):
        next(it)
    print(f"  skipped in {time.time()-t0:.0f}s", file=sys.stderr)

    def fill(n_tokens: int, tag: str) -> np.ndarray:
        buf = np.empty(n_tokens, dtype=np.uint16)
        pos, docs, t0 = 0, 0, time.time()
        while pos < n_tokens:
            text = next(it)["text"]
            ids = tok.encode(text).ids
            ids.append(EOS_ID)
            take = min(len(ids), n_tokens - pos)
            buf[pos:pos + take] = np.array(ids[:take], dtype=np.uint16)
            pos += take
            docs += 1
            if docs % 5000 == 0:
                print(f"  [{tag}] {pos}/{n_tokens} tokens, {docs} docs, {time.time()-t0:.0f}s",
                      file=sys.stderr)
        print(f"  [{tag}] done: {pos} tokens from {docs} docs in {time.time()-t0:.0f}s",
              file=sys.stderr)
        return buf

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train = fill(TRAIN_TOKENS, "train")
    train.tofile(DATA_DIR / "train.bin")
    del train

    val = fill(VAL_TOKENS, "val")
    val.tofile(DATA_DIR / "val.bin")
    del val

    import json
    meta = dict(vocab_size=tok.get_vocab_size(), eos_id=EOS_ID,
                train_tokens=TRAIN_TOKENS, val_tokens=VAL_TOKENS,
                skip_docs=SKIP_DOCS, dtype="uint16")
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {DATA_DIR/'train.bin'} and {DATA_DIR/'val.bin'}")


def load_shard(name: str) -> np.memmap:
    import json
    meta = json.loads((DATA_DIR / "meta.json").read_text())
    n = meta[f"{name}_tokens"]
    return np.memmap(DATA_DIR / f"{name}.bin", dtype=np.uint16, mode="r", shape=(n,))


if __name__ == "__main__":
    pack()
    import os
    os._exit(0)  # see train_tokenizer.py -- same benign datasets/pyarrow shutdown quirk
