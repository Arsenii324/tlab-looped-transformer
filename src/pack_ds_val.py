"""Pack a validation shard with the REBUILT DataSphere vocabulary, so DS checkpoints can be probed
on in-distribution tokens.

`data/val.bin` is packed with the shipped vocabulary. A DataSphere checkpoint reading it sees token
ids that mean something else (sec4.27). For a CE number that is fatal; for a geometric probe like
`src/depth_key_rank.py` it is merely off-distribution -- but off-distribution is avoidable now that
`src/rebuild_ds_tokenizer.py` reproduces the kernel's vocabulary.

Writes `data/val_datasphere.bin` (uint16). Documents are taken from far enough into the stream to sit
past what the kernels train on.

    python src/pack_ds_val.py [n_tokens]
"""
from __future__ import annotations
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "val_datasphere.bin"
SKIP_DOCS = 200_000          # past the kernels' training window

def main() -> int:
    n_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
    from tokenizers import Tokenizer
    from datasets import load_dataset
    tok = Tokenizer.from_file(str(ROOT / "configs" / "tokenizer_datasphere.json"))
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    ids: list[int] = []
    for i, item in enumerate(ds):
        if i < SKIP_DOCS:
            continue
        ids.extend(tok.encode(item["text"]).ids)
        if len(ids) >= n_tokens:
            break
    arr = np.array(ids[:n_tokens], dtype=np.uint16)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(OUT)
    print(f"wrote {OUT}: {len(arr):,} tokens, skipped {SKIP_DOCS:,} docs")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
