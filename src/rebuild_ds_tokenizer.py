"""Reproduce the DataSphere kernels' tokenizer locally, so DS checkpoints stop being un-evaluable.

Every DataSphere job here trains its OWN BPE inside the job and returns weights WITHOUT it, so a DS
checkpoint evaluated against `configs/tokenizer.json` reports CE ~9.3 against a chance level of 8.3178
(sec6.0 row 35; one artifact was quarantined for exactly this). That made every DS arm comparable to
nothing outside its own job family.

But the kernel's `train_tokenizer()` is DETERMINISTIC -- first 5,000 documents of a fixed non-shuffled
stream, NFKC normalizer, BPE(unk_token="<unk>"), vocab 4096, fixed special tokens -- and the function
is byte-identical across all four frozen kernels (md5 1dab774d..). So it can be rebuilt here, with no
job and no quota, and the result checked against the thing it must reproduce: a DS checkpoint must
stop evaluating at chance.

    python src/rebuild_ds_tokenizer.py            # writes configs/tokenizer_datasphere.json
"""
from __future__ import annotations
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "tokenizer_datasphere.json"

def main() -> int:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from datasets import load_dataset
    # Transcribed from runs_frozen/ds_dc_s0/main.py:797-814 -- identical in all four frozen kernels.
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    def text_iterator():
        for i, item in enumerate(ds):
            if i >= 5000:
                break
            yield item["text"]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=4096,
                                  special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
                                  show_progress=False)
    tok.train_from_iterator(text_iterator(), trainer=trainer)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(OUT))
    print(f"wrote {OUT}, vocab_size={tok.get_vocab_size()}")
    same = (ROOT / "configs" / "tokenizer.json").read_bytes() == OUT.read_bytes()
    print(f"byte-identical to the shipped tokenizer: {same}  (expected False -- different procedure)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
