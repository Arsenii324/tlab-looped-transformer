"""Push the best checkpoint to a public Hugging Face model repo. Not run automatically by anything
else in this project -- pushing anywhere public is the user's call, per PLAN.md sec 6.

WHAT THIS FILE GOT WRONG, AND WHY IT IS WORTH A PARAGRAPH. The first version uploaded exactly two
files: `last.pt` -> `model.pt`, and a generated card carrying config, token count and step. It did
NOT upload `configs/tokenizer.json`. That is the *same* failure this project already caught once and
wrote a gate against (sec6.0 row 20): a vocabulary mismatch does not raise -- it reports
CE ~ ln(4096) = 8.32, which reads as a broken model rather than a missing file. The earlier fix
landed in the repo README and never reached the shipping path, so the gate existed, was proven to
work, and could not be run by the person who needed it -- the card had no CE@1 to pass to
`--expect-ce1`, which is exactly what the README instructs a grader to read from it.

Both halves are fixed here, and the script now REFUSES to upload rather than shipping an
unverifiable artifact:
  * `configs/tokenizer.json` goes up alongside the weights, and `src/model.py` with it so the
    checkpoint is loadable without cloning the GitHub repo;
  * the card carries CE@1, best CE and its depth, val perplexity, bits/byte and this checkpoint's
    OWN state norms, all read from the checkpoint's own `eval_*.json` rather than transcribed;
  * the card prints the gate command with the number already substituted.

Usage: python src/upload_checkpoint.py checkpoints/<run_name> <hf_repo_id>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig, LoopedTransformer  # noqa: E402
from eval import BYTES_PER_TOKEN  # noqa: E402
from plateau import plateau  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def build(ckpt: dict) -> LoopedTransformer:
    m = LoopedTransformer(ModelConfig(**ckpt["model_cfg"]))
    m.load_state_dict(ckpt["model"]); m.eval()
    return m


def n_parameters(model: LoopedTransformer) -> int:
    """MUST go through `model.num_parameters()` (i.e. `.parameters()`, which de-duplicates shared
    tensors), NOT `sum(v.numel() for v in state_dict.values())`.

    `self.lm_head_weight = self.embed.weight` registers the SAME nn.Parameter under a second name,
    so a state_dict sum counts the tied embedding twice: 9,064,608 becomes 10,899,616 -- the
    difference is exactly vocab*hidden = 4096*448 = 1,835,008. The first version of this card
    printed the state_dict sum, which reports **this model as violating the task's 10M parameter
    cap when it does not**. Weight tying is the whole point of the architecture; a counter that
    misses it is worse than no counter."""
    return model.num_parameters()


def state_norms(ckpt_dir: pathlib.Path, ckpt: dict, loops=(1, 8, 16, 64)) -> dict | None:
    """This checkpoint's OWN ‖h‖ at a few depths. In the card because sec4.6's radial-clamp levels
    are absolute numbers read off a DIFFERENT checkpoint, and the three released models differ in
    scale by up to 380x -- copying the report's levels onto these weights inflates the state instead
    of constraining it. A reader who has the model's own norms cannot make that mistake."""
    val_path = ROOT / "data" / "val.bin"
    if not val_path.exists():
        return None
    val = np.memmap(val_path, dtype=np.uint16, mode="r")
    seq = ckpt["train_cfg"]["seq_len"]
    rng = np.random.default_rng(0)
    ix = rng.integers(0, len(val) - seq - 1, size=4)
    x = torch.from_numpy(np.stack([val[i:i + seq] for i in ix]).astype(np.int64))
    model = build(ckpt)
    with torch.no_grad():
        _, sn = model(x, n_loops=max(loops), return_all_loops=False)
    return {t: sn[t - 1] for t in loops}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir", type=str)
    ap.add_argument("repo_id", type=str, help="e.g. Arsen4ikVar/tlab-looped-transformer")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the card and print what would be uploaded; touch no network")
    args = ap.parse_args()

    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    name = ckpt_dir.name
    ckpt = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"])

    # Refuse to ship an artifact that cannot be verified. Both of these are the failure this file's
    # docstring is about, so they are checked before anything is created, not after.
    tok_path = ROOT / "configs" / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"{tok_path} is missing. Uploading weights without the vocabulary that produced them "
            f"makes every eval land at chance (ln {cfg.vocab_size} = {math.log(cfg.vocab_size):.4f}) "
            f"and look like a broken model. Refusing to upload.")
    eval_path = ckpt_dir / f"eval_{name}.json"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"{eval_path} is missing, so the card cannot state a CE@1 -- and without one the "
            f"tokenizer gate this project wrote (src/check_tokenizer_identity.py --expect-ce1) "
            f"cannot be run by whoever downloads this. Run: python src/eval.py {ckpt_dir} "
            f"--max-loops 64")

    curve = {int(k): v for k, v in json.load(eval_path.open())["val_ce"].items()}
    ce1 = curve[1]
    best_d = min(curve, key=curve.get)
    best = curve[best_d]
    lo, hi, _ = plateau(curve)
    grid = f"dense {min(curve)}..{max(curve)}" if len(curve) > 12 else f"sparse {sorted(curve)}"
    norms = state_norms(ckpt_dir, ckpt)
    n_params = n_parameters(build(ckpt))

    norm_block = ""
    if norms:
        norm_block = (
            "\n## This checkpoint's own state norms\n\n"
            "| loop | " + " | ".join(str(t) for t in norms) + " |\n"
            "|---|" + "---|" * len(norms) + "\n"
            "| ‖h‖ | " + " | ".join(f"{v:.1f}" for v in norms.values()) + " |\n\n"
            "Stated because the released models differ in state scale by up to 380x. Any absolute "
            "clamp/threshold level in the report was measured on **one** checkpoint and does not "
            "transfer; derive levels from these numbers or re-run `src/radial_clamp.py` on this "
            "checkpoint, which does it for you.\n")

    readme = f"""---
tags: [looped-transformer, weight-tied, fineweb, from-scratch]
---
# {args.repo_id.split('/')[-1]}

Weight-tied looped transformer: one {cfg.layers_per_loop}-layer Qwen3-style block applied `r` times,
{cfg.hidden_size}-dim, **{n_params:,} parameters**, trained from scratch on FineWeb next-token
prediction. T-Lab test task submission. Run `{name}`, {ckpt.get('tokens', 0)/1e6:.1f}M tokens, step {ckpt.get('step')}.

## Results

| metric | value |
|---|---|
| CE @ 1 loop | **{ce1:.4f}** |
| best val CE | **{best:.4f}** (at {best_d} loops) |
| val perplexity | **{math.exp(best):.2f}** |
| bits/byte | **{best/math.log(2)/BYTES_PER_TOKEN:.4f}** (at {BYTES_PER_TOKEN} bytes/token) |
| useful-depth plateau | **[{lo}, {hi}]** on the {grid} eval grid |
| loop gain (CE@1 − CE@best) | **{ce1-best:.4f}** |

Perplexity is **tokenizer-dependent** and this model uses its own {cfg.vocab_size}-token BPE, so it
is not comparable across submissions; bits/byte is the figure that survives a change of tokenizer.

## Files, and why the tokenizer is one of them

- `model.pt` — weights (`torch.load`, `weights_only=False`; contains `model`, `model_cfg`, `train_cfg`)
- `tokenizer.json` — **the vocabulary these weights were trained with.** Do not substitute another
  one and do not retrain it: a mismatch raises nothing and reports CE ≈ ln({cfg.vocab_size}) =
  {math.log(cfg.vocab_size):.4f}, i.e. chance, which looks like a broken model rather than a broken setup.
- `model.py` — the architecture, so this checkpoint loads without cloning the GitHub repo.

## Verify the download before trusting a number

```bash
python src/check_tokenizer_identity.py <this checkpoint> --expect-ce1 {ce1:.4f}
```

That gate judges vocabulary against *chance* and protocol drift against the sample's own SEM, so it
distinguishes "wrong vocabulary" from "slightly different eval batch". Expect |diff| well under 0.1.

```python
import torch
from model import Config, LoopedTransformer
ck = torch.load("model.pt", map_location="cpu", weights_only=False)
m = LoopedTransformer(Config(**ck["model_cfg"])); m.load_state_dict(ck["model"]); m.eval()
logits_per_loop, state_norms = m(input_ids, n_loops={best_d}, return_all_loops=True)
```
{norm_block}
## Config

`{json.dumps(dataclasses.asdict(cfg))}`

See the GitHub repo's `report.md` for the full ablation set, the negative results, and the
failure log (§6.0).
"""
    (ckpt_dir / "README.md").write_text(readme)

    uploads = [(ckpt_dir / "last.pt", "model.pt"),
               (tok_path, "tokenizer.json"),
               (ROOT / "src" / "model.py", "model.py"),
               (ckpt_dir / "README.md", "README.md")]

    if args.dry_run:
        print(f"DRY RUN -- would upload to {args.repo_id}:")
        for src, dst in uploads:
            print(f"  {str(src):<60} -> {dst}  ({src.stat().st_size/1e6:.2f} MB)")
        print(f"\ncard written to {ckpt_dir/'README.md'}")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, exist_ok=True, private=args.private)
    for src, dst in uploads:
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=args.repo_id)
        print(f"  uploaded {dst}")
    print(f"uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
