"""Push the best checkpoint to a public Hugging Face model repo. Not run automatically by anything
else in this project -- pushing anywhere public is the user's call, per PLAN.md sec 6.

Usage: python src/upload_checkpoint.py checkpoints/<run_name> <hf_repo_id>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir", type=str)
    ap.add_argument("repo_id", type=str, help="e.g. Arsen4ikVar/tlab-looped-transformer")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    ckpt = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"])

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, exist_ok=True, private=args.private)

    readme = f"""---
tags: [looped-transformer, weight-tied, fineweb, from-scratch]
---
# {args.repo_id.split('/')[-1]}

Weight-tied looped transformer, {cfg.hidden_size}-dim, {cfg.layers_per_loop} layers/loop, trained on
FineWeb next-token prediction. T-Lab test task submission.

Trained {ckpt.get('tokens', 0)/1e6:.1f}M tokens, step {ckpt.get('step')}.

Config: `{json.dumps(dataclasses.asdict(cfg))}`

See the repo README and report.md for the full architecture, ablation results, and how to load this
checkpoint with `src/model.py::LoopedTransformer`.
"""
    (ckpt_dir / "README.md").write_text(readme)

    api.upload_file(path_or_fileobj=str(ckpt_dir / "last.pt"), path_in_repo="model.pt",
                     repo_id=args.repo_id)
    api.upload_file(path_or_fileobj=str(ckpt_dir / "README.md"), path_in_repo="README.md",
                     repo_id=args.repo_id)
    print(f"uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
