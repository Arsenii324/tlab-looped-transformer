"""Train one short chunk (a bounded number of seconds) of one run, resuming from its checkpoint if
one exists. Exists to be invoked repeatedly as a fresh subprocess -- see LOG.md 2026-08-13 01:52:
sustained MPS load in a single process silently corrupted output (all-zero forward passes, no
exception) after ~700s on this hardware. A fresh process per chunk means a fresh Metal context, and
no chunk runs long enough to reach that failure window.

Usage: python src/train_one_chunk.py <config.json> <chunk_seconds>
config.json: {"model_cfg": {...}, "train_cfg": {...}}
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig  # noqa: E402
from train import TrainConfig, run  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    config_path = pathlib.Path(sys.argv[1])
    chunk_seconds = float(sys.argv[2])
    cfg = json.loads(config_path.read_text())
    model_cfg = ModelConfig(**cfg["model_cfg"])
    train_cfg = TrainConfig(**cfg["train_cfg"])

    log_path = ROOT / "checkpoints" / f"{train_cfg.run_name}_history.json"
    ckpt_path = pathlib.Path(train_cfg.ckpt_dir) / train_cfg.run_name / "last.pt"
    resume = ckpt_path.exists()

    hist = run(model_cfg, train_cfg, log_path=log_path, max_seconds=chunk_seconds, resume=resume)
    print(f"CHUNK_DONE steps_logged={len(hist)} "
          f"last_step={hist[-1]['step'] if hist else -1}")


if __name__ == "__main__":
    main()
