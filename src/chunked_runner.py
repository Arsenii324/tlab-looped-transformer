"""Run one named training run as a sequence of short subprocesses, each resuming from the last
checkpoint. Shared by run_screening.py and run_full.py so the hardening lives in one place --
LOG.md 2026-08-13 01:52/02:15: a single continuous process silently corrupted output (all-zero
forward passes, no exception) after ~700s on this hardware, and a first attempt at fixing this in
run_screening.py alone left run_full.py on the old, unprotected path, which would have hit the same
failure on any run longer than ~700s (i.e. every full-budget run).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "checkpoints" / "_arm_configs"
CHUNK_SECONDS = 240   # well under the measured ~700s failure window (2.9x margin)
MAX_RETRIES_PER_CHUNK = 3   # failures observed intermittent, not permanent (LOG.md 2026-08-13 02:05)


def run_chunked(name: str, mcfg, tcfg, budget_seconds: float, fresh: bool = True) -> dict:
    """`fresh`: if True, deletes any pre-existing checkpoint for this run_name before starting (the
    right default for a NEW run -- a stale checkpoint from a differently-configured prior attempt
    would otherwise resume silently into the wrong run). Pass False to continue an already-started
    run across separate top-level invocations of this function (e.g. topping up a full run's budget
    in a later call)."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIGS_DIR / f"{name}.json"
    config_path.write_text(json.dumps(dict(model_cfg=dataclasses.asdict(mcfg),
                                            train_cfg=dataclasses.asdict(tcfg))))

    ckpt_path = pathlib.Path(tcfg.ckpt_dir) / name / "last.pt"
    if fresh and ckpt_path.exists():
        ckpt_path.unlink()

    tokens_per_step = tcfg.batch_size * tcfg.seq_len
    total_steps = tcfg.total_tokens // tokens_per_step
    hist_path = ROOT / "checkpoints" / f"{name}_history.json"

    t0 = time.time()
    chunk_i = 0
    consecutive_failures = 0
    while time.time() - t0 < budget_seconds:
        chunk_i += 1
        remaining = budget_seconds - (time.time() - t0)
        this_chunk = min(CHUNK_SECONDS, remaining)
        if this_chunk < 15:
            break
        print(f"  [{name}] chunk {chunk_i}, budget {this_chunk:.0f}s "
              f"(elapsed {time.time()-t0:.0f}/{budget_seconds:.0f}s)", flush=True)
        proc = subprocess.run(
            [sys.executable, "-u", str(ROOT / "src" / "train_one_chunk.py"),
             str(config_path), str(this_chunk)],
            capture_output=True, text=True)
        print(proc.stdout[-4000:], flush=True)
        if proc.returncode != 0:
            consecutive_failures += 1
            print(f"  [{name}] CHUNK FAILED (exit {proc.returncode}, "
                  f"{consecutive_failures}/{MAX_RETRIES_PER_CHUNK} consecutive):\n"
                  f"{proc.stderr[-2000:]}", flush=True)
            if consecutive_failures >= MAX_RETRIES_PER_CHUNK:
                print(f"  [{name}] giving up after {consecutive_failures} consecutive failures -- "
                      f"the last GOOD checkpoint is kept (train.py raises before saving a "
                      f"degenerate step).", flush=True)
                break
            continue
        consecutive_failures = 0
        if ckpt_path.exists() and hist_path.exists():
            h = json.loads(hist_path.read_text())
            if h and h[-1]["step"] >= total_steps - 1:
                print(f"  [{name}] reached target step {total_steps}, run complete", flush=True)
                break

    history = json.loads(hist_path.read_text()) if hist_path.exists() else []
    return dict(model_cfg=dataclasses.asdict(mcfg), train_cfg=dataclasses.asdict(tcfg),
                history=history, elapsed_s=time.time() - t0)
