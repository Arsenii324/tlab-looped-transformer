"""Local screening sweep: one axis at a time from the center config, reduced token budget.

Each arm runs as a SEQUENCE OF SHORT SUBPROCESSES, each resuming from the last checkpoint, rather
than one continuous process. This is not incidental structure -- LOG.md 2026-08-13 01:52 records
sustained MPS load silently corrupting output (all-zero forward passes, no exception) after ~700s in
a single process on this hardware; CHUNK_SECONDS is kept well under that with margin. A fresh
subprocess means a fresh Metal context each chunk. train.py's degenerate-output check (loss==0.0,
NaN/Inf, or zero state norm) is the second layer of defense -- if the chunking mitigation is
insufficient, that check raises instead of silently producing garbage, and this driver script stops
the arm rather than resuming past a corrupted checkpoint.

Token/time budget is REVISED DOWN from the original plan, honestly, not quietly: local MPS
throughput with the bounded-subset supervision fix is ~1000-1300 tok/s (measured, LOG.md), and
Kaggle's weekly GPU quota is exhausted. batch_size is kept modest (8, not 32) given the machine has
been under real memory pressure -- a smaller footprint is worth some throughput.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig  # noqa: E402
from train import TrainConfig  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "checkpoints" / "screening_results.json"
CONFIGS_DIR = ROOT / "checkpoints" / "_arm_configs"
SECONDS_PER_ARM = 18 * 60          # ~18 min/arm x 7 = 126 min ~= 2.1h total
CHUNK_SECONDS = 240                 # well under the measured ~700s failure window (2.9x margin)
MAX_RETRIES_PER_CHUNK = 3           # failures observed intermittent, not permanent (LOG.md
# 2026-08-13 02:05: a chunk failed at step 1 with loss==0.0, the very next fresh subprocess
# succeeded normally) -- retry before giving up on the whole arm.


def tokens_for_budget(seconds, tok_per_s_estimate=1100):
    return int(seconds * tok_per_s_estimate)


def run_arm_chunked(name: str, mcfg: ModelConfig, tcfg: TrainConfig, arm_seconds: float) -> dict:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIGS_DIR / f"{name}.json"
    config_path.write_text(json.dumps(dict(model_cfg=dataclasses.asdict(mcfg),
                                            train_cfg=dataclasses.asdict(tcfg))))

    ckpt_path = pathlib.Path(tcfg.ckpt_dir) / name / "last.pt"
    if ckpt_path.exists():
        ckpt_path.unlink()  # fresh start for this arm; a stale checkpoint from a prior attempt
        # under a DIFFERENT config would silently resume into the wrong run otherwise.

    tokens_per_step = tcfg.batch_size * tcfg.seq_len
    total_steps = tcfg.total_tokens // tokens_per_step

    arm_t0 = time.time()
    chunk_i = 0
    consecutive_failures = 0
    while time.time() - arm_t0 < arm_seconds:
        chunk_i += 1
        remaining = arm_seconds - (time.time() - arm_t0)
        this_chunk = min(CHUNK_SECONDS, remaining)
        if this_chunk < 15:  # not enough time left for a meaningful chunk
            break
        print(f"  [{name}] chunk {chunk_i}, budget {this_chunk:.0f}s "
              f"(arm elapsed {time.time()-arm_t0:.0f}/{arm_seconds:.0f}s)", flush=True)
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
                print(f"  [{name}] giving up on this arm after {consecutive_failures} "
                      f"consecutive failures -- the checkpoint at the last GOOD chunk is kept, "
                      f"not the failed one (train.py raises before saving on a degenerate step).",
                      flush=True)
                break
            continue  # retry: re-check elapsed time, try another chunk (same resume point)
        consecutive_failures = 0
        # done for this arm if the checkpoint has reached the token budget
        if ckpt_path.exists():
            saved_step = json.loads((ROOT / "checkpoints" / f"{name}_history.json").read_text()) \
                if (ROOT / "checkpoints" / f"{name}_history.json").exists() else []
            if saved_step and saved_step[-1]["step"] >= total_steps - 1:
                print(f"  [{name}] reached target step {total_steps}, arm complete", flush=True)
                break

    hist_path = ROOT / "checkpoints" / f"{name}_history.json"
    history = json.loads(hist_path.read_text()) if hist_path.exists() else []
    return dict(model_cfg=dataclasses.asdict(mcfg), train_cfg=dataclasses.asdict(tcfg),
                history=history, elapsed_s=time.time() - arm_t0)


def main():
    center = ModelConfig()
    budget_tokens = tokens_for_budget(SECONDS_PER_ARM)
    common = dict(batch_size=8, seq_len=256, device="mps", total_tokens=budget_tokens,
                  eval_every_tokens=budget_tokens // 12,  # frequent checkpoints -> fine resume granularity
                  eval_batches=6, warmup_steps=40, supervise_k=5,
                  min_train_loops=4, max_train_loops=32)

    arms = [
        ("center", center, TrainConfig(run_name="center", **common)),
        ("truncate8", dataclasses.replace(center, truncate_bptt=8),
         TrainConfig(run_name="truncate8", **common)),
        ("no_state_renorm", dataclasses.replace(center, state_renorm=False),
         TrainConfig(run_name="no_state_renorm", **common)),
        ("inject_concat", dataclasses.replace(center, inject_mode="concat"),
         TrainConfig(run_name="inject_concat", **common)),
        ("inject_none", dataclasses.replace(center, inject_mode="none"),
         TrainConfig(run_name="inject_none", **common)),
        ("no_depth_init", dataclasses.replace(center, depth_init=False),
         TrainConfig(run_name="no_depth_init", **common)),
        ("fixed_loops16", center,
         TrainConfig(run_name="fixed_loops16",
                     **{**common, "min_train_loops": 16, "max_train_loops": 16})),
    ]

    results = {}
    sweep_t0 = time.time()
    for name, mcfg, tcfg in arms:
        print(f"\n########## ARM {name} (budget {budget_tokens/1e6:.2f}M tokens, "
              f"~{SECONDS_PER_ARM/60:.0f} min, chunked at {CHUNK_SECONDS}s) ##########", flush=True)
        results[name] = run_arm_chunked(name, mcfg, tcfg, SECONDS_PER_ARM)
        RESULTS_PATH.write_text(json.dumps(results, indent=2))
        print(f"########## ARM {name} done, {results[name]['elapsed_s']:.0f}s, "
              f"{len(results[name]['history'])} eval points, "
              f"total sweep elapsed {time.time()-sweep_t0:.0f}s ##########", flush=True)

    print("\n=== SCREENING SUMMARY ===")
    for name, r in results.items():
        if not r["history"]:
            print(f"  {name}: NO EVAL LOGGED")
            continue
        last = r["history"][-1]["val_curve"]
        best_r = min(last, key=last.get)
        print(f"  {name:<18} best: r={best_r:>3} CE={last[best_r]:.4f}  "
              f"r1={last.get('1', last.get(1, float('nan'))):.4f}  full curve={last}")


if __name__ == "__main__":
    main()
