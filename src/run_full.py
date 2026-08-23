"""Full-budget run of a chosen config, selected from the screening sweep's results.

Not run automatically -- called with the winning arm's name once screening concludes and the
per-loop val curves are actually read (checkpoints/screening_results.json), per PLAN.md sec 5: "at
most 2-3 configs reach full budget, chosen and logged with reasoning, not run because they were next
in a queue."

Runs via chunked_runner.run_chunked -- NOT a single continuous run() call. LOG.md 2026-08-13 01:52/
02:15: sustained MPS load silently corrupted output after ~700s in one process; a full-budget run is
necessarily much longer than that, so it needs the same chunked-subprocess protection screening uses,
not a bespoke unprotected path (an earlier version of this file had exactly that gap, caught and
fixed before ever running it for real).

Token budget is set from whatever wall-clock remains, not from the original 100M ceiling, which is
not reachable this session (LOG.md: Kaggle quota exhausted, local MPS throughput measured at
~1000-1300 tok/s with the bounded-subset-supervision fix). Pass --seconds explicitly rather than
trusting a default, since how much time remains is a fact about the clock, not the code.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig  # noqa: E402
from train import TrainConfig  # noqa: E402
from chunked_runner import run_chunked  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_arm_config(results_path: pathlib.Path, arm_name: str) -> tuple[ModelConfig, int, int]:
    """Returns (model_cfg, min_train_loops, max_train_loops). The loop-count schedule is a TrainConfig
    field, not a ModelConfig field -- for most arms it's irrelevant (they differ from center only in
    model_cfg, and every screening arm's TrainConfig used the same common min/max_train_loops=4,32),
    but fixed_loops16 differs from center ONLY in this field (its model_cfg is literally center's
    model_cfg, unchanged). A prior version of this function returned only model_cfg and this file
    separately hardcoded min_train_loops=4, max_train_loops=32 below regardless of arm_name -- so a
    full run of "fixed_loops16" silently trained with the default randomized 4-32 schedule instead of
    the fixed-16 schedule that arm actually tested in screening. Caught by reading the checkpoint's own
    saved train_cfg after the run completed, not by inspection beforehand -- see LOG.md 2026-08-13
    ~09:55. checkpoints/full_fixed_loops16/ (built before this fix) is kept and reinterpreted in
    report.md as an extra center-config data point, not discarded, since the run itself is valid, only
    mislabeled.
    """
    results = json.loads(results_path.read_text())
    if arm_name not in results:
        raise KeyError(f"{arm_name} not in {list(results)}")
    mc = results[arm_name]["model_cfg"]
    tc = results[arm_name]["train_cfg"]
    return ModelConfig(**mc), tc["min_train_loops"], tc["max_train_loops"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_name", type=str, help="e.g. center, truncate8, no_state_renorm ...")
    ap.add_argument("--results", type=str,
                     default=str(ROOT / "checkpoints" / "screening_results.json"))
    ap.add_argument("--seconds", type=float, required=True,
                     help="wall-clock budget for this run -- pass explicitly, based on actual "
                          "remaining time, not a guess")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tok-per-s-estimate", type=float, default=1100)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--resume-existing", action="store_true",
                     help="continue an already-started full run instead of starting fresh")
    args = ap.parse_args()

    model_cfg, min_train_loops, max_train_loops = load_arm_config(pathlib.Path(args.results), args.arm_name)
    total_tokens = int(args.seconds * args.tok_per_s_estimate)  # upper bound; the chunk loop's own
    # wall-clock budget is authoritative, this only sizes the LR schedule / total_steps denominator.
    run_name = args.run_name or f"full_{args.arm_name}"

    # eval_every_tokens MUST be comfortably smaller than what one 240s chunk can train, or no
    # checkpoint is ever saved within a chunk's budget and chunking provides zero continuity --
    # exactly what happened on the first attempt at this run: eval_every_steps worked out to 386
    # against ~141 steps/chunk, so four chunks (~970s) ran, corrupted GPU-driver-wise or not, and
    # were discarded on every fresh-start subprocess boundary with nothing ever checkpointed.
    # Target ~8 checkpoint opportunities per chunk (well under half the chunk's step budget) using
    # the same measured throughput this file already estimates total_tokens from.
    from chunked_runner import CHUNK_SECONDS
    steps_per_chunk_est = CHUNK_SECONDS / (args.batch_size * 256 / args.tok_per_s_estimate)
    eval_every_tokens = max(50_000, int(steps_per_chunk_est / 8) * args.batch_size * 256)
    train_cfg = TrainConfig(
        run_name=run_name, batch_size=args.batch_size, seq_len=256,
        device="mps", total_tokens=total_tokens, eval_every_tokens=eval_every_tokens,
        eval_batches=6, warmup_steps=150, supervise_k=5,
        min_train_loops=min_train_loops, max_train_loops=max_train_loops,
    )
    print(f"eval_every_tokens={eval_every_tokens} (~{eval_every_tokens // (args.batch_size*256)} "
          f"steps, vs an estimated {steps_per_chunk_est:.0f} steps/chunk)")

    print(f"FULL RUN: arm={args.arm_name} run_name={run_name} model_cfg={dataclasses.asdict(model_cfg)}")
    print(f"budget: {args.seconds:.0f}s (~{args.seconds/3600:.2f}h), "
          f"token estimate {total_tokens/1e6:.1f}M at {args.tok_per_s_estimate:.0f} tok/s, "
          f"total_steps={total_tokens // (args.batch_size * 256)}")

    result = run_chunked(run_name, model_cfg, train_cfg, args.seconds, fresh=not args.resume_existing)
    hist = result["history"]
    print(f"done in {result['elapsed_s']:.0f}s, {len(hist)} eval points. "
          f"Final eval: {hist[-1] if hist else 'none logged'}")


if __name__ == "__main__":
    main()
