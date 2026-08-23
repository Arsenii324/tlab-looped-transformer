"""Second-seed check on the winning axis (report.md sec 8 item 5): `center` and `no_state_renorm`,
re-run at seed=1 (screening used seed=0 for every arm), same screening-scale budget (18 min/arm),
same chunked-subprocess safety as run_screening.py -- reuses chunked_runner.py directly rather than
duplicating it, unlike run_screening.py's own copy (left alone there since it already worked; no
reason to carry the duplication forward into a new script).

Answers: is `no_state_renorm`'s -0.746 nat screening margin over `center` seed noise, or real? The
screening result was never in doubt as a same-seed comparison (matched across the two arms) -- this
is about whether a DIFFERENT seed reproduces the same direction and a similar magnitude.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig  # noqa: E402
from train import TrainConfig  # noqa: E402
from chunked_runner import run_chunked  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SECONDS_PER_ARM = 18 * 60
SEED = 1


def tokens_for_budget(seconds, tok_per_s_estimate=1100):
    return int(seconds * tok_per_s_estimate)


def main():
    center = ModelConfig()
    budget_tokens = tokens_for_budget(SECONDS_PER_ARM)
    common = dict(batch_size=8, seq_len=256, device="mps", total_tokens=budget_tokens,
                  eval_every_tokens=budget_tokens // 12, eval_batches=6, warmup_steps=40,
                  supervise_k=5, min_train_loops=4, max_train_loops=32, seed=SEED)

    arms = [
        ("center_seed1", center, TrainConfig(run_name="center_seed1", **common)),
        ("no_state_renorm_seed1", dataclasses.replace(center, state_renorm=False),
         TrainConfig(run_name="no_state_renorm_seed1", **common)),
    ]

    results = {}
    for name, mcfg, tcfg in arms:
        print(f"=== {name} (seed={SEED}) ===", flush=True)
        results[name] = run_chunked(name, mcfg, tcfg, SECONDS_PER_ARM, fresh=True)
        hist = results[name]["history"]
        if hist:
            last = hist[-1]["val_curve"]
            best_r = min(last, key=last.get)
            print(f"  {name}: best r={best_r} CE={last[best_r]:.4f} r1={last.get(1, float('nan')):.4f} "
                  f"tokens={hist[-1]['tokens']}", flush=True)

    out_path = ROOT / "checkpoints" / "second_seed_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
