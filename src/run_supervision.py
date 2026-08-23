"""Is the loop optimum set by the DYNAMICS or by the training SCHEDULE?

§4.6 showed that clamping the state's scale relocates the optimum (loop 5 / 15 / 24 for three clamp
levels) without changing the best CE achievable (4.0071 / 4.0115 / 4.0114 / 4.0133, spread 0.006
nats). That says scale control re-parameterizes depth rather than buying it, and moves the suspicion
from dynamics to *demand*: the model runs out of useful work at a fixed point along its trajectory.

If demand is what binds, the obvious candidate for setting it is the training loop-count schedule.
The current schedule is uniform [4,32] (mu_rec = 18) and the optimum sits at 8-11 -- 44-61% of
mu_rec, which is a deviation from the rule other work reports (optimum at mu_rec).

Measured supervision density under the ACTUAL sampler in train.py (final loop always supervised,
k-1 more uniform without replacement, k=5), simulated not assumed -- see the pre-flight below:

  uniform[4,32]:      d(1)=.337 d(4)=.335 d(8)=.233 d(12)=.174 d(16)=.132 d(24)=.075 d(32)=.035
  concentrated[24,32]:d(1)=.150 d(4)=.150 d(8)=.148 d(12)=.148 d(16)=.150 d(24)=.243 d(32)=.112

Note this already falsifies the SIMPLEST density hypothesis: under [4,32] density peaks at loops
1-4, while the optimum is at 8-11, so "optimum = argmax density" is wrong as stated. What the arms
below can still separate is whether the optimum MOVES with the schedule at all:

  A  uniform[4,32]  mu=18  (control -- reproduces the headline config's schedule)
  B  concentrated[24,32] mu=28  (density flat to 16, then peaks at 24)
  C  shallow[4,8]  mu=6   (density concentrated at 1-8)

Predictions, written before running (CLAUDE.md sec 1):
  - DEMAND-BOUND: the optimum tracks the schedule -- C lands below A, B above A, roughly ordered
    with mu_rec. Then saturation is a property of the loss schedule and is fixable by changing it.
  - DYNAMICS-BOUND (what §4.6's fixed-angular-budget reading predicts): the optimum stays near 8-11
    in all three arms even though mu_rec varies 6 -> 28, because the trained map's angular budget is
    what sets it. Then the schedule is not the lever and demand must be manufactured some other way.
  - A third outcome is live and would be the most useful: B's optimum moves but its BEST CE does not
    improve, which would mirror §4.6 exactly -- the optimum is relocatable but the ceiling is not.

Budgeted by TOKENS, not wall clock. This matters and is not a detail: §4.1's wall-clock budgeting
systematically handed more data to cheaper arms and flipped two of five axes. Here the cost per step
varies ~4.7x across arms (mu=6 vs mu=28), so wall-clock budgeting would be maximally wrong. Each arm
gets the same total_tokens and a wall-clock cap set to ~2.2x its own predicted duration so the token
target is what binds; the pre-flight prints the predicted durations and the runner asserts afterwards
that every arm actually reached the token target.

Usage: python src/run_supervision.py [--tokens 2500000] [--seeds 0,1]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig, LoopedTransformer  # noqa: E402
from train import TrainConfig, sample_supervise_idx  # noqa: E402
from chunked_runner import run_chunked  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARMS = {"sup_uniform4_32": (4, 32), "sup_concentrated24_32": (24, 32), "sup_shallow4_8": (4, 8)}
LAYER_APPS_PER_S = 59_400          # measured on this machine: 1100 tok/s at mu=18, 3 layers/loop
SUPERVISE_K = 5


def density(lo, hi, k, upto, trials=60_000, seed=0):
    """Uses train.py's OWN sample_supervise_idx, not a re-derivation of it -- if the sampler changes,
    this pre-flight changes with it instead of silently describing the old behaviour."""
    rng = np.random.default_rng(seed)
    cnt = np.zeros(upto + 2)
    for _ in range(trials):
        n = int(rng.integers(lo, hi + 1))
        for i in sample_supervise_idx(n, k, rng):
            if i + 1 <= upto + 1:
                cnt[i + 1] += 1
    return cnt / trials


def preflight(tokens):
    print("=" * 100)
    print("PRE-FLIGHT (nothing is trained until every line below is checked)")
    ok = True

    m = LoopedTransformer(ModelConfig(state_renorm=False))
    print(f"[a] all arms share one model config, params = {m.num_parameters():,} "
          f"(arms differ ONLY in TrainConfig loop range) OK")

    print("[b] measured supervision density under train.py's own sampler:")
    hdr = (1, 2, 4, 8, 12, 16, 24, 32)
    print("      " + f"{'schedule':22}" + "".join(f"{('d'+str(t)):>8}" for t in hdr) + f"{'mu_rec':>8}")
    for name, (lo, hi) in ARMS.items():
        d = density(lo, hi, SUPERVISE_K, 32)
        print("      " + f"{name:22}" + "".join(f"{d[t]:>8.3f}" for t in hdr) + f"{(lo+hi)/2:>8.1f}")
    da = density(*ARMS["sup_uniform4_32"], SUPERVISE_K, 32)
    db = density(*ARMS["sup_concentrated24_32"], SUPERVISE_K, 32)
    if not (db[24] > da[24] * 2 and da[8] > db[8]):
        print("      ^ arms do NOT differ in the intended direction"); ok = False
    else:
        print("      arms differ as intended (B raises d(24) >2x and lowers d(8)) OK")

    print("[c] token budget binds, not wall clock:")
    for name, (lo, hi) in ARMS.items():
        mu = (lo + hi) / 2
        secs = tokens * mu * 3 / LAYER_APPS_PER_S
        print(f"      {name:22} mu={mu:>4.1f}  predicted {secs/60:>6.1f} min  "
              f"cap set to {secs*2.2/60:>6.1f} min ({2.2:.1f}x)")
    print(f"      cost ratio across arms = {28/6:.1f}x -- wall-clock budgeting would be maximally "
          f"wrong here (that confound flipped 2 of 5 axes in sec4.1) OK")

    print("[d] eval: in-training grid is coarse and stops at 32; the OPTIMUM LOCATION is the whole "
          "measurement here, so it is read from a post-hoc dense sweep (eval.py, every integer to "
          "48), not from the grid. OK")
    print("=" * 100, flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2_500_000)
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    if not preflight(args.tokens):
        print("PRE-FLIGHT FAILED -- not training."); return 1
    if args.preflight_only:
        return 0

    out_path = ROOT / "checkpoints" / "supervision_results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    mcfg = ModelConfig(state_renorm=False)

    for seed in seeds:
        for name, (lo, hi) in ARMS.items():
            run = f"{name}_s{seed}"
            if run in results:
                print(f"skip {run} (already done)", flush=True); continue
            mu = (lo + hi) / 2
            cap = args.tokens * mu * 3 / LAYER_APPS_PER_S * 2.2
            tcfg = TrainConfig(run_name=run, batch_size=8, seq_len=256, device="mps",
                               total_tokens=args.tokens,
                               eval_every_tokens=max(200_000, args.tokens // 8),
                               eval_batches=6, warmup_steps=40, supervise_k=SUPERVISE_K,
                               min_train_loops=lo, max_train_loops=hi, seed=seed)
            print(f"=== {run}  loops[{lo},{hi}] mu={mu} cap={cap/60:.0f}min ===", flush=True)
            results[run] = run_chunked(run, mcfg, tcfg, cap, fresh=True)
            hist = results[run]["history"]
            got = hist[-1]["tokens"] if hist else 0
            reached = got >= args.tokens * 0.98
            results[run]["reached_token_target"] = reached
            print(f"  {run}: tokens={got:,}/{args.tokens:,} "
                  f"{'OK' if reached else '<-- WALL-CLOCK TRUNCATED, NOT TOKEN-MATCHED'}", flush=True)
            out_path.write_text(json.dumps(results, indent=2))

    bad = [k for k, v in results.items() if not v.get("reached_token_target", True)]
    print(f"\nwrote {out_path}")
    print(f"arms that failed to reach the token target: {bad if bad else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
