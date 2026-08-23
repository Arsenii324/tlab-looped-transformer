"""Item 5: eps = lambda/(N*sqrt(L)) residual scaling vs this project's depth_init.

depth_init scales o_proj/down_proj ONCE at init by 1/sqrt(2*n_loop_eff); residual_scale is a
persistent multiplier with the constant tied to the actual loop count and layers-per-loop. The
claimed benefit is LR transfer across loop counts, which -- if real -- predicts the fixed-vs-random
loop-count difference (§4.1's `fixed_loops16`, already token-corrected to ~null) should vanish.

Recorded objection, same as the model.py docstring: eps*N = O(1) bounds total displacement from h0
regardless of N, which is §4.3's dilution arriving by another road. §4.6 showed the achievable CE is
invariant to how fast the angular budget is spent, so this may reproduce the ceiling with a cleaner
LR story rather than lift it. Both outcomes are informative; that is why it is worth one screen.

Token-budgeted, arms share one model shape and differ only in the init/scaling rule.
"""
from __future__ import annotations
import argparse, dataclasses, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig, LoopedTransformer  # noqa: E402
from train import TrainConfig  # noqa: E402
from chunked_runner import run_chunked  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAYER_APPS_PER_S = 59_400
# n_loop_eff=18, NOT the config default of 24. The sampler is U[4,32], whose mean is 18; 24 was
# never the mean of anything here. Huginn's own convention keys the init off the MEAN recurrence
# (verified from their LaTeX), so eps = lambda/(N*sqrt(L)) must use N=18. Caught by audit before
# this arm ran -- the same constant is wrong in the frozen `depth_init` runs, where it is left
# alone because changing it would invalidate comparability with everything already completed.
ARMS = {"rs_depth_init":  dict(depth_init=True,  residual_scale=None, n_loop_eff=18),
        "rs_lambda1":     dict(depth_init=False, residual_scale=1.0,  n_loop_eff=18),
        "rs_lambda2":     dict(depth_init=False, residual_scale=2.0,  n_loop_eff=18)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2_500_000)
    ap.add_argument("--seeds", type=str, default="0,1")
    args = ap.parse_args()
    print("PRE-FLIGHT")
    params = set()
    for n, kw in ARMS.items():
        m = LoopedTransformer(ModelConfig(state_renorm=False, **kw))
        params.add(m.num_parameters())
        eps = getattr(m, "_residual_eps", None)
        print(f"  {n:16} {kw}  params={m.num_parameters():,} eps={eps}")
    assert len(params) == 1, f"arms not parameter-matched: {params}"
    print(f"  all arms parameter-matched at {params.pop():,}; token-budgeted at {args.tokens:,}\n",
          flush=True)

    out = ROOT / "checkpoints" / "residual_scale_results.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        for name, kw in ARMS.items():
            run = f"{name}_s{seed}"
            if run in res:
                print(f"skip {run}", flush=True); continue
            mcfg = ModelConfig(state_renorm=False, **kw)
            cap = args.tokens * 18.0 * 3 / LAYER_APPS_PER_S * 2.2
            tcfg = TrainConfig(run_name=run, batch_size=8, seq_len=256, device="mps",
                               total_tokens=args.tokens,
                               eval_every_tokens=max(200_000, args.tokens // 8),
                               eval_batches=6, warmup_steps=40, supervise_k=5,
                               min_train_loops=4, max_train_loops=32, seed=seed)
            print(f"=== {run} ===", flush=True)
            res[run] = run_chunked(run, mcfg, tcfg, cap, fresh=True)
            h = res[run]["history"]
            got = h[-1]["tokens"] if h else 0
            res[run]["reached_token_target"] = got >= args.tokens * 0.98
            print(f"  {run}: tokens={got:,} "
                  f"{'OK' if res[run]['reached_token_target'] else 'TRUNCATED'}", flush=True)
            out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
