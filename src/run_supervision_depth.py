"""Terminal-only loss vs dense per-loop supervision — the largest untested axis in this design.

Every run in this project supervises "final loop + up to 4 sampled others" (`supervise_k=5`). Two
independent papers say that costs perplexity:
  * Sharma & Vu (arXiv 2606.24898) Table 2, WikiText-103, K=4, matched, 3 seeds:
      terminal-only  5.40 (44M) / 4.88 (129M)   vs   per-loop  6.04 / 5.35
    i.e. terminal-only wins by 0.64 and 0.47 PPL.
  * LoopFormer, ~1B, 25B Pile tokens: Base-Loop (final loss) 10.91 vs Base-Loop-EE (early-exit
    supervision) 11.60 on Pile, 24.53 vs 26.55 on FineWeb-Edu.

But the same source says the cost of winning: their Table 14 has terminal-only at CE 5.52 for K=1
against 1.54 at K=4 — **the intermediate exits become unusable**. So this is not a free win, it is a
trade between absolute loss and exit elasticity, and this project cares about both (§4.7 needs
usable exits for any early-exit result at all).

This arm measures that trade at 9M on FineWeb, which neither paper does. `supervise_k=1` means only
the final loop is supervised (`sample_supervise_idx` always includes the last index).

Prediction, written before running: terminal-only improves best CE by O(0.1) nats AND collapses the
loop gain, because nothing trains the intermediate readouts. If both happen, the report gains a
measured trade-off. If CE does NOT improve, that contradicts two papers at this scale and is worth
more than the confirmation would be.

Everything else is held fixed: same model shape, same schedule U[4,32], same tokens, seeds 0 and 1.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig, LoopedTransformer  # noqa: E402
from train import TrainConfig  # noqa: E402
from chunked_runner import run_chunked  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAYER_APPS_PER_S = 59_400
ARMS = {"sd_dense_k5": 5, "sd_terminal_k1": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2_500_000)
    ap.add_argument("--seeds", type=str, default="0,1")
    args = ap.parse_args()
    print("PRE-FLIGHT")
    m = LoopedTransformer(ModelConfig(state_renorm=False))
    print(f"  one model shape, params={m.num_parameters():,}; arms differ ONLY in supervise_k")
    print(f"  arms: {ARMS}   schedule U[4,32] held fixed, token-budgeted at {args.tokens:,}\n", flush=True)

    out = ROOT / "checkpoints" / "supervision_depth_results.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for seed in [int(x) for x in args.seeds.split(",")]:
        for name, k in ARMS.items():
            run = f"{name}_s{seed}"
            if run in res:
                print(f"skip {run}", flush=True); continue
            mcfg = ModelConfig(state_renorm=False)
            cap = args.tokens * 18.0 * 3 / LAYER_APPS_PER_S * 2.2
            tcfg = TrainConfig(run_name=run, batch_size=8, seq_len=256, device="mps",
                               total_tokens=args.tokens,
                               eval_every_tokens=max(200_000, args.tokens // 8),
                               eval_batches=6, warmup_steps=40, supervise_k=k,
                               min_train_loops=4, max_train_loops=32, seed=seed)
            print(f"=== {run} (supervise_k={k}) ===", flush=True)
            res[run] = run_chunked(run, mcfg, tcfg, cap, fresh=True)
            h = res[run]["history"]
            got = h[-1]["tokens"] if h else 0
            res[run]["reached_token_target"] = got >= args.tokens * 0.98
            if h:
                c = h[-1]["val_curve"]; b = min(c, key=c.get)
                print(f"  {run}: tokens={got:,} best r={b} CE={c[b]:.4f} "
                      f"gain={c['1']-c[b]:.4f}", flush=True)
            out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
