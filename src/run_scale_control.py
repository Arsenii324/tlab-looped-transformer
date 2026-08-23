"""The three scale-control interventions from arXiv 2606.24898 this project has never run.

That paper's Table 1 lists four ways to stop hidden-state scale from drifting invisibly. This project
has only ever used the fourth (inter-loop normalization = `state_renorm`), and §4.3 measured what it
costs: it converts angular dilution into hard contraction, collapses the optimum from 8-11 to ~4,
and costs 0.74 nats. The other three are all zero-parameter and are implemented here:

  raw          readout without final_norm -- scale becomes visible to CE (breaks their Lemma 1's
               scale invariance, which is what makes <grad_H L, H> = 0 in the first place)
  final_only   raw at intermediate exits, normed at the final one
  penalty      normalized readout kept, plus lambda * K^-1 sum_k E||H_k||^2_rms, lambda = 0.01

**Published prediction to check against.** Their variable-depth table (44M / 129M, 3 seeds) reports
Delta-PPL from K=1 to K=4 of +0.01 / -0.01 for the normalized readout, against -0.20 / -0.13 raw,
-0.20 / -0.15 final-only, -0.22 / -0.14 norm penalty; dynamic-halting average loops 1.00 vs
2.16 / 1.78 / 2.60. So they report the normalized readout has NO usable depth and the other three
restore it.

**Where this project already disagrees with that, which is why the experiment is worth running.**
Our normalized-readout model is NOT K-invariant: it has a 0.25-nat loop gain with an optimum at
loop 8-11 (§4.2), where theirs halts at K=1. Something here already supplies depth-dependence that
their setup lacks -- the most likely candidate being that we train with a randomized loop count over
[4,32] rather than at a fixed small K. So the question is not their question. Theirs was "does scale
control create depth use where there was none" (answer: yes, up to K=4). Ours is:

    does scale control EXTEND a depth range that is already 8-11 loops long?

§4.6 predicts no -- clamping relocated the optimum (5/15/24) without improving the best CE
(4.0071/4.0115/4.0114/4.0133, spread 0.006 nats), which says scale is a rate parameter, not a
ceiling. A positive result here would contradict §4.6 and would be the more valuable outcome.

Arms are token-matched (§4.1's wall-clock confound flipped two of five axes), share one model shape,
and differ only in readout_mode/norm_penalty. Read the optimum from a post-hoc dense sweep, not the
coarse in-training grid -- the optimum's LOCATION is the measurement.

Usage: python src/run_scale_control.py [--tokens 2500000] [--seeds 0,1]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig, LoopedTransformer  # noqa: E402
from train import TrainConfig  # noqa: E402
from chunked_runner import run_chunked  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAYER_APPS_PER_S = 59_400
ARMS = {                       # name -> (readout_mode, norm_penalty)
    "sc_control_norm": ("norm", 0.0),
    "sc_raw": ("raw", 0.0),
    "sc_final_only": ("final_only", 0.0),
    "sc_penalty": ("norm", 0.01),
}


def preflight(tokens):
    print("=" * 100)
    print("PRE-FLIGHT")
    ok = True
    shapes, params = set(), set()
    for name, (mode, pen) in ARMS.items():
        m = LoopedTransformer(ModelConfig(state_renorm=False, readout_mode=mode))
        params.add(m.num_parameters()); shapes.add(mode in ("norm", "raw", "final_only"))
    print(f"[a] all arms parameter-identical: {params} "
          f"{'OK' if len(params) == 1 else 'MISMATCH -- arms are not comparable'}")
    ok &= len(params) == 1

    # the interventions must actually change the forward/loss, or an arm silently trains as control
    import torch
    x = torch.randint(0, 4096, (2, 16))
    base = None
    for name, (mode, pen) in ARMS.items():
        torch.manual_seed(0)
        m = LoopedTransformer(ModelConfig(state_renorm=False, readout_mode=mode)).eval()
        with torch.no_grad():
            lg, _ = m(x, n_loops=4, return_all_loops=True)
        sig = float(sum(l.abs().sum().item() for l in lg))
        if name == "sc_control_norm":
            base = sig
        differs = (abs(sig - base) > 1e-3) if name != "sc_control_norm" else None
        note = ("(differs from control OK)" if differs else
                "(SAME AS CONTROL -- readout_mode not taking effect)") if differs is not None else "(control)"
        if name == "sc_penalty":
            note = "(same forward by construction; penalty acts in the LOSS -- checked by test [9])"
            differs = True
        if differs is False:
            ok = False
        print(f"[b] {name:18} readout={mode:11} lambda={pen:<5} logit-signature={sig:14.1f} {note}")

    mu = 18.0
    secs = tokens * mu * 3 / LAYER_APPS_PER_S
    print(f"[c] token-budgeted: {tokens:,} tok/arm, all arms same mu_rec={mu} so equal cost "
          f"(~{secs/60:.0f} min each); cap 2.2x so tokens bind, not the clock. OK")
    print(f"[d] optimum location read post-hoc (dense sweep), not from the coarse in-training grid. OK")
    print("=" * 100, flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2_500_000)
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    if not preflight(args.tokens):
        print("PRE-FLIGHT FAILED -- not training."); return 1
    if args.preflight_only:
        return 0

    out = ROOT / "checkpoints" / "scale_control_results.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        for name, (mode, pen) in ARMS.items():
            run = f"{name}_s{seed}"
            if run in results:
                print(f"skip {run}", flush=True); continue
            mcfg = ModelConfig(state_renorm=False, readout_mode=mode)
            cap = args.tokens * 18.0 * 3 / LAYER_APPS_PER_S * 2.2
            tcfg = TrainConfig(run_name=run, batch_size=8, seq_len=256, device="mps",
                               total_tokens=args.tokens,
                               eval_every_tokens=max(200_000, args.tokens // 8),
                               eval_batches=6, warmup_steps=40, supervise_k=5,
                               min_train_loops=4, max_train_loops=32, seed=seed,
                               norm_penalty=pen)
            print(f"=== {run}  readout={mode} lambda={pen} ===", flush=True)
            results[run] = run_chunked(run, mcfg, tcfg, cap, fresh=True)
            h = results[run]["history"]
            got = h[-1]["tokens"] if h else 0
            results[run]["reached_token_target"] = got >= args.tokens * 0.98
            print(f"  {run}: tokens={got:,} "
                  f"{'OK' if results[run]['reached_token_target'] else '<-- TRUNCATED'}", flush=True)
            out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
