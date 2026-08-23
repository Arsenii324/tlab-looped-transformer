"""Does a prelude/coda sandwich beat a flat loop at a FIXED 10M-parameter budget?

The field's answer to "what makes loops keep working" is increasingly topology: Huginn, Ouro and
Parcae all wrap the recurrent core in unshared prelude and coda layers, and the controlled
Ouro->Huginn transformation reports large gains from that envelope alone. This model had neither --
a pure flat loop with a learned h0.

The reason that is a real experiment here rather than a free win is arithmetic. One DecoderLayer at
H=448 is 2,409,568 params; a naive prelude+coda on top of the existing 3-layer block costs 13.88M
against a 10M ceiling. At 730M params a sandwich is nearly free. At 10M it must be paid for OUT OF
THE LOOP BLOCK -- the one thing loops multiply. So every arm below holds total layers at 3 and total
params at 9,065,056 (verified in-run, not assumed), and differs only in how those 3 layers are split
between "reused r times" and "run once":

    P0R3C0   flat, the current best config (control)
    P1R1C1   full sandwich, 1-layer recurrent core
    P1R2C0   prelude only, 2-layer core
    P0R2C1   coda only,    2-layer core

Loop ranges are scaled so every arm does the SAME layer applications per step (iso-depth), which is
the comparison that isn't rigged: flat R3 at [4,32] loops = 12-96 layer-applications, so R1 arms run
[12,96] and R2 arms run [6,48]. A sandwich arm therefore gets 3x MORE LOOPS for the same compute --
which is directly the axis the task scores, and is the strongest argument for the topology if it
works: a thinner recurrent core is a cheaper loop, so the budget buys more iterations of it.

Predictions, written before running (CLAUDE.md sec 1):
  - If the envelope hypothesis holds at this scale, P1R1C1 beats P0R3C0 on best-loop val CE, and its
    optimum sits at a substantially higher loop count (it has 3x the loops to work with).
  - If depth-per-loop is what actually matters, P0R3C0 wins and the sandwich arms saturate early
    despite more loops -- i.e. the envelope is a large-scale luxury and this is where it fails.
  - The prelude-only / coda-only arms separate which half of the envelope carries any effect. A
    coda is applied at EVERY readout (each loop is an exit), so it is the more expensive half at
    inference; if the gain is all prelude, that matters for the early-exit design.
This is a screening-scale result (~18 min/arm) and per CLAUDE.md it may motivate scaling a config
up; it may NOT by itself retire the axis.

Usage: python src/run_sandwich.py [--seconds-per-arm 1080]
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
from param_budget import total_params  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 0
BASE_LOOPS = (4, 32)          # the flat R3 arm's trained loop range
DEPTH_PER_STEP = (12, 96)     # = 3 * BASE_LOOPS; every arm matches this


def arms():
    """(name, layers_per_loop, n_prelude, n_coda). Loop range is derived, not written by hand, so
    the iso-depth property cannot drift out of sync with the layer counts."""
    for name, L, pre, coda in (("sand_P0R3C0", 3, 0, 0), ("sand_P1R1C1", 1, 1, 1),
                                ("sand_P1R2C0", 2, 1, 0), ("sand_P0R2C1", 2, 0, 1)):
        mcfg = ModelConfig(state_renorm=False, layers_per_loop=L, n_prelude=pre, n_coda=coda)
        lo, hi = DEPTH_PER_STEP[0] // L, DEPTH_PER_STEP[1] // L
        yield name, mcfg, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds-per-arm", type=int, default=18 * 60)
    ap.add_argument("--tok-per-s", type=int, default=1100)
    args = ap.parse_args()
    budget_tokens = args.seconds_per_arm * args.tok_per_s

    print("arm            layers  prelude  coda  loop_range  layer_apps/step  params")
    specs = list(arms())
    for name, mcfg, lo, hi in specs:
        b = total_params(mcfg.hidden_size, mcfg.n_heads, mcfg.n_kv_heads, mcfg.head_dim,
                         mcfg.intermediate_size, mcfg.vocab_size, mcfg.layers_per_loop,
                         mcfg.inject_mode, mcfg.n_prelude, mcfg.n_coda,
                         mcfg.state_renorm)
        real = LoopedTransformer(mcfg).num_parameters()
        want = b["total"] - b["exit_head_reserve"]
        assert real == want, f"{name}: model {real} != budget {want}"
        span = (lo * mcfg.layers_per_loop + mcfg.n_prelude + mcfg.n_coda,
                hi * mcfg.layers_per_loop + mcfg.n_prelude + mcfg.n_coda)
        print(f"{name:14s} {mcfg.layers_per_loop:>6} {mcfg.n_prelude:>8} {mcfg.n_coda:>5} "
              f"  [{lo:>2},{hi:>2}]      {span[0]:>3}-{span[1]:<3}      {real:,}")
    n_par = {LoopedTransformer(m).num_parameters() for _, m, _, _ in specs}
    assert len(n_par) == 1, f"arms are NOT parameter-matched: {n_par}"
    print(f"all arms parameter-matched at {n_par.pop():,}\n", flush=True)

    results = {}
    out_path = ROOT / "checkpoints" / "sandwich_results.json"
    if out_path.exists():
        results = json.loads(out_path.read_text())
        if results:
            print(f"resuming: {sorted(results)} already aggregated", flush=True)
    for name, mcfg, lo, hi in specs:
        # An arm whose history is already on disk is done -- recover it rather than retraining. The
        # aggregate json is written after each arm, so a crash in the driver can leave a fully
        # trained arm out of it (this is exactly what happened to sand_P1R2C0).
        hist_path = ROOT / "checkpoints" / f"{name}_history.json"
        if name not in results and hist_path.exists():
            # <name>_history.json is a bare LIST of eval records (train.py writes it); run_chunked
            # wraps it with the configs. Rebuild that same shape so a recovered arm is byte-compatible
            # with a freshly-run one downstream.
            recs = json.loads(hist_path.read_text())
            if recs and recs[-1]["tokens"] >= budget_tokens * 0.95:
                tc = TrainConfig(run_name=name, batch_size=8, seq_len=256, device="mps",
                                 total_tokens=budget_tokens, eval_every_tokens=budget_tokens // 10,
                                 eval_batches=6, warmup_steps=40, supervise_k=5,
                                 min_train_loops=lo, max_train_loops=hi, seed=SEED)
                results[name] = dict(model_cfg=dataclasses.asdict(mcfg),
                                      train_cfg=dataclasses.asdict(tc), history=recs,
                                      elapsed_s=None, recovered_from_disk=True)
                print(f"  {name}: recovered from disk ({recs[-1]['tokens']} tokens)", flush=True)
        if name in results:
            continue
        tcfg = TrainConfig(run_name=name, batch_size=8, seq_len=256, device="mps",
                           total_tokens=budget_tokens, eval_every_tokens=budget_tokens // 10,
                           eval_batches=6, warmup_steps=40, supervise_k=5,
                           min_train_loops=lo, max_train_loops=hi, seed=SEED)
        print(f"=== {name}  loops[{lo},{hi}] ===", flush=True)
        results[name] = run_chunked(name, mcfg, tcfg, args.seconds_per_arm, fresh=True)
        hist = results[name]["history"]
        if hist:
            last = hist[-1]["val_curve"]
            best = min(last, key=last.get)
            # .get, not [] -- val_curve keys are the fixed in-training eval grid, which need not
            # contain this arm's own min loop count (lo=6 for the R2 arms is not on the grid). The
            # first version indexed it directly and a KeyError killed the driver AFTER arm 3 had
            # finished training, losing only the aggregation. Same string-key class of bug as the
            # earlier `.get(1, ...)` one -- json dict keys are strings, and the grid is not the range.
            rmin = last.get(str(lo))
            print(f"  {name}: best r={best} CE={last[best]:.4f} "
                  f"r_min={f'{rmin:.4f}' if rmin is not None else f'n/a (r={lo} not on eval grid)'} "
                  f"tokens={hist[-1]['tokens']}", flush=True)
        (ROOT / "checkpoints" / "sandwich_results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {ROOT/'checkpoints'/'sandwich_results.json'}")


if __name__ == "__main__":
    main()
