"""Stochastic exploration during loops -- the one direction the task names that had no arm.

The brief lists "exploration во время лупов" as a suggested mechanism and cites EBT, whose loop IS
Langevin descent on an energy. Nothing in this project had tested it.

The motivation here is not the analogy but this project's own measurement. §4.3 found consecutive
loop increments aligned at cos(du_t, du_{t-1}) -> 0.9999 -- the state travels an almost perfectly
straight ray. That is MAXIMAL coherence: every loop pushes in the direction the previous one did, and
nothing ever deviates. If depth is wasted because the trajectory commits early and never explores,
then breaking that coherence is precisely the intervention the geometry suggests.

Noise is relative to each token's ||h|| (the readout is scale-invariant, so absolute noise would
vanish from view exactly as ||h|| grows 18x), annealed 1/sqrt(t) by default, and applied at TRAIN
TIME ONLY so every evaluation in this report stays deterministic and comparable. Zero parameters.

Swept, not A/B'd (METHODS.md rule 2): sigma = 0 is bit-exactly the existing model, so the control is
inside the sweep. Predictions, recorded before running:
  - if the ray's coherence is what wastes depth, some sigma > 0 raises loop gain and/or moves the
    optimum deeper;
  - if the coherence is load-bearing (the drift IS the computation, as Huginn's "sliders" reading
    suggests), noise degrades monotonically in sigma and the answer is that exploration hurts;
  - a third outcome is live: absolute CE improves as a regulariser while loop gain does not, which
    would be the third instance of this project's recurring metric-vs-mechanism split.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig, LoopedTransformer  # noqa: E402
from train import TrainConfig  # noqa: E402
from chunked_runner import run_chunked  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SIGMAS = (0.0, 0.05, 0.15, 0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2_500_000)
    ap.add_argument("--seeds", type=str, default="0")
    args = ap.parse_args()
    print("PRE-FLIGHT")
    ps = set()
    for s in SIGMAS:
        m = LoopedTransformer(ModelConfig(state_renorm=False, explore_noise=s)); ps.add(m.num_parameters())
    assert len(ps) == 1, f"arms not parameter-matched: {ps}"
    print(f"  all sigmas parameter-identical at {ps.pop():,} (noise adds no parameters)")
    print(f"  sigma=0.0 is bit-exactly the existing model -> control is inside the sweep")
    print(f"  noise is TRAIN-ONLY; eval stays deterministic so numbers remain comparable\n", flush=True)

    out = ROOT / "checkpoints" / "explore_results.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for seed in [int(x) for x in args.seeds.split(",")]:
        for sig in SIGMAS:
            run = f"expl_s{sig}_seed{seed}"
            if run in res:
                print(f"skip {run}", flush=True); continue
            mcfg = ModelConfig(state_renorm=False, explore_noise=sig)
            cap = args.tokens * 18.0 * 3 / 59_400 * 2.2
            tcfg = TrainConfig(run_name=run, batch_size=8, seq_len=256, device="mps",
                               total_tokens=args.tokens,
                               eval_every_tokens=max(200_000, args.tokens // 8),
                               eval_batches=6, warmup_steps=40, supervise_k=5,
                               min_train_loops=4, max_train_loops=32, seed=seed)
            print(f"=== {run} (sigma={sig}) ===", flush=True)
            res[run] = run_chunked(run, mcfg, tcfg, cap, fresh=True)
            h = res[run]["history"]
            if h:
                c = h[-1]["val_curve"]; b = min(c, key=c.get)
                print(f"  {run}: tokens={h[-1]['tokens']:,} best r={b} CE={c[b]:.4f} "
                      f"gain={c['1']-c[b]:.4f}", flush=True)
            out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
