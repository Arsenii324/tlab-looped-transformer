"""One path at different speeds, or different paths? The decisive test for §8.2.

§4.6 showed that clamping the state's RMS relocates the loop optimum (5 / 15 / 24 for three clamp
levels) without improving the best CE. The natural reading is that scale is a RATE parameter: every
token traverses the same trajectory, and the clamp only sets how fast. But there is a second reading
that fits the same aggregate numbers -- different clamps could put tokens on genuinely DIFFERENT
trajectories that happen to bottom out at similar average CE.

Those differ in a way that is measurable per token:

  RATE hypothesis   each token has one achievable minimum; changing the clamp changes WHERE along
                    the loop axis it occurs, not its VALUE. So per-token, min over CLAMP LEVELS at a
                    fixed loop count should recover roughly the same value as min over LOOPS at the
                    native scale -- the clamp is a reparameterisation of the same path.
  PATH hypothesis   different clamps expose different reachable states, so min over clamp levels
                    should reach LOWER than min over loops -- there is something new at other scales.

Reports, per token: min over loops at native scale (the §4.7 oracle), min over clamp levels at each
fixed loop count, and the union. If the union is no better than the loop-only minimum, the clamp adds
no reachable states and the rate reading holds. Zero training; reuses the clamp path.
"""
from __future__ import annotations
import argparse, math, pathlib, sys
import numpy as np, torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval import load_checkpoint  # noqa: E402
from radial_clamp import clamped_curve  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("--max-loops", type=int, default=32)
    ap.add_argument("--n-batches", type=int, default=15); ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cp = pathlib.Path(args.checkpoint)
    model, cfg, ck = load_checkpoint(cp / "last.pt" if cp.is_dir() else cp, dev)
    seq_len = ck["train_cfg"]["seq_len"]
    val = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    import json
    dyn = json.load((cp / f"dynamics_{cp.name}.json").open())
    H = cfg.hidden_size
    levels = {None: None}
    for k in (1, 8, 16):
        levels[f"h{k}"] = dyn["state_norm"][k - 1] / math.sqrt(H)

    mats = {}
    for name, rms in levels.items():
        acc = {t: [] for t in range(args.max_loops)}
        clamped_curve(model, val, args.max_loops, seq_len, args.n_batches, args.batch_size,
                      dev, rms, per_token_out=acc)
        M = np.stack([np.concatenate(acc[t], 0).reshape(-1) for t in range(args.max_loops)], axis=1)
        mats[name or "native"] = M
        print(f"  scored {name or 'native':>8}: {M.shape[0]:,} tokens x {M.shape[1]} loops", flush=True)

    native = mats["native"]
    loop_min = native.min(1)                       # best loop, native scale (the §4.7 oracle)
    all_stack = np.stack(list(mats.values()), 0)   # [clamp, tok, loop]
    clamp_min_at_best_loop = all_stack.min(0)[np.arange(len(native)), native.argmin(1)]
    union_min = all_stack.min(0).min(1)            # best over BOTH axes

    print(f"\n{'quantity':46} {'CE':>9}")
    print(f"{'best fixed loop, native (population)':46} {native.mean(0).min():>9.4f}")
    print(f"{'per-token min over LOOPS, native':46} {loop_min.mean():>9.4f}")
    print(f"{'per-token min over CLAMPS at its best loop':46} {clamp_min_at_best_loop.mean():>9.4f}")
    print(f"{'per-token min over BOTH axes (union)':46} {union_min.mean():>9.4f}")
    extra = loop_min.mean() - union_min.mean()
    print(f"\n  extra reachable by varying scale as well as depth: {extra:.4f} nats")
    frac = (all_stack.min(0).argmin(1) != native.argmin(1)).mean()
    print(f"  fraction of tokens whose best (scale, depth) differs in depth from native: {frac:.3f}")
    if extra < 0.02:
        print("\n  => RATE reading supported: clamping exposes essentially no states that depth alone")
        print("     could not reach. Scale is a reparameterisation of the same path, which is what")
        print("     §4.6's invariant ceiling implies and §8.2's trilemma assumes.")
    else:
        print("\n  => PATH reading supported: other scales reach measurably lower per token, so the")
        print("     clamp is not merely a rate control and §8.2 needs revising.")



def _persist_stdout(name, text):
    # PERSIST (traceability audit 2026-08-23): this printed its numbers and saved nothing, so
    # every claim it supports was reproducible but not traceable -- verifying one meant
    # re-running it, which only works while its inputs survive.
    import pathlib as _pl
    _dst = _pl.Path(__file__).resolve().parents[1] / "checkpoints" / f"{name}_report.txt"
    _dst.write_text(text)
    print(f"wrote {_dst}")

if __name__ == "__main__":
    import io as _io, contextlib as _cl
    _buf = _io.StringIO()
    with _cl.redirect_stdout(_buf):
        main()
    _out = _buf.getvalue()
    print(_out, end="")
    _persist_stdout("rate_vs_path", _out)

