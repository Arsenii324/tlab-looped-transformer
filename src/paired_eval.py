"""Common-random-numbers evaluation: one frozen held-out set, per-sequence CE, paired bootstrap.

Why this exists. This project's own eval noise is ~0.065 nats between two independent samples of the
SAME checkpoint (report.md §8 item 3) -- larger than four of the five screening effects in §4.1 and
larger than the entire difference between several arms compared elsewhere. Any conclusion drawn from
comparing two numbers produced by two different random draws is therefore uninterpretable at the
resolution this project needs.

The fix costs no compute. Freeze one evaluation set -- the same sequences in the same order -- and
reuse it for every arm and every loop count. Then the comparison between two arms is a PAIRED one:
the same sequence contributes to both, so whatever makes a sequence intrinsically easy or hard
cancels in the difference. What remains is bootstrapped over sequences to get an interval on the
paired delta rather than a bare difference of two point estimates.

The val shard is 6M tokens and previous evals used 0.3-0.7% of it, so a frozen set of a few thousand
sequences is both affordable and a strict improvement in coverage.

Two things this deliberately does NOT do. It does not re-derive CE -- it calls the same
`model.readout` path `eval.py` uses, one loop at a time. And it does not replace the published
numbers: those stay as they are, produced by the protocol they were produced by. This is the
instrument for NEW comparisons, and for re-testing old ones at higher resolution.

Usage:
  python src/paired_eval.py build --n-seq 2048                # freeze the set once
  python src/paired_eval.py score <ckpt_dir> [--max-loops 64] # per-sequence CE matrix
  python src/paired_eval.py compare <ckpt_a> <ckpt_b>         # paired delta + bootstrap CI
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval import load_checkpoint, BYTES_PER_TOKEN  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "frozen_eval_set.npz"

if torch.backends.mps.is_available():
    torch.mps.set_per_process_memory_fraction(10.0e9 / torch.mps.recommended_max_memory())


def build(n_seq: int, seq_len: int, val_path: pathlib.Path, seed: int = 20260823):
    """Frozen once and written to disk. Every later comparison loads THIS file, so the eval set is a
    fixed artifact of the project rather than a function of whatever seed a script happened to use."""
    val = np.memmap(val_path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(val) - seq_len - 1, size=n_seq)
    starts.sort()                       # deterministic order, and memmap-friendly
    x = np.stack([val[i:i + seq_len] for i in starts]).astype(np.int64)
    y = np.stack([val[i + 1:i + seq_len + 1] for i in starts]).astype(np.int64)
    np.savez_compressed(FROZEN, x=x, y=y, starts=starts, seq_len=seq_len, seed=seed)
    print(f"froze {n_seq} sequences x {seq_len} tokens = {n_seq*seq_len:,} scored tokens -> {FROZEN}")
    print(f"  ({n_seq*seq_len/len(val)*100:.2f}% of the {len(val):,}-token val shard; "
          f"previous evals used 0.3-0.7%)")


@torch.no_grad()
def score(ckpt_dir: pathlib.Path, max_loops: int, batch_size: int):
    d = np.load(FROZEN)
    x_all, y_all = d["x"], d["y"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    p = ckpt_dir / "last.pt" if ckpt_dir.is_dir() else ckpt_dir
    model, cfg, ckpt = load_checkpoint(p, device)
    n = len(x_all)
    ce = np.zeros((n, max_loops), dtype=np.float64)     # per-sequence, per-loop mean CE
    for b0 in range(0, n, batch_size):
        xb = torch.from_numpy(x_all[b0:b0 + batch_size]).to(device)
        yb = torch.from_numpy(y_all[b0:b0 + batch_size]).to(device)
        _, _, states = model(xb, n_loops=max_loops, return_all_loops=False,
                             supervise_idx=set(), return_states=True)
        cos, sin = model.rope(xb.shape[1], xb.device, states[0].dtype)
        for r in range(max_loops):
            lg = model.readout(states[r], cos, sin)
            # reduction='none' then mean over positions -> one number per SEQUENCE, which is the
            # unit the bootstrap resamples. Aggregating to a scalar here would destroy the pairing.
            per_tok = F.cross_entropy(lg.reshape(-1, lg.size(-1)), yb.reshape(-1), reduction="none")
            # .cpu() BEFORE .double(): MPS has no float64, so calling .double() on an MPS tensor
            # raises "Cannot convert a MPS Tensor to float64 dtype". Precision is unaffected -- the
            # mean is taken in float32 on device, then widened on the host for accumulation.
            ce[b0:b0 + xb.shape[0], r] = per_tok.view(xb.shape[0], -1).mean(1).cpu().double().numpy()
        if (b0 // batch_size) % 10 == 0:
            print(f"  {b0+xb.shape[0]}/{n}", flush=True)
    out = ckpt_dir / f"paired_{ckpt_dir.name}.npz"
    np.savez_compressed(out, ce=ce, max_loops=max_loops, tokens=ckpt.get("tokens"),
                        step=ckpt.get("step"))
    m = ce.mean(0)
    best = int(m.argmin()) + 1
    print(f"wrote {out}\n  best loop {best}  CE {m[best-1]:.4f}  CE@1 {m[0]:.4f}  "
          f"loop gain {m[0]-m[best-1]:.4f}  (n={len(ce)} sequences)")
    return out


def boot(delta: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    means = delta[idx].mean(1)
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(a: pathlib.Path, b: pathlib.Path):
    ca = np.load(a)["ce"]; cb = np.load(b)["ce"]
    assert ca.shape[0] == cb.shape[0], "different eval sets -- not paired, refusing to compare"
    R = min(ca.shape[1], cb.shape[1])
    print(f"paired over {ca.shape[0]} identical sequences; loops 1..{R}")
    print(f"{'loop':>5} {'A':>9} {'B':>9} {'B-A':>9} {'95% CI':>22} {'sig':>4}")
    for r in [1, 2, 4, 8, 11, 16, 24, 32, 48, 64]:
        if r > R:
            continue
        d = cb[:, r-1] - ca[:, r-1]
        m, lo, hi = boot(d)
        sig = "*" if (lo > 0) or (hi < 0) else ""
        print(f"{r:>5} {ca[:,r-1].mean():>9.4f} {cb[:,r-1].mean():>9.4f} {m:>9.4f} "
              f"[{lo:>8.4f},{hi:>8.4f}] {sig:>4}")
    # the quantity the task is actually scored on
    ga = ca[:, 0] - ca.min(1); gb = cb[:, 0] - cb.min(1)
    m, lo, hi = boot(gb - ga)
    print(f"\nLOOP GAIN (CE@1 - best, per sequence):  A {ga.mean():.4f}  B {gb.mean():.4f}  "
          f"B-A {m:+.4f}  95% CI [{lo:.4f},{hi:.4f}]  "
          f"{'SIGNIFICANT' if (lo>0 or hi<0) else 'not distinguishable'}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pb = sub.add_parser("build"); pb.add_argument("--n-seq", type=int, default=2048)
    pb.add_argument("--seq-len", type=int, default=256)
    pb.add_argument("--val", type=str, default=str(ROOT / "data" / "val.bin"))
    ps = sub.add_parser("score"); ps.add_argument("ckpt"); ps.add_argument("--max-loops", type=int, default=64)
    ps.add_argument("--batch-size", type=int, default=8)
    pc = sub.add_parser("compare"); pc.add_argument("a"); pc.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.n_seq, args.seq_len, pathlib.Path(args.val))
    elif args.cmd == "score":
        score(pathlib.Path(args.ckpt), args.max_loops, args.batch_size)
    else:
        compare(pathlib.Path(args.a), pathlib.Path(args.b))


if __name__ == "__main__":
    main()
