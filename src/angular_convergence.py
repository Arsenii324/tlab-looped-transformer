"""Does the unit state converge, or drift forever? Two models, fit against each other.

WHY THIS EXISTS. §2 claims "saturation without convergence" and argued it from rho >~ 1 -- a
linearisation whose deep-loop readings sit inside their own estimator bias (see jacobian_spec.py's
null). That is an indirect instrument for a question that can be asked directly, because
`u_t = h_t/||h_t||` is the ENTIRE input to the block (RMSNorm is scale-invariant) and the entire
input to the readout. If `u` settles, the loop is doing nothing the model can see, whatever ||h||
does.

THE TWO HYPOTHESES, and they are distinguishable with one free parameter each:

  CONVERGENT   ||u_t - u*|| = A * t^-b        there is a fixed point u* of the induced sphere map
                                              G(u) = F(u)/||F(u)||; the trajectory settles onto it
  LOG-DRIFT    ||u_t - u_T|| = C * ln(T/t)    steps of size C/t that stay ALIGNED, so the
                                              accumulated path diverges and there is no u* at all

These are not a stylistic choice. §4.3 measures both ingredients of the second: the per-loop angular
step decays as 1/t, and consecutive steps are aligned at cos -> 0.9999. Sum an aligned 1/t step and
you get a logarithm, which diverges. The fit below just checks whether the composition holds.

WHY THE DISTINCTION IS EASY TO GET WRONG, recorded because it was: the CONSECUTIVE-STEP exponent is
~ -1.0, and the DISTANCE-TO-LIMIT exponent is ~ -0.6. Quoting the first as if it were the second
makes `||u_t - u*|| ~ 1/t` -- the convergent prediction -- look confirmed. They are different
quantities and only the second bears on convergence.

Note `u*` is unobservable, so the distance is measured to `u_T` at the largest loop run. That biases
TOWARD the convergent hypothesis (the distance is 0 at t=T by construction), which is the safe
direction: log-drift winning anyway is the stronger result. The fit window excludes the endpoint.

Usage: python src/angular_convergence.py <ckpt_dir> [...] [--loops 384]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig, LoopedTransformer  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def r2(y, pred):
    return 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def fit(U, R: int, lo: int = 8, hi: int | None = None) -> dict:
    hi = hi or (2 * R) // 3
    if hi <= lo:
        raise ValueError(f"fit window [{lo},{hi}) is empty at R={R}; need more loops")
    t = np.arange(1, R + 1)
    dist = np.array([(U[i] - U[-1]).norm(dim=-1).mean().item() for i in range(R)])
    step = np.array([(U[i] - U[i - 1]).norm(dim=-1).mean().item() for i in range(1, R)])

    b, a = np.polyfit(np.log(t[lo:hi]), np.log(dist[lo:hi]), 1)          # 2 params
    Lt = np.log(R / t[lo:hi])
    C = float((dist[lo:hi] * Lt).sum() / (Lt * Lt).sum())                 # 1 param
    b_step, _ = np.polyfit(np.log(t[lo:hi]), np.log(step[lo:hi]), 1)

    return dict(
        power_exponent=float(b), power_r2=float(r2(dist[lo:hi], np.exp(a) * t[lo:hi] ** b)),
        logdrift_C=C, logdrift_r2=float(r2(dist[lo:hi], C * Lt)),
        step_exponent=float(b_step),
        tail_motion=float(step[R // 3:].sum()),   # angular motion accumulated over the last 2/3
        dist_at={k: float(dist[k - 1]) for k in (8, 32, 128) if k <= R})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--loops", type=int, default=384)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    args = ap.parse_args()

    val = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)   # fixed: every checkpoint sees the SAME tokens
    ix = rng.integers(0, len(val) - args.seq - 1, size=args.batch)
    x = torch.from_numpy(np.stack([val[i:i + args.seq] for i in ix]).astype(np.int64))

    out = {}
    for cp in args.checkpoints:
        p = pathlib.Path(cp)
        p = p / "last.pt" if p.is_dir() else p
        d = torch.load(p, map_location="cpu", weights_only=False)
        m = LoopedTransformer(ModelConfig(**d["model_cfg"]))
        m.load_state_dict(d["model"]); m.eval()
        with torch.no_grad():
            _, _, st = m(x, n_loops=args.loops, return_all_loops=False,
                          supervise_idx=set(), return_states=True)
        U = [h.float() / h.float().norm(dim=-1, keepdim=True) for h in st]
        r = fit(U, args.loops)
        out[p.parent.name] = r
        win = "CONVERGENT" if r["power_r2"] > r["logdrift_r2"] else "LOG-DRIFT"
        print(f"{p.parent.name}")
        print(f"   power law  t^{r['power_exponent']:+.3f}  R2={r['power_r2']:.4f}   (2 params)")
        print(f"   log-drift  C={r['logdrift_C']:.4f}     R2={r['logdrift_r2']:.4f}   (1 param)  -> {win}")
        print(f"   consecutive-step exponent {r['step_exponent']:+.3f}  "
              f"(this is the 1/t of §4.3 -- NOT the distance-to-limit exponent)")
        print(f"   angular motion over the last 2/3 of the sweep: {r['tail_motion']:.4f} rad "
              f"(a converged trajectory would give ~0)")
    dst = ROOT / "checkpoints" / "angular_convergence.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
