"""What does it cost to attend to a KV cache written at a DIFFERENT loop depth?

The early-exit question this project actually needs answered. If tokens exit at different depths,
the attention cache is ragged: a token computing at depth t reads keys/values that earlier positions
wrote at depth k != t. Every published treatment of this (River-LLM's "KV Cache Absence", MoR's two
cache strategies, SkipDecode's monotonically-decreasing exit depth, LLA's cross-loop codec) assumes
DEEPER IS BETTER and the exited state is a degraded proxy. This model says otherwise: past loop 8 CE
rises monotonically out to loop 105 (§4.2), so a deep cache entry is not a better version of a
shallow one -- it may be a worse one. That inverts the cost-benefit of every KV-recompute scheme, and
it is measurable here in one pass.

Definition used, stated because several are possible. For cell (k, t): run the clean trajectory,
then re-run ONLY loop t with each layer's keys/values sourced from that same layer's input on loop
k, queries still coming from the loop-t stream. Everything else is untouched. So the cell isolates a
single-step substitution -- "this token computes at depth t while its context was cached at depth k"
-- rather than compounding the substitution over the whole trajectory.

Note this is teacher-forced, so it is NOT a generation experiment: every position is processed at
every depth in parallel and no mixed-depth cache is ever built. That is the point. In the free-FLOP
teacher-forced setting early exit is a DEPTH-SELECTION mechanism, not a compute-saving one, and this
grid measures the penalty a real (generating) exiter would additionally pay.

Reads the diagonal as its own control: cell (t, t) must reproduce the clean per-loop CE exactly,
which is asserted rather than assumed.

Usage: python src/cross_depth_kv.py <ckpt_dir> [--loops 1,2,4,8,16,32,64]
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
from eval import load_checkpoint  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

if torch.backends.mps.is_available():
    torch.mps.set_per_process_memory_fraction(9.0e9 / torch.mps.recommended_max_memory())


@torch.no_grad()
def grid(model, X, Y, loops, device, batch_size=4):
    R = max(loops)
    n = len(X)
    ce = np.full((len(loops), len(loops)), np.nan)     # [k_index, t_index]
    diag = np.full(len(loops), np.nan)
    tot = 0
    acc = np.zeros((len(loops), len(loops)))
    accd = np.zeros(len(loops))
    for b0 in range(0, n, batch_size):
        xb = torch.from_numpy(X[b0:b0+batch_size]).to(device)
        yb = torch.from_numpy(Y[b0:b0+batch_size]).to(device)
        B, T = xb.shape
        e = model.embed(xb)
        cos, sin = model.rope(T, xb.device, e.dtype)
        for layer in model.prelude:
            e = layer(e, cos, sin)
        h = model.h0.expand(B, T, -1) + e
        states, layer_inputs = [h], []
        for t in range(R):
            h_in = (h + e) if t > 0 else h
            col = []
            h = model.block(h_in, cos, sin, collect=col)
            if model.loop_norm is not None:
                h = model.loop_norm(h)
            layer_inputs.append(col)      # per-layer inputs ON loop t -- the depth-t cache content
            states.append(h)
        def ce_of(hh):
            lg = model.readout(hh, cos, sin)
            return F.cross_entropy(lg.reshape(-1, lg.size(-1)), yb.reshape(-1),
                                   reduction="sum").item()
        for ti, t in enumerate(loops):
            accd[ti] += ce_of(states[t])
            for ki, k in enumerate(loops):
                h_prev = states[t-1]
                h_in = (h_prev + e) if t > 1 else h_prev
                hh = model.block(h_in, cos, sin, kv_sources=layer_inputs[k-1])
                if model.loop_norm is not None:
                    hh = model.loop_norm(hh)
                acc[ki, ti] += ce_of(hh)
        tot += xb.numel()
        if (b0 // batch_size) % 10 == 0:
            print(f"  {b0+B}/{n}", flush=True)
    return acc / tot, accd / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--loops", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("--n-seq", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    loops = [int(x) for x in args.loops.split(",")]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cp = pathlib.Path(args.checkpoint)
    model, cfg, ck = load_checkpoint(cp / "last.pt" if cp.is_dir() else cp, device)
    d = np.load(ROOT / "data" / "frozen_eval_set.npz")
    X, Y = d["x"][:args.n_seq], d["y"][:args.n_seq]
    print(f"{cp.name} tokens={ck.get('tokens')} seqs={len(X)} grid={loops}", flush=True)

    M, diag = grid(model, X, Y, loops, device, args.batch_size)

    # CONTROL: the diagonal (k==t) is an ordinary forward step and must match the clean curve.
    dmax = max(abs(M[i, i] - diag[i]) for i in range(len(loops)))
    print(f"\nCONTROL diagonal vs clean per-loop CE: max|diff| = {dmax:.3e} "
          f"{'OK' if dmax < 1e-4 else '<-- SUBSTITUTION PATH IS WRONG, table below is meaningless'}")

    print(f"\nrows = cache depth k, cols = compute depth t; entry = val CE\n")
    print("  k\\t " + "".join(f"{t:>9}" for t in loops))
    for ki, k in enumerate(loops):
        print(f"{k:>5} " + "".join(f"{M[ki,ti]:>9.4f}" for ti in range(len(loops))))
    print("\nclean diag " + "".join(f"{v:>9.4f}" for v in diag))
    print("\nper compute-depth t: best cache depth k (and its gain over the matched k=t cache)")
    for ti, t in enumerate(loops):
        ki = int(np.nanargmin(M[:, ti]))
        print(f"  t={t:>3}: best k={loops[ki]:>3}  CE {M[ki,ti]:.4f}  vs matched {M[ti,ti]:.4f}  "
              f"({M[ki,ti]-M[ti,ti]:+.4f})")
    out = cp / f"crossdepth_{cp.name}.json"
    out.write_text(json.dumps(dict(loops=loops, matrix=M.tolist(), clean_diag=diag.tolist(),
                                    control_max_diff=dmax, tokens=ck.get("tokens")), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
