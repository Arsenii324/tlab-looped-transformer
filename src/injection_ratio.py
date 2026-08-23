"""How large is the re-injected input `e` relative to the state it is added to?

WHY THIS EXISTS, AND THE PREDICTION IT TESTS. §4.3 reports `‖e‖/‖h‖ = 1.3e-3 falling to 7e-5` over
loops 1..64 and reads it as *dilution*: the input is re-injected at every loop but becomes
negligible against a state whose norm grows ~18x, so late loops effectively iterate on the state
alone. That measurement was made on the 46M checkpoint.

The norm-penalty arm changes the premise. Its state norms are ~380x smaller than the 46M model's
(‖h‖@1 = 4.4 vs 1659.5) while ‖e‖ is set by the embedding table and is NOT penalised directly. So
the same additive injection should be a first-order term there rather than a rounding error.

    PREDICTION (written before running): ‖e‖/‖h_1‖ is O(1) on the norm-penalty checkpoint and
    O(1e-3) on the other two. If it holds, the +0.2263 loop-1 damage that drives 88% of that arm's
    loop-gain advantage has a candidate mechanism -- at loop 1 the readout sees a state that is
    substantially raw embedding, which is not a state any readout can decode well.

    FALSIFIER: if the ratio is ~1e-3 in all three arms, the penalty scales `e` and `h` together
    (i.e. it reaches the embedding through the tied head) and dilution is scale-free. Then this
    explains nothing and the loop-1 damage stays unexplained.

Reports the ratio per loop, not just at loop 1, because dilution is a claim about the whole
trajectory and a single loop cannot distinguish "starts large and dilutes" from "always large".
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


def injection_ratio(model: LoopedTransformer, x: torch.Tensor, n_loops: int) -> dict:
    """‖e‖/‖h_t‖ per loop, both measured per-token then averaged -- NOT a ratio of averages, which
    would hide the per-token spread that decides whether this is a uniform property or a tail."""
    if model.cfg.n_prelude:
        raise ValueError("with a prelude, `e` is the prelude output, not the embedding; this probe "
                         "reads model.embed directly and would measure the wrong tensor")
    with torch.no_grad():
        e = model.embed(x)                                   # [B,T,H], the tensor _inject() adds
        _, _, states = model(x, n_loops=n_loops, return_all_loops=False, return_states=True)
    e_norm = e.float().norm(dim=-1)                          # [B,T]
    out = {}
    for t, h in enumerate(states, start=1):
        r = (e_norm / h.float().norm(dim=-1)).flatten()
        out[t] = dict(mean=r.mean().item(), median=r.median().item(),
                      p95=r.quantile(0.95).item(), max=r.max().item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--loops", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    val = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)   # fixed: every checkpoint sees the SAME tokens
    ix = rng.integers(0, len(val) - args.seq - 1, size=args.batch)
    x = torch.from_numpy(np.stack([val[i:i + args.seq] for i in ix]).astype(np.int64)).to(args.device)

    results = {}
    show = [t for t in (1, 2, 4, 8, 16, 32, 64) if t <= args.loops]
    print(f"{'checkpoint':<32} " + "  ".join(f"{'e/h@'+str(t):>9}" for t in show))
    for name in args.ckpts:
        p = ROOT / "checkpoints" / name / "last.pt"
        if not p.exists():
            print(f"{name:<32} ABSENT"); continue
        d = torch.load(p, map_location=args.device, weights_only=False)
        m = LoopedTransformer(ModelConfig(**d["model_cfg"])).to(args.device)
        m.load_state_dict(d["model"]); m.eval()
        r = injection_ratio(m, x, args.loops)
        results[name] = r
        print(f"{name:<32} " + "  ".join(f"{r[t]['mean']:9.2e}" for t in show))
    out = ROOT / "checkpoints" / "injection_ratio.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
