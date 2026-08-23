"""SCSE's anchor-response diagnostic (arXiv 2607.27656), on this project's checkpoints.

THEIR CONSTRUCTION, verified from `papers/sources/2607.27656/lvr.tex` lines 131-137, not relayed:
let `e` be the input representation and `h*(e)` a reference state computed once from `e` and held
fixed through the unroll -- the *input-conditioned anchor*. Define the anchor-relative deviation
`Delta_t = h_t - h*(e)` and let `T_t(Delta; e)` be the model's one-step map in those coordinates.
Evaluating at zero deviation gives the ZERO-DEVIATION FORCING BIAS

    b_t(e) := T_t(0; e)

i.e. the next deviation produced from the anchor itself. `b_t(e) = 0` exactly when the anchor is a
one-step fixed point. SCSE sets it to zero by construction; additive-injection models do not.

WHAT THEIR THEORY DOES AND DOES NOT SAY, because it is easy to over-read: the bias is "a design
degree of freedom whose task effect can be harmful, neutral, or beneficial depending on the readout
and loss." **SCSE does not claim the forcing bias causes saturation.** So this report may cite it as
a prior formalisation of a quantity it measures, without conceding that its own saturation is
thereby explained.

WHAT IS DIFFERENT HERE, and why the measurement is worth making at all: SCSE reports `R_t` on one
trained model. This project has three checkpoints whose state norms span **380x** (||h||@8 =
6639.7 / 2334.4 / 17.5), which is a range their study does not cover.

PREDICTION, WRITTEN BEFORE RUNNING (CLAUDE.md sec 1):
  1. This block is WEIGHT-TIED with no loop conditioning, so `T_t` has no `t` dependence and
     **b_t(e) = b(e) is CONSTANT in t**. Any variation in R_t therefore comes entirely from the
     denominator -- the realized update. This is a structural claim and if it fails the
     implementation is wrong, not the model.
  2. `b` is set by the embedding scale, which §4.3's injection probe measured as essentially
     unchanged across arms (||e|| 1.504 -> 1.573), while the realized update scales with ||h||.
     So **R_t should be far LARGER in the norm-penalty arm** (small ||h||) than in the other two --
     the same regime split the injection ratio found (e/h@1 = 3.6e-1 vs 3.2e-3).
  3. R_t should DECAY in t in every arm, since ||h|| grows and b does not.

  FALSIFIER: if R_t is comparable across the three arms, then `b` scales with the state rather than
  the input, the anchor is not doing what the construction assumes here, and nothing about the
  forcing bias transfers to this architecture.

Anchor choice: `h*(e) = h0 + e`, the model's own initial state -- computed once from `e`, fixed
through the unroll, and it is literally what `forward()` starts from (`model.py`: `h = h0 + e`), so
it is the anchor this architecture already has rather than one imposed for the diagnostic.

Lands in §4.3 as EXPLANATION, not in §3.5 as evidence. Recorded here so that placement is a
pre-commitment and not a later choice.

Usage: python src/anchor_response.py <ckpt_dir> [...]
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


@torch.no_grad()
def anchor_response(model: LoopedTransformer, x: torch.Tensor, n_loops: int) -> dict:
    B, T = x.shape
    e = model.embed(x)
    cos, sin = model.rope(T, x.device, e.dtype)
    for layer in model.prelude:
        e = layer(e, cos, sin)
    h_star = model.h0.expand(B, T, -1) + e          # the anchor: computed once, held fixed

    # b(e) = T(0; e): the one-step map applied AT the anchor. In this architecture the loop body is
    # h_in = inject(h, e) -> block(h_in), so at h = h_star the produced next-deviation is
    #     b = block(inject(h_star, e)) - h_star
    b = model.block(model._inject(h_star, e), cos, sin) - h_star
    b_norm = b.float().norm(dim=-1)                  # [B,T], constant in t by construction

    _, _, states = model(x, n_loops=n_loops, return_all_loops=False, supervise_idx=set(),
                         return_states=True)
    prev = h_star
    out = {}
    for t, h in enumerate(states, start=1):
        upd = (h - prev).float().norm(dim=-1)        # realized update at this step
        out[t] = dict(R=float((b_norm / upd.clamp_min(1e-8)).mean()),
                      b_norm=float(b_norm.mean()), update=float(upd.mean()),
                      dev=float((h - h_star).float().norm(dim=-1).mean()))
        prev = h
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--loops", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    args = ap.parse_args()

    val = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    ix = rng.integers(0, len(val) - args.seq - 1, size=args.batch)
    x = torch.from_numpy(np.stack([val[i:i + args.seq] for i in ix]).astype(np.int64))

    show = [t for t in (1, 2, 4, 8, 16, 32, 64) if t <= args.loops]
    print(f"{'checkpoint':<30} {'||b||':>8} " + " ".join(f"{'R@'+str(t):>8}" for t in show))
    res = {}
    for cp in args.checkpoints:
        p = pathlib.Path(cp); p = p / "last.pt" if p.is_dir() else p
        d = torch.load(p, map_location="cpu", weights_only=False)
        m = LoopedTransformer(ModelConfig(**d["model_cfg"]))
        m.load_state_dict(d["model"]); m.eval()
        r = anchor_response(m, x, args.loops)
        res[p.parent.name] = r
        print(f"{p.parent.name:<30} {r[1]['b_norm']:>8.3f} "
              + " ".join(f"{r[t]['R']:>8.4f}" for t in show))
    dst = ROOT / "checkpoints" / "anchor_response.json"
    dst.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
