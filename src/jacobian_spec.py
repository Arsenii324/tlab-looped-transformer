"""Spectral norm of the loop map's Jacobian -- the actual definition of contraction.

§4.3 concludes the winning config "does not contract", but that came from a FINITE h0 perturbation
(noise_scale = 1.0 against ||h0|| ~ 1.7, a ~60% perturbation, well outside the linear regime).
Contraction is a property of the Jacobian: F contracts iff sigma_max(dF/dh) < 1. A map can contract
in nearly every direction while having one neutral direction -- plausibly the radial one §4.3 itself
identifies -- and a finite perturbation aligned with it would read "no contraction" from a map that
contracts almost everywhere. So the proxy and the definition can disagree, and only the definition
settles it.

Power iteration with finite-difference JVPs at the ACTUAL trajectory points, eps scaled to the local
state norm so the linearisation is genuine.
"""
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval import load_checkpoint  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


@torch.no_grad()
def sigma_max(model, h, e, cos, sin, iters=12, eps_rel=1e-3):
    """Largest singular value of the one-loop map at state h, by power iteration on J^T J via
    finite-difference JVPs (two forward passes per iteration)."""
    def F(x):
        y = model.block(model._inject(x, e), cos, sin)
        return model.loop_norm(y) if model.loop_norm is not None else y
    base = F(h)
    v = torch.randn_like(h); v /= v.norm()
    eps = eps_rel * h.norm() / v.norm()
    s = 0.0
    for _ in range(iters):
        Jv = (F(h + eps * v) - base) / eps
        s = Jv.norm().item() / v.norm().item()
        # one-sided power iteration: renormalise the image direction back through the map
        v = Jv / (Jv.norm() + 1e-12)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--loops", type=str, default="2,8,32,64")
    ap.add_argument("--batch", type=int, default=2); ap.add_argument("--seq", type=int, default=64)
    args = ap.parse_args()
    want = [int(x) for x in args.loops.split(",")]
    val = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    for cp in args.checkpoints:
        cp = pathlib.Path(cp)
        model, cfg, ck = load_checkpoint(cp / "last.pt" if cp.is_dir() else cp, "cpu")
        x = torch.from_numpy(val[:args.batch*args.seq].astype(np.int64)).view(args.batch, args.seq)
        e = model.embed(x)
        cos, sin = model.rope(args.seq, x.device, e.dtype)
        for l in model.prelude:
            e = l(e, cos, sin)
        h = model.h0.expand(args.batch, args.seq, -1) + e
        out = {}
        with torch.no_grad():
            for t in range(1, max(want) + 1):
                if t in want:
                    out[t] = sigma_max(model, h, e, cos, sin)
                h_in = model._inject(h, e) if t > 1 else h
                h = model.block(h_in, cos, sin)
                if model.loop_norm is not None:
                    h = model.loop_norm(h)
        print(f"{cp.name:32} state_renorm={cfg.state_renorm}")
        for t in want:
            print(f"    loop {t:>3}: sigma_max = {out[t]:.4f}  "
                  f"{'CONTRACTS' if out[t] < 1 else 'does NOT contract'}")


if __name__ == "__main__":
    main()
