"""Spectral RADIUS of the loop map's Jacobian -- the actual criterion for convergence to a fixed point.

§4.3 concludes the winning config "does not contract", but that came from a FINITE h0 perturbation
(noise_scale = 1.0 against ||h0|| ~ 1.7, a ~60% perturbation, well outside the linear regime).
Convergence is a property of the Jacobian, and the criterion is `rho(dF/dh) < 1`. A map can contract
in nearly every direction while having one neutral direction -- plausibly the radial one §4.3 itself
identifies -- and a finite perturbation aligned with it would read "no contraction" from a map that
contracts almost everywhere. So the proxy and the definition can disagree, and only the definition
settles it.

NAMING, CORRECTED 2026-08-23. This file previously said `sigma_max` throughout and claimed to
power-iterate on `J^T J`. Its loop never applied `J^T` and therefore never computed a singular value;
it is plain power iteration on `J`, which converges to the spectral radius. Confirmed against a known
non-normal operator (`--null`, rho=1 vs sigma_max=10.1: the loop returns ~1.09). Every number this
file ever produced is a rho estimate. Note also that `sigma_max < 1` is only the *sufficient* Banach
condition; `rho < 1` is the iff, so the quantity actually measured is the more appropriate one.

Power iteration with finite-difference JVPs at the ACTUAL trajectory points, eps scaled to the local
state norm so the linearisation is genuine.

Usage: python src/jacobian_spec.py --null                 # instrument null, no checkpoint needed
       python src/jacobian_spec.py <ckpt_dir> [...]       # rho at the swept loop indices
"""
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval import load_checkpoint  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


@torch.no_grad()
def spectral_radius(model, h, e, cos, sin, iters=12, eps_rel=1e-3):
    """Spectral radius rho(dF/dh) of the one-loop map at state h, by power iteration on J via
    finite-difference JVPs (one extra forward pass per iteration).

    THIS FUNCTION WAS CALLED `sigma_max` AND ITS DOCSTRING CLAIMED TO POWER-ITERATE ON `J^T J`.
    It never did: the loop is `v <- Jv/||Jv||`, which never applies J^T (that would need a VJP, not
    a JVP) and is plain power iteration on J. Plain power iteration on J converges to |lambda_1|,
    the SPECTRAL RADIUS -- not the largest singular value. Verified against a known non-normal
    operator by `null_check()` below, where the two differ by 10x:

        A = [[1, 10], [0, 1]]    rho = 1.0000    sigma_max = 10.0990
        this loop returns 1.0942  ->  it is estimating rho, not sigma_max

    The mislabelling mattered in the right direction. rho <= sigma_max always, and the convergence
    criterion for a fixed point is `rho < 1` (sigma_max < 1 is the Banach *sufficient* condition, not
    an iff -- a non-normal map can have sigma_max > 1 and still converge). So the report's hedge that
    these numbers "only bound rho from above" was itself wrong: they ARE rho, and rho > 1 rules out
    local convergence rather than merely failing to establish it.

    Known bias: on a defective (Jordan-block) operator, 12 iterations overshoot by ~9% -- see the
    null. Power iteration also oscillates rather than converging when the dominant eigenvalue is a
    complex pair. Treat these as rho estimates biased slightly HIGH, which is the conservative
    direction for a `rho > 1` claim only if the true value is not near 1; at loop 64 the measured
    1.0015 is inside that bias and must NOT be read as strictly greater than 1."""
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



def null_check():
    """Instrument null: run the SAME iteration on an operator whose rho and sigma_max are known and
    differ by 10x. Required by CLAUDE.md sec 5 -- an instrument that decides a hypothesis needs one."""
    A = torch.tensor([[1.0, 10.0], [0.0, 1.0]], dtype=torch.float64)
    rho_true, smax_true = 1.0, 10.0990195
    h = torch.randn(2, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    base = A @ h
    v = torch.randn(2, dtype=torch.float64, generator=torch.Generator().manual_seed(1)); v /= v.norm()
    eps = 1e-3 * h.norm() / v.norm()
    s = 0.0
    for _ in range(12):
        Jv = (A @ (h + eps * v) - base) / eps
        s = Jv.norm().item() / v.norm().item()
        v = Jv / (Jv.norm() + 1e-12)
    assert abs(s - rho_true) < 0.15, f"iteration returned {s}, expected ~rho={rho_true}"
    assert abs(s - smax_true) > 5.0, f"iteration returned {s}, which is sigma_max, not rho"
    print(f"NULL PASS: iteration returns {s:.4f} -> rho ({rho_true}), NOT sigma_max ({smax_true:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*")
    ap.add_argument("--null", action="store_true", help="run the instrument null and exit")
    ap.add_argument("--loops", type=str, default="2,8,32,64")
    ap.add_argument("--batch", type=int, default=2); ap.add_argument("--seq", type=int, default=64)
    args = ap.parse_args()
    if args.null:
        null_check(); return
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
                    out[t] = spectral_radius(model, h, e, cos, sin)
                h_in = model._inject(h, e) if t > 1 else h
                h = model.block(h_in, cos, sin)
                if model.loop_norm is not None:
                    h = model.loop_norm(h)
        print(f"{cp.name:32} state_renorm={cfg.state_renorm}")
        for t in want:
            print(f"    loop {t:>3}: rho = {out[t]:.4f}  "
                  f"{'CONTRACTS' if out[t] < 1 else 'does NOT contract'}")


if __name__ == "__main__":
    main()
