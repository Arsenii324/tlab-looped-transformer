"""Noise-robust replacements for `argmin of the loop curve`.

WHY THIS EXISTS. Most depth claims in this report are stated as "the optimum sits at loop k",
computed as argmin over the eval sweep. On these curves that statistic is close to meaningless:
the basin is flat, and the argmin is routinely decided by margins of 1e-4 to 3e-3 nats against a
noise floor two orders of magnitude larger. Measured margins between the best and second-best grid
point, read off the raw JSON:

    rs_lambda2_s0     argmin 12 beats loop 8 by 0.0001 nats
    rs_lambda1_s0     argmin 12 beats loop 8 by 0.0006
    sd_dense_k5_s1    argmin  8 beats loop 12 by 0.0003
    sd_terminal_k1_s0 argmin 16 beats loop 12 by 0.0034

A statistic decided at 1e-4 cannot support a claim about where a model "wants" to stop. The
residual-scale arms are the cautionary case: reported by argmin they look like a second
intervention that shifts the optimum 8 -> 12 and breaks the t/L rule; their curves are in fact
tied to within 0.0001 nats over that whole interval, and the control is *better* in absolute CE.

WHAT TO USE INSTEAD. The honest object is the interval of depths that are all as good as the best,
to within a tolerance set by the noise floor -- not the single grid point that happens to win.

    plateau(curve, tol)   -> (lo, hi) contiguous run of depths within `tol` of the minimum
    plateau_mid(curve)    -> geometric midpoint of that interval
    onset(curve, tol)     -> shallowest depth within `tol` of the minimum (the "how few loops
                             can I get away with" question, which is the deployment-relevant one)

GRID DEPENDENCE -- the one way to misuse these functions. A plateau is a set of *evaluated* depths,
so both the band and its midpoint depend on which loop counts were swept. Measured on the headline
checkpoint's own curve:

    dense every-integer 1..64        plateau [5,14]   mid 8.4
    sparse {1,2,4,8,12,16,24,32}     plateau [8,12]   mid 9.8

Same model, same weights, same tolerance -- a 17% difference in midpoint from grid choice alone.
**Never compare midpoints across arms that were swept on different grids.** Restrict to the shared
grid first (this is why the CUDA deep-terminal arm is compared to §4.14 on {1,2,4,8,16,24,32} rather
than on its own denser sweep). Within one experiment all arms share a grid, so intra-experiment
comparisons are safe.

Default tol=0.01 nats: comfortably above the 1e-4..3e-3 argmin margins that motivated this file,
and comfortably below the effect sizes actually claimed (a 0.07-nat shift at t/L=2, a 0.13-nat
loop-gain gap). Every function takes tol explicitly so a caller can show the result is not an
artifact of one threshold; `sweep_tol` does that automatically.
"""
from __future__ import annotations


def _norm(curve):
    c = {int(t): float(v) for t, v in curve.items()}
    if not c:
        raise ValueError("empty loop curve")
    return c


def plateau(curve, tol: float = 0.01):
    """Contiguous run of swept depths whose CE is within `tol` of the curve minimum.

    Contiguity matters: a non-contiguous set means the curve is not basin-shaped and the whole
    'optimal depth' framing is suspect, so that case is reported rather than silently unioned.
    """
    c = _norm(curve)
    ts = sorted(c)
    m = min(c.values())
    ok = [t for t in ts if c[t] - m <= tol]
    lo_i, hi_i = ts.index(ok[0]), ts.index(ok[-1])
    contiguous = all(c[t] - m <= tol for t in ts[lo_i:hi_i + 1])
    return ok[0], ok[-1], contiguous


def plateau_mid(curve, tol: float = 0.01):
    """Geometric midpoint of the plateau. Geometric, not arithmetic: the sweep grid is
    geometric (1,2,4,8,...), so an arithmetic midpoint would be biased toward the deep end."""
    lo, hi, _ = plateau(curve, tol)
    return (lo * hi) ** 0.5


def onset(curve, tol: float = 0.01):
    """Shallowest depth already within `tol` of the best the model ever achieves."""
    return plateau(curve, tol)[0]


def sweep_tol(curve, tols=(0.005, 0.01, 0.02, 0.05)):
    """Same interval at several tolerances -- the check that a plateau claim is not an artifact
    of one threshold choice."""
    return {t: plateau(curve, t) for t in tols}
