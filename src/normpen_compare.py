"""Resolve §4.6's pre-registered prediction once the two 90M Kaggle arms land.

The prediction, quoted from report.md §4.6 (written before either run finished):

    "if the 90M norm-penalty arm improves **best** CE (not merely relocates the optimum), that is
     evidence training-time scale control changes the learned path; if it only relocates the optimum
     while best CE stays within noise of the control, the clamp's account extends to training-time
     interventions too."

Two things this script must not get wrong, both of which bit this project today:
  * "relocates the optimum" is restated as "shifts the PLATEAU" -- argmin on these curves is decided
    far below the noise floor (report §4.15; the headline curve's own argmin margin is 0.0002 nats).
  * plateau bands are only comparable on a SHARED eval grid, so the curves are intersected first.

Both arms are Kaggle T4 = CUDA. The applicable floor is therefore the CUDA one being measured by the
`tlab-cuda-null` job, NOT the 0.031-0.068 MPS figure. Until that lands the script uses the inferred
~0.05 and says so, and it prints the raw margin so the verdict can be recomputed against the measured
number without re-running anything.

Usage: python src/normpen_compare.py <control_eval.json> <normpen_eval.json> [floor]
"""
from __future__ import annotations
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from plateau import plateau, plateau_mid, sweep_tol


def curve(path):
    d = json.load(open(path))
    for k in ("val_ce", "curve", "ce_by_loop"):
        if k in d and isinstance(d[k], dict):
            return {int(t): float(v) for t, v in d[k].items()}, d
    raise KeyError(f"no loop curve found in {path}; keys={list(d)}")


def main(p_ctl, p_pen, floor=0.05):
    floor = float(floor)
    c_ctl, d_ctl = curve(p_ctl)
    c_pen, d_pen = curve(p_pen)
    shared = sorted(set(c_ctl) & set(c_pen))
    if not shared:
        print("NO SHARED GRID -- refusing to compare"); return 1
    print(f"shared grid: {len(shared)} points, loops {min(shared)}..{max(shared)}")
    for nm, d in (("control", d_ctl), ("normpen", d_pen)):
        print(f"  {nm}: tokens={d.get('tokens', 'NA')}")

    r_ctl = {t: c_ctl[t] for t in shared}
    r_pen = {t: c_pen[t] for t in shared}
    b_ctl, b_pen = min(r_ctl.values()), min(r_pen.values())
    lo_c, hi_c, _ = plateau(r_ctl)
    lo_p, hi_p, _ = plateau(r_pen)

    print(f"\n{'arm':<10} {'best CE':>9} {'ppl':>9} {'CE@1':>9} {'gain':>8} {'plateau':>12} {'mid':>6}")
    import math
    for nm, r, b, lo, hi in (("control", r_ctl, b_ctl, lo_c, hi_c), ("normpen", r_pen, b_pen, lo_p, hi_p)):
        print(f"{nm:<10} {b:>9.4f} {math.exp(b):>9.3f} {r[min(r)]:>9.4f} "
              f"{r[min(r)]-b:>8.4f} {'['+str(lo)+','+str(hi)+']':>12} {plateau_mid(r):>6.1f}")

    d_best = b_pen - b_ctl
    moved = (lo_p, hi_p) != (lo_c, hi_c)
    print(f"\nbest CE difference (normpen - control) = {d_best:+.4f} nats   [floor used: {floor:.3f}]")
    print(f"plateau moved: {moved}  ({lo_c},{hi_c}) -> ({lo_p},{hi_p})")
    print("tolerance sweeps:")
    for nm, r in (("control", r_ctl), ("normpen", r_pen)):
        print(f"  {nm}: " + "  ".join(f"{t}:[{a},{b}]" for t, (a, b, _) in sweep_tol(r).items()))

    print("\nVERDICT against the pre-registered prediction:")
    if d_best < -floor:
        print("  normpen IMPROVES best CE beyond the floor ->")
        print("  training-time scale control CHANGES THE LEARNED PATH (clamp's account does NOT extend).")
    elif abs(d_best) <= floor:
        print("  best CE is WITHIN the floor ->")
        if moved:
            print("  ...and the plateau moved: the clamp's account EXTENDS to training-time interventions,")
            print("     i.e. scale control relocates where depth is spent without changing the ceiling.")
        else:
            print("  ...and the plateau did NOT move: the intervention is inert on both axes.")
    else:
        print(f"  normpen is WORSE by {d_best:+.4f}, beyond the floor -> the penalty hurts the path.")
    print("\n(Recompute with the measured CUDA floor by passing it as the third argument.)")
    return 0



def _persist_stdout(name, text):
    # PERSIST (traceability audit 2026-08-23): this printed its numbers and saved nothing, so
    # every claim it supports was reproducible but not traceable -- verifying one meant
    # re-running it, which only works while its inputs survive.
    import pathlib as _pl
    _dst = _pl.Path(__file__).resolve().parents[1] / "checkpoints" / f"{name}_report.txt"
    _dst.write_text(text)
    print(f"wrote {_dst}")

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
