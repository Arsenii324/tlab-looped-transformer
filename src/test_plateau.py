"""Tests for plateau.py, which is now load-bearing for most depth claims in the report.

Includes the degenerate cases the function is supposed to handle honestly rather than smooth over,
and one exact-reproduction test against a number already published in report.md §4.9 -- if the
statistic cannot re-derive a figure the report already stands on, it is not usable for re-testing it.
"""
from __future__ import annotations
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from plateau import plateau, plateau_mid, onset, sweep_tol

ROOT = pathlib.Path(__file__).resolve().parents[1]
fails = []


def check(name, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) and not isinstance(want, bool) else (got == want)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}: got {got!r} want {want!r}")
    if not ok:
        fails.append(name)


print("[1] textbook basin: minimum at 8, symmetric, wide tolerance")
c = {1: 1.0, 2: 0.5, 4: 0.2, 8: 0.0, 16: 0.2, 32: 0.5}
check("plateau@0.01", plateau(c, 0.01)[:2], (8, 8))
check("plateau@0.25", plateau(c, 0.25)[:2], (4, 16))
check("onset@0.25", onset(c, 0.25), 4)

print("[2] FLAT curve -- every depth within tolerance; must return the whole range, not a point")
c = {1: 5.0, 2: 5.001, 4: 5.002, 8: 5.0, 16: 5.003, 32: 5.001}
lo, hi, contig = plateau(c, 0.01)
check("flat plateau spans everything", (lo, hi), (1, 32))
check("flat plateau is contiguous", contig, True)

print("[3] NON-CONTIGUOUS: two separate dips -- must be reported, not silently unioned")
c = {1: 0.0, 2: 0.5, 4: 0.5, 8: 0.001, 16: 0.5, 32: 0.5}
lo, hi, contig = plateau(c, 0.01)
check("band spans the outer hits", (lo, hi), (1, 8))
check("flagged non-contiguous", contig, False)

print("[4] geometric midpoint, not arithmetic (the sweep grid is geometric)")
c = {4: 0.0, 8: 0.005, 16: 0.005, 32: 1.0}
check("mid == sqrt(4*16)", round(plateau_mid(c, 0.01), 6), 8.0, tol=1e-6)

print("[5] degenerate input must raise, not return something plausible")
try:
    plateau({}, 0.01)
    check("empty curve raises", False, True)
except ValueError:
    check("empty curve raises", True, True)

print("[6] tolerance monotonicity: a wider tolerance can never shrink the band")
c = {1: 0.30, 2: 0.10, 4: 0.02, 8: 0.0, 16: 0.02, 32: 0.10, 64: 0.30}
prev = None
mono = True
for t, (lo, hi, _) in sorted(sweep_tol(c).items()):
    if prev and not (lo <= prev[0] and hi >= prev[1]):
        mono = False
    prev = (lo, hi)
check("band grows monotonically with tol", mono, True)

print("[7] EXACT reproduction of a published number: report §4.9 states the L=16 arm peaks at 8")
p = ROOT / "checkpoints" / "train_at_L_results.json"
if p.exists():
    d = json.load(open(p))
    c = {int(t): v for t, v in d["trainL16"]["history"][-1]["val_curve"].items()}
    check("trainL16 plateau@0.01", plateau(c, 0.01)[:2], (8, 8))
    check("trainL16 mid/L == 0.50", round(plateau_mid(c, 0.01) / 16, 4), 0.5, tol=1e-9)
else:
    print("  [skip] train_at_L_results.json absent")

print("[8] FALSIFICATION: a deliberately broken tolerance test must fail the [1] check")
c = {1: 1.0, 2: 0.5, 4: 0.2, 8: 0.0, 16: 0.2, 32: 0.5}
broken = plateau(c, 0.25)[:2] == (8, 8)   # would be true only if tol were ignored
check("broken-tolerance detector fires (expects False)", broken, False)

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
