"""Re-derive the screening comparison from the actual results JSON, not from memory of watching the
run. Prints the per-arm final per-loop val CE curve and a same-seed delta against the center arm
(all arms share seed=0 for model init, data sampling, and loop-count sampling -- see run_screening.py
-- so a same-seed delta is meaningful here, not just a coincidence of matched randomness).
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "checkpoints" / "screening_results.json"


def main():
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_PATH
    if not path.exists():
        print(f"no results yet at {path}")
        return 1
    results = json.loads(path.read_text())

    print(f"{len(results)} arms\n")
    center_curve = None
    if "center" in results and results["center"]["history"]:
        center_curve = {int(k): v for k, v in results["center"]["history"][-1]["val_curve"].items()}

    rows = []
    for name, r in results.items():
        if not r["history"]:
            print(f"{name}: NO EVAL LOGGED (arm may not have started or crashed)")
            continue
        last = r["history"][-1]
        curve = {int(k): v for k, v in last["val_curve"].items()}
        best_r = min(curve, key=curve.get)
        rows.append((name, curve, best_r, last["tokens"], r["elapsed_s"]))

    print(f"{'arm':<18}{'tokens':>10}{'best_r':>8}{'best_CE':>10}{'r1_CE':>9}{'delta_vs_center':>16}")
    for name, curve, best_r, tokens, elapsed in rows:
        delta = ""
        if center_curve and name != "center":
            d = curve[best_r] - center_curve.get(best_r, float("nan"))
            delta = f"{d:+.4f}"
        print(f"{name:<18}{tokens/1e6:>9.2f}M{best_r:>8}{curve[best_r]:>10.4f}{curve.get(1, float('nan')):>9.4f}"
              f"{delta:>16}")

    print("\nfull per-loop curves:")
    for name, curve, *_ in rows:
        print(f"  {name:<18} " + " ".join(f"r{r}={v:.3f}" for r, v in sorted(curve.items())))

    print("\nREADING")
    print("  Lower CE at higher r than at r=1 is the headline signal (loops keep helping).")
    print("  A curve that's flat or U-shaped (best at low r, worse at high r) means this arm's loops")
    print("  are not yet paying off at this token budget -- could be undertrained, not necessarily a")
    print("  genuine negative; check whether the FULL-budget run of a promising axis changes this.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
