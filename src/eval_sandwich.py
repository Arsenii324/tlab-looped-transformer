"""Post-hoc dense per-loop sweep over the four sandwich arms, at matched LAYER-APPLICATION budget.

Two reasons this exists rather than reading run_sandwich.py's in-training numbers:

  1. Coverage. The in-training eval grid is fixed at {1,2,4,8,12,16,24,32} and does not reach the
     R1 arm's trained max of 96 loops -- so the cheap grid cannot see that arm inside the range it
     was actually trained on. Reporting a "best loop count" from a grid that stops at a third of the
     trained range would be measuring the grid, not the model.
  2. Resolution. In-training evals are 6-batch estimates; report.md sec 4.2 already records that these
     disagree with dedicated post-hoc evals by up to ~0.061 nats, larger than several effects here.

Matched budget, not matched loop count: an R1 arm's loop is a third the depth of an R3 arm's, so
comparing both at "loop 32" compares 32 layer-applications against 96. Each arm is swept to
192/layers_per_loop loops, i.e. 192 layer-applications for everyone, which is 2x the trained ceiling
in every arm. Both readings are then available -- per-loop (does looping help?) and per-unit-compute
(is this topology worth its cost?).

Usage: python src/eval_sandwich.py
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPTH_BUDGET = 192          # layer-applications every arm is swept to
ARMS = {"sand_P0R3C0": 3, "sand_P1R1C1": 1, "sand_P1R2C0": 2, "sand_P0R2C1": 2}
COOLDOWN_S = 45             # MPS driver dislikes back-to-back GPU workloads (report.md sec 6)


def main():
    out = {}
    for i, (name, L) in enumerate(ARMS.items()):
        ckpt = ROOT / "checkpoints" / name
        if not (ckpt / "last.pt").exists():
            print(f"SKIP {name}: no checkpoint", flush=True)
            continue
        if i:
            time.sleep(COOLDOWN_S)
        max_loops = DEPTH_BUDGET // L
        print(f"=== {name}: layers_per_loop={L}, sweeping to {max_loops} loops "
              f"({max_loops * L} layer-applications) ===", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "src" / "eval.py"), str(ckpt),
                            "--max-loops", str(max_loops), "--batch-size", "8",
                            "--n-batches", "24"],
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout[-2500:])
        if r.returncode != 0:
            print(f"FAILED {name}: {r.stderr[-1500:]}", flush=True)
            continue
        ev = json.loads((ckpt / f"eval_{name}.json").read_text())
        ce = {int(k): v for k, v in ev["val_ce"].items()}
        best = min(ce, key=ce.get)
        out[name] = dict(layers_per_loop=L, max_loops=max_loops, best_loop=best,
                          best_ce=ce[best], ce_at_1=ce[1], loop_gain=ce[1] - ce[best],
                          best_depth=best * L, val_ce=ev["val_ce"],
                          val_bits_per_byte=ev["val_bits_per_byte"], tokens=ev["tokens"])
        (ROOT / "checkpoints" / "sandwich_eval.json").write_text(json.dumps(out, indent=2))

    print(f"\n{'arm':14} {'L':>2} {'best_r':>7} {'depth':>6} {'best_CE':>9} {'CE@r1':>8} "
          f"{'loop_gain':>10} {'bpb':>7}")
    for name, d in out.items():
        print(f"{name:14} {d['layers_per_loop']:>2} {d['best_loop']:>7} {d['best_depth']:>6} "
              f"{d['best_ce']:>9.4f} {d['ce_at_1']:>8.4f} {d['loop_gain']:>10.4f} "
              f"{d['val_bits_per_byte'][str(d['best_loop'])]:>7.4f}")
    print(f"\nwrote {ROOT/'checkpoints'/'sandwich_eval.json'}")


if __name__ == "__main__":
    main()
