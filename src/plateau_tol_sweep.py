"""Do the report's BAND claims survive a change in the plateau tolerance?

`src/plateau.py` uses tol=0.01: the band is every swept depth whose CE is within 0.01 nats of the
curve minimum. That constant was chosen once, early, and never varied -- and EVERY band claim in the
report inherits it. This project retired `argmin` for exactly this class of reason (134 of 165 curves
had argmin margins under the noise floor), so leaving its replacement's one free parameter unswept was
the largest unexamined instrument choice left.

Verdict is on the END edge of the paired band (control -> arm), because that is what the report's
claims are about: widening = depth stays useful longer.

    python src/plateau_tol_sweep.py
"""
from __future__ import annotations
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from plateau import plateau
ROOT = pathlib.Path(__file__).resolve().parents[1]
TOLS = (0.005, 0.01, 0.02)
PAIRS = [
    ("ds_dc0.json", "dc_control_s0", "dc_w3_s0", "duo-causal W3 s0"),
    ("ds_dc1.json", "dc_control_s1", "dc_w3_s1", "duo-causal W3 s1"),
    ("ds_dc0.json", "dc_control_s0", "dc_w2_s0", "duo-causal W2 s0"),
    ("ds_dc1.json", "dc_control_s1", "dc_w2_s1", "duo-causal W2 s1"),
    ("ds_xsa.json", "xsa_control_s0", "xsa_on_s0", "XSA s0"),
    ("ds_xsa1.json", "xsa_control_s1", "xsa_on_s1", "XSA s1"),
    ("ds_div.json", "dv_control_s0", "dv_lora_r4_s0", "LoRA cycled s0"),
    ("ds_div.json", "dv_control_s0", "dv_lora_fixed0_s0", "LoRA pin-0 s0"),
    ("ds_divx1.json", "dx_control_s1", "dx_cycled_s1", "LoRA cycled s1"),
    ("ds_divx1.json", "dx_control_s1", "dx_pin2_s1", "LoRA pin-2 s1"),
    ("ds_rec2.json", "rec_dense_s2", "rec_sw90_s2", "annealing 10M"),
]

def curve(f, k):
    d = json.load(open(ROOT / "checkpoints" / f))
    return {int(a): b for a, b in d[k]["history"][-1]["val_curve"].items()}

def main():
    rows, robust, fragile = [], [], []
    for f, c, a, name in PAIRS:
        try:
            cc, aa = curve(f, c), curve(f, a)
        except Exception as e:
            print(f"  SKIP {name}: {e}"); continue
        verdicts = []
        for tol in TOLS:
            lc, la = plateau(cc, tol), plateau(aa, tol)
            verdicts.append("widens" if la[1] > lc[1] else "narrows" if la[1] < lc[1] else "same")
            rows.append(dict(pair=name, tol=tol, onset=[lc[0], la[0]], end=[lc[1], la[1]],
                             verdict=verdicts[-1]))
        (robust if len(set(verdicts)) == 1 else fragile).append((name, verdicts))
    print(f"  ROBUST across tol {TOLS} ({len(robust)}):")
    for n, v in robust: print(f"    {n:20s} {v[0]}")
    print(f"\n  TOLERANCE-DEPENDENT ({len(fragile)}):")
    for n, v in fragile: print(f"    {n:20s} {v}")
    (ROOT / "checkpoints" / "plateau_tol_sweep.json").write_text(json.dumps(rows, indent=2))
    print(f"\n  -> checkpoints/plateau_tol_sweep.json")

if __name__ == "__main__":
    main()
