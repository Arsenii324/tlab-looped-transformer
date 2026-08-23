"""Sweep every stored loop curve and flag the ones whose argmin is not resolvable.

Motivated by a near-miss: the residual_scale arms' argmin says the optimum moved 8 -> 12 (which
would have been a second t/L-breaking intervention); the curves are tied to 1e-4 nats there. This
script asks the same question of every curve this project has stored, so the answer is systematic
rather than whichever result I happened to open.

Flag levels, against margin = CE(runner-up grid point) - CE(argmin):
    FRAGILE   margin < 0.005   -- argmin is noise; any claim about "the optimum" is unsupported
    WEAK      margin < 0.010   -- state as a plateau, not a point
    OK        otherwise
Emits the plateau band alongside, since that is what a FRAGILE curve should be reported as instead.
"""
from __future__ import annotations
import json, glob, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from plateau import plateau, plateau_mid


def curves_in(path):
    """Yield (label, {loop: ce}) from either results-style or history-style JSON."""
    try:
        d = json.load(open(path))
    except Exception:
        return
    base = os.path.basename(path).replace(".json", "")
    if isinstance(d, list):
        if d and isinstance(d[-1], dict) and "val_curve" in d[-1]:
            yield base, {int(t): v for t, v in d[-1]["val_curve"].items()}
        return
    if not isinstance(d, dict):
        return
    for k, v in d.items():
        if isinstance(v, dict) and "history" in v and v["history"]:
            h = v["history"][-1]
            if isinstance(h, dict) and "val_curve" in h:
                yield f"{base}:{k}", {int(t): c for t, c in h["val_curve"].items()}
    if "history" in d and d["history"] and "val_curve" in d["history"][-1]:
        yield base, {int(t): c for t, c in d["history"][-1]["val_curve"].items()}
    # Post-hoc dense sweeps store the curve under `val_ce` with no history wrapper. Missing this
    # shape silently skipped sandwich_eval.json -- the very file §4.5's table is computed from,
    # and the one whose argmins turned out to have 0.0000-nat margins. A coverage gap in an audit
    # reads exactly like a clean audit, so shapes are enumerated explicitly here.
    for k, v in d.items():
        if isinstance(v, dict) and "val_ce" in v and isinstance(v["val_ce"], dict):
            yield f"{base}:{k}", {int(t): c for t, c in v["val_ce"].items()}


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "*.json"))):
        for label, c in curves_in(f):
            if len(c) < 3:
                continue
            am = min(c, key=c.get)
            second = min((t for t in c if t != am), key=lambda t: c[t])
            margin = c[second] - c[am]
            lo, hi, contig = plateau(c, 0.01)
            flag = "FRAGILE" if margin < 0.005 else ("WEAK" if margin < 0.010 else "OK")
            rows.append((flag, margin, label, am, lo, hi, plateau_mid(c), contig))
    rows.sort(key=lambda r: (r[0] != "FRAGILE", r[0] != "WEAK", r[1]))
    print(f"{'flag':<8} {'margin':>8}  {'argmin':>6} {'plateau@0.01':>14} {'mid':>6}  curve")
    for flag, m, label, am, lo, hi, mid, contig in rows:
        print(f"{flag:<8} {m:>8.4f}  {am:>6} {'['+str(lo)+','+str(hi)+']':>14} {mid:>6.1f}  {label}"
              + ("" if contig else "   NON-CONTIGUOUS"))
    n = len(rows)
    nf = sum(1 for r in rows if r[0] == "FRAGILE")
    nw = sum(1 for r in rows if r[0] == "WEAK")
    print(f"\n{nf}/{n} curves have an UNRESOLVABLE argmin (<0.005), {nw} more are weak (<0.010).")


if __name__ == "__main__":
    main()
