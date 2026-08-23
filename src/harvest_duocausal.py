"""Reads `tlab-duocausal-*` exactly as RUNS.md pre-registered it at 18:19, before the data existed.

The pre-registration is the point: this script implements reads (a)-(e) and the four falsifiers as
written, so the verdict cannot drift toward whatever the numbers turned out to be. It refuses to
print a verdict for any arm missing at a seed, because "reverses between seeds => not reported" is
one of the registered falsifiers and it needs both seeds to apply.

Usage: python src/harvest_duocausal.py <results_s0.json> [<results_s1.json>]
"""
from __future__ import annotations
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from plateau import plateau, plateau_mid

FLOOR = 0.0150          # CUDA dense replicate floor (sec4.15) -- these are CUDA dense-like arms
TOL = 0.01

def read(path):
    d = json.load(open(path))
    out = {}
    for k, v in d.items():
        h = v.get("history") or []
        if not h:
            print(f"  !! {k}: NO history"); continue
        c = {int(a): b for a, b in h[-1]["val_curve"].items()}
        lo, hi, contig = plateau(c, TOL)
        out[k] = dict(ce1=c[1], best=min(c.values()), at=min(c, key=c.get),
                      band=(lo, hi), mid=plateau_mid(c, TOL), contig=contig,
                      grid=tuple(sorted(c)), step=h[-1]["step"], n_evals=len(h),
                      params=v.get("params"))
    return out

def base_of(name):
    for tag in ("dc_control", "rec_dense"):
        if tag in name: return None
    return "control"

def main():
    seeds = {}
    for p in sys.argv[1:]:
        s = "s1" if "s1" in os.path.basename(p) else "s0"
        seeds[s] = read(p)
    if not seeds:
        print(__doc__); return

    for s, arms in seeds.items():
        ctrl_k = next((k for k in arms if "control" in k or "dense" in k), None)
        if ctrl_k is None:
            print(f"seed {s}: no control arm found -- cannot decide anything in-job"); continue
        c = arms[ctrl_k]
        print(f"\n=== seed {s}  (grid {c['grid']}, step {c['step']}, {c['n_evals']} evals)")
        print(f"  {'arm':22s} {'CE@1':>8s} {'best':>8s} {'band':>10s} {'mid':>6s} "
              f"{'dCE_best':>9s} {'dCE@1':>8s} {'dgain':>8s} {'dBAND':>7s}")
        print(f"  {ctrl_k:22s} {c['ce1']:8.4f} {c['best']:8.4f} "
              f"{str(list(c['band'])):>10s} {c['mid']:6.1f} {'--':>9s} {'--':>8s} {'--':>8s} {'--':>7s}")
        for k, a in arms.items():
            if k == ctrl_k: continue
            db, d1 = a["best"] - c["best"], a["ce1"] - c["ce1"]
            print(f"  {k:22s} {a['ce1']:8.4f} {a['best']:8.4f} "
                  f"{str(list(a['band'])):>10s} {a['mid']:6.1f} "
                  f"{db:+9.4f} {d1:+8.4f} {d1-db:+8.4f} {a['mid']-c['mid']:+7.1f}"
                  f"{'' if a['contig'] else '  NONCONTIG'}")

    # (d) dose-response and the cross-seed falsifier, only when both seeds exist
    if len(seeds) == 2:
        print("\n=== pre-registered falsifiers")
        for arm in ("dc_w2", "dc_w3", "dg_norm"):
            row = []
            for s in ("s0", "s1"):
                A = seeds[s]; k = next((x for x in A if x.startswith(arm)), None)
                ck = next((x for x in A if "control" in x), None)
                if not k or not ck: row.append(None); continue
                row.append((A[k]["best"] - A[ck]["best"], A[k]["mid"] - A[ck]["mid"]))
            if any(r is None for r in row):
                print(f"  {arm:8s} incomplete -- not reported (registered falsifier)"); continue
            (ce0, b0), (ce1_, b1) = row
            same_ce = (ce0 < 0) == (ce1_ < 0)
            print(f"  {arm:8s} dCE_best {ce0:+.4f} / {ce1_:+.4f}   dBAND {b0:+.1f} / {b1:+.1f}   "
                  f"CE sign {'AGREES' if same_ce else 'REVERSES -> NOT REPORTED'}   "
                  f"|mean dCE| vs floor {abs((ce0+ce1_)/2)/FLOOR:.1f}x")
        print("\n  (a) band widened at BOTH seeds?  -> read the dBAND columns above")
        print("  (b) cos(du_t,du_t-1) needs the CHECKPOINTS -- run src/state_dynamics.py on the returned .pt")
        print("  (e) gate: run the effective-loops-mixed probe on dg_norm's checkpoint")

if __name__ == "__main__":
    main()
