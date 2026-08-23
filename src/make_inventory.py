"""Regenerate submission/EXPERIMENTS.md from the stored artifacts.

Exists so the inventory cannot drift from the JSON it describes: the table is derived, never
transcribed. Also re-runs the coverage check (which arms appear in report.md by identifier, by
best-CE value, or by neither), because "no experiment went unreported" is a claim and this project
does not ship claims it has not measured.

    python src/make_inventory.py            # writes submission/EXPERIMENTS.md
    python src/make_inventory.py --check    # coverage only, writes nothing
"""
from __future__ import annotations
import argparse, glob, json, os, pathlib, sys
sys.path.insert(0, os.path.dirname(__file__))
from plateau import plateau, plateau_mid
ROOT = pathlib.Path(__file__).resolve().parents[1]

def collect():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "checkpoints" / "*.json"))):
        try: d = json.load(open(f))
        except Exception: continue
        if not isinstance(d, dict): continue
        for k, v in d.items():
            if not (isinstance(v, dict) and isinstance(v.get("history"), list) and v["history"]):
                continue
            h = v["history"][-1]
            if "val_curve" not in h: continue
            c = {int(a): b for a, b in h["val_curve"].items()}
            tc = v.get("train_cfg") or {}
            try:
                lo, hi, _ = plateau(c, 0.01); band, mid = f"[{lo},{hi}]", f"{plateau_mid(c,0.01):.1f}"
            except Exception:
                band, mid = "-", "-"
            rows.append(dict(arm=k, src=os.path.basename(f)[:-5], dev=tc.get("device", "cuda"),
                             tok=h.get("tokens") or tc.get("total_tokens"),
                             lo=tc.get("min_train_loops"), hi=tc.get("max_train_loops"),
                             k=tc.get("supervise_k"), ce1=c.get(1), best=min(c.values()),
                             at=min(c, key=c.get), band=band, mid=mid, grid=len(c)))
    rows.sort(key=lambda r: (r["src"], r["arm"]))
    return rows

def coverage(rows):
    rep = (ROOT / "report.md").read_text()
    def seen(x):
        return any(f"{x:.{d}f}" in rep for d in (4, 3)) if x is not None else False
    absent = [r for r in rows if r["arm"] not in rep and not seen(r["best"])]
    return (sum(1 for r in rows if r["arm"] in rep), sum(1 for r in rows if seen(r["best"])), absent)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rows = collect(); byname, bynum, absent = coverage(rows)
    print(f"arms: {len(rows)}   by identifier: {byname}   by best-CE: {bynum}   "
          f"by neither: {len(absent)}")
    for r in absent: print(f"   ABSENT  {r['arm']:26s} {r['src']:32s} best={r['best']:.4f}")
    if a.check: return
    out = [f"Total arms with a final validation curve: **{len(rows)}**\n",
           "| # | arm | source | dev | tokens | loops | k | CE@1 | best CE | @r | band | mid | grid |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        tok = f"{r['tok']/1e6:.2f}M" if isinstance(r["tok"], (int, float)) else "?"
        loops = f"{r['lo']}-{r['hi']}" if r["lo"] is not None else "?"
        out.append(f"| {i} | `{r['arm']}` | {r['src']} | {r['dev']} | {tok} | {loops} | "
                   f"{r['k'] if r['k'] is not None else '?'} | {r['ce1']:.4f} | **{r['best']:.4f}** | "
                   f"{r['at']} | {r['band']} | {r['mid']} | {r['grid']} |")
    p = ROOT / "submission" / "EXPERIMENTS.md"
    head = p.read_text().split("## The inventory")[0] if p.exists() else "# All experiments\n\n## The inventory\n"
    p.write_text(head + "## The inventory\n\n" + "\n".join(out) + "\n")
    print(f"wrote {p}")

if __name__ == "__main__":
    main()
