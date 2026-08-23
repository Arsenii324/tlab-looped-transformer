"""Is 'loop gain trades against CE' a real correlation, or a story told over a few hand-picked arms?

The report asserts this split in §4.5, §4.9, §4.11, §4.12 and §4.14, each time from one comparison.
That is exactly the pattern that produces a narrative rather than a finding. Every arm this project
has stored is now pooled and the correlation computed, within device+budget strata so that the
trivial confounds (more tokens -> lower CE; deeper schedule -> higher gain) do not manufacture it.

Reports Spearman rho (rank, so it does not assume linearity) per stratum and pooled-within-stratum.
"""
from __future__ import annotations
import json, glob, os, sys, math
sys.path.insert(0, os.path.dirname(__file__))
from plateau import plateau_mid


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):                       # average ties, else rho is wrong on flat data
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else None


rows = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "*results*.json"))):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    for k, v in d.items():
        if not (isinstance(v, dict) and "history" in v and v["history"]):
            continue
        h = v["history"][-1]
        if "val_curve" not in h:
            continue
        c = {int(t): x for t, x in h["val_curve"].items()}
        if len(c) < 3 or 1 not in c:
            continue
        t = v.get("train_cfg", {})
        best = min(c.values())
        rows.append(dict(arm=k, dev=t.get("device", "?"), tok=t.get("total_tokens", 0),
                         ce=best, gain=c[1] - best, mid=plateau_mid(c)))

print(f"pooled {len(rows)} arms\n")
strata = {}
for r in rows:
    strata.setdefault((r["dev"], r["tok"]), []).append(r)

allx, ally = [], []
for (dev, tok), rs in sorted(strata.items(), key=lambda kv: -len(kv[1])):
    if len(rs) < 3:
        continue
    x = [r["gain"] for r in rs]
    y = [r["ce"] for r in rs]
    rho = spearman(x, y)
    print(f"stratum device={dev} tokens={tok:,}  n={len(rs)}  spearman(gain, best CE) = "
          f"{rho:+.3f}" if rho is not None else "")
    for r in sorted(rs, key=lambda r: r["gain"]):
        print(f"    {r['arm']:<26} gain={r['gain']:.4f}  CE={r['ce']:.4f}  mid={r['mid']:.1f}")
    # centre within stratum so strata can be pooled without their level differences driving rho
    mx, my = sum(x) / len(x), sum(y) / len(y)
    allx += [a - mx for a in x]
    ally += [b - my for b in y]
    print()

rho = spearman(allx, ally)
print(f"POOLED WITHIN STRATA: n={len(allx)}  spearman(gain, best CE) = {rho:+.3f}"
      if rho is not None else "insufficient data")
print("positive rho => higher loop gain goes with WORSE (higher) CE, i.e. the split is real")
