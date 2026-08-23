"""Does the t/L collapse (§4.9) survive a second seed, or was it luck?

The worry, stated plainly in LOG.md: §4.9's collapse rests on five single-seed arms, and the
spread of their curves at t/L=2 is 0.0225 nats -- BELOW the ~0.06 single-arm seed noise measured
independently in §4.10. A cross-arm spread tighter than the noise of a single arm is either a
real structural collapse or a coincidence, and only a second seed tells them apart.

Three quantities, because they answer different questions and get conflated:
  [A] seed noise at matched (L, t): |CE_s0 - CE_s1|, computed BOTH on raw CE and on curves
      re-zeroed at t/L=1. These are different yardsticks and the distinction is the whole point:
      re-zeroing removes the common vertical offset between two seeds, which is most of what raw
      seed noise IS. §4.9 as published compares a re-zeroed spread (0.0225) against a raw-CE noise
      figure (~0.06 from §4.10's fixed-g sweep, which is a spread in absolute best CE) and concludes
      the collapse is tighter than noise. That comparison is apples-to-oranges. The like-for-like
      raw cross-L spread is 0.058-0.124 -- i.e. NOT below raw seed noise. Only [A-rezeroed] can
      license the collapse claim, and it needs seed 1.
  [B] cross-L spread at matched t/L, per seed. This is the collapse statistic from §4.9.
  [D] the same, re-zeroed at t/L=1 -- THE REPORT'S OWN NORMALIZATION (§4.9 subtracts each arm's CE
      at its own trained L). Reproducing §4.9's published table exactly is what licenses using this
      script on seed 1 at all; a collapse statistic that cannot re-derive the number it is meant to
      re-test is not an instrument.
  [C] the same, on SHAPE (each curve re-zeroed to its own minimum). The collapse claim is about
      the shape of the depth curve, not its height; a common vertical offset between arms would
      inflate [B] while leaving the collapse intact, so [C] is the honest version of the test.

Verdict rule, fixed BEFORE looking (so it can't be fitted to the answer):
  collapse REPRODUCES if [C] stays below [A] in both seeds independently.
  collapse was LUCK      if [C] in seed 1 is comparable to or larger than the cross-L spread you
                            would get by pairing arms at random.
"""
from __future__ import annotations
import json, sys, itertools, statistics as st


def curves(path):
    d = json.load(open(path))
    out = {}
    for k, v in d.items():
        if not k.startswith("trainL"):
            continue
        # seed-1 arms are named trainL16_s1; strip any suffix after the loop count
        import re as _re
        m = _re.match(r"trainL(\d+)", k)
        if not m:
            continue
        L = int(m.group(1))
        vc = v["history"][-1]["val_curve"]
        out[L] = {int(t): ce for t, ce in vc.items()}
    return out


def at_ratio(c, L, r):
    """CE at t = r*L, only when that t was actually evaluated. No interpolation: inventing
    intermediate points is exactly how a collapse gets manufactured."""
    t = r * L
    return c.get(t) if float(t).is_integer() else None


def main(p0, p1=None):
    cs = {0: curves(p0)}
    if p1:
        cs[1] = curves(p1)
    Ls = sorted(cs[0])
    print(f"arms: L = {Ls}   seeds: {sorted(cs)}")

    # [A] seed noise at matched (L,t) -- raw AND re-zeroed. Only the latter is the right
    # yardstick for a re-zeroed collapse statistic.
    A, A_rz = [], []
    if 1 in cs:
        for L in Ls:
            if L not in cs[1]:
                continue
            b0, b1 = cs[0][L].get(L), cs[1][L].get(L)
            for t in sorted(set(cs[0][L]) & set(cs[1][L])):
                A.append(abs(cs[0][L][t] - cs[1][L][t]))
                if b0 is not None and b1 is not None:
                    A_rz.append(abs((cs[0][L][t] - b0) - (cs[1][L][t] - b1)))
        print(f"\n[A] seed noise at matched (L,t), RAW:       n={len(A)}  "
              f"median={st.median(A):.4f}  mean={sum(A)/len(A):.4f}  max={max(A):.4f}")
        print(f"[A] seed noise at matched (L,t), RE-ZEROED: n={len(A_rz)}  "
              f"median={st.median(A_rz):.4f}  mean={sum(A_rz)/len(A_rz):.4f}  max={max(A_rz):.4f}"
              f"\n    ^ THIS is the yardstick for [D]'s spreads, not the 0.06 raw figure from §4.10.")
    else:
        print("\n[A] seed noise: seed-1 data not present yet")

    # [B]/[C] cross-L spread at matched t/L, raw and shape-only
    # [D] first: exact reproduction of §4.9's table is the instrument's null.
    print("\n[D] re-zeroed at t/L=1 (report §4.9's normalization) -- seed 0 must match the published row")
    PUBLISHED = {0.5: 0.0373, 2: 0.0225, 4: 0.0633}   # from report.md §4.9 'spread' column
    for r in (0.5, 1, 2, 4, 8):
        cells = []
        for s_ in sorted(cs):
            vals = []
            for L in sorted(cs[s_]):
                v, base = at_ratio(cs[s_][L], L, r), cs[s_][L].get(L)
                if v is not None and base is not None:
                    vals.append(v - base)
            cells.append((s_, vals))
        line = f"  t/L={r:<4}"
        for s_, vals in cells:
            line += f"  seed{s_}: mean={sum(vals)/len(vals):+.4f} spread={max(vals)-min(vals):.4f} (n={len(vals)})" if len(vals) > 1 else f"  seed{s_}: --"
        if r in PUBLISHED and cells and len(cells[0][1]) > 1:
            got = max(cells[0][1]) - min(cells[0][1])
            line += f"   [published {PUBLISHED[r]:.4f}, diff {abs(got-PUBLISHED[r]):.1e}]"
        print(line)

    for label, shape in (("[B] raw CE", False), ("[C] shape (re-zeroed to each arm's own min)", True)):
        print(f"\n{label}")
        print(f"  {'t/L':>5}  " + "  ".join(f"seed{s}_spread" for s in sorted(cs)) + "   n_arms")
        for r in (0.25, 0.5, 1, 2, 4):
            row = []
            n_arms = None
            for s in sorted(cs):
                vals = []
                for L in sorted(cs[s]):
                    v = at_ratio(cs[s][L], L, r)
                    if v is None:
                        continue
                    if shape:
                        v = v - min(cs[s][L].values())
                    vals.append(v)
                n_arms = len(vals)
                row.append(f"{max(vals)-min(vals):.4f}" if len(vals) > 1 else "  --  ")
            print(f"  {r:>5}  " + "  ".join(f"{x:>12}" for x in row) + f"   {n_arms}")

    # verdict
    if 1 in cs:
        yard = st.median(A_rz) if A_rz else st.median(A)
        worst = 0.0
        for s in sorted(cs):
            for r in (0.5, 1, 2):
                vals = [at_ratio(cs[s][L], L, r) - min(cs[s][L].values())
                        for L in sorted(cs[s]) if at_ratio(cs[s][L], L, r) is not None]
                if len(vals) > 1:
                    worst = max(worst, max(vals) - min(vals))
        print(f"\nVERDICT: worst shape-spread over t/L in {{0.5,1,2}} = {worst:.4f} vs seed noise "
              f"{yard:.4f} -> collapse {'REPRODUCES' if worst < yard else 'DOES NOT reproduce'} "
              f"under the pre-registered rule")


if __name__ == "__main__":
    main(*sys.argv[1:])
