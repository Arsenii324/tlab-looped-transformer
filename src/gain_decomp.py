"""Does an intervention raise loop gain by improving DEPTH, or by damaging LOOP 1?

Loop gain is a difference, `CE@1 - CE_best`, so it rises either when the deep end improves or when the
shallow end degrades. Those are opposite mechanisms and this report has been quoting the difference
without separating them. The identity is exact:

    Delta_gain = Delta_CE@1 - Delta_CE_best

so for any paired comparison the two contributions can be read off directly. Classification used here:

    DEPTH-DRIVEN    the deep end improves and supplies most of the change (|dCE_best| > |dCE@1|)
    DAMAGE-DRIVEN   loop 1 degrades and supplies most of it -- the gain is inflated, not earned
    BOTH-IMPROVE    both endpoints improve; the gap widens on merit. The strongest outcome.
    BOTH-WORSEN     both endpoints degrade

Surfaced by an external reviewer, who spotted it on the 90M norm-penalty pair. Verified below.
"""
from __future__ import annotations
import json, glob, os, sys
sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.join(os.path.dirname(__file__), "..")


def curve_of(path, key=None):
    d = json.load(open(path))
    if key:
        d = d[key]
    if "history" in d and d["history"]:
        return {int(t): v for t, v in d["history"][-1]["val_curve"].items()}
    if "val_ce" in d:
        return {int(t): float(v) for t, v in d["val_ce"].items()}
    raise KeyError(path)


def decomp(ctl, arm, label, shared_grid=True):
    if shared_grid:
        g = sorted(set(ctl) & set(arm))
        ctl = {t: ctl[t] for t in g}; arm = {t: arm[t] for t in g}
    c1, a1 = ctl[min(ctl)], arm[min(arm)]
    cb, ab = min(ctl.values()), min(arm.values())
    d1, db = a1 - c1, ab - cb
    dg = d1 - db
    if db < 0 and d1 < 0:
        cls = "BOTH-IMPROVE"
    elif db > 0 and d1 > 0:
        cls = "BOTH-WORSEN"
    elif abs(db) >= abs(d1):
        cls = "DEPTH-DRIVEN"
    else:
        cls = "DAMAGE-DRIVEN"
    share = abs(d1) / (abs(d1) + abs(db)) * 100 if (abs(d1) + abs(db)) else 0
    return dict(label=label, dCE_best=db, dCE_at1=d1, dgain=dg, cls=cls, share_from_loop1=share)


def show(rows):
    print(f"{'comparison':<42} {'dCE_best':>9} {'dCE@1':>9} {'dgain':>8} {'loop1 share':>11}  class")
    for r in rows:
        print(f"{r['label']:<42} {r['dCE_best']:>+9.4f} {r['dCE_at1']:>+9.4f} {r['dgain']:>+8.4f} "
              f"{r['share_from_loop1']:>10.0f}%  {r['cls']}")


if __name__ == "__main__":
    R, rows = ROOT, []
    P = os.path.join
    # 90M norm penalty vs its seed-matched control (the reviewer's case)
    rows.append(decomp(curve_of(P(R,"checkpoints/full_control90_kaggle/results.json"),"no_state_renorm"),
                       curve_of(P(R,"checkpoints/full_normpen_kaggle/results.json"),"no_state_renorm_normpen"),
                       "90M: norm penalty vs control"))
    # annealing, in-job controls, both seeds
    d = json.load(open(P(R,"checkpoints/anneal_rep2_results.json")))
    for s in (0,1):
        for tag in ("sw90","sw75"):
            rows.append(decomp({int(x):y for x,y in d[f"a3_dense_s{s}"]["history"][-1]["val_curve"].items()},
                               {int(x):y for x,y in d[f"a3_{tag}_s{s}"]["history"][-1]["val_curve"].items()},
                               f"mu18 s{s}: anneal {tag} vs in-job dense"))
    # terminal-only vs dense, in-job, three schedules
    for f,ct,ar,lab in (("checkpoints/deep_terminal_results.json","dt_mu18_dense","dt_mu18_term","mu18: terminal vs dense"),
                        ("checkpoints/deep_terminal_results.json","dt_mu32_dense","dt_mu32_term","mu32: terminal vs dense"),
                        ("checkpoints/deep_mu40_results.json","d3_mu40_dense","d3_mu40_term","mu40: terminal vs dense")):
        rows.append(decomp(curve_of(P(R,f),ct), curve_of(P(R,f),ar), lab))
    # scale-control readout interventions (2.5M, seed 0)
    sc = json.load(open(P(R,"checkpoints/scale_control_results.json")))
    base = {int(x):y for x,y in sc["sc_control_norm_s0"]["history"][-1]["val_curve"].items()}
    for k,lab in (("sc_raw_s0","2.5M: raw readout vs norm"),("sc_final_only_s0","2.5M: final-only vs norm"),
                  ("sc_penalty_s0","2.5M: norm penalty vs norm")):
        if k in sc:
            rows.append(decomp(base, {int(x):y for x,y in sc[k]["history"][-1]["val_curve"].items()}, lab))
    show(rows)
    print("\nDAMAGE-DRIVEN rows raise loop gain by degrading loop 1. Any claim resting on their gain"
          "\nincrease needs restating; BOTH-IMPROVE rows widen the gap on merit.")
