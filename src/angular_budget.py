"""CHORD, not arc -- read this first.

This file samples the state ONCE PER LOOP, so `B = sum ||u_t - u_(t-1)||` is a CHORD sum, not a
path length. Sampling the same trajectory at LAYER resolution (3x per loop) reverses the sign of
the dense-vs-terminal comparison: chord gives 1.20, arc gives 0.80 (report.md sec4.16c
correction (c)). Neither is "the angular budget" without saying which. Any interpretive verdict
printed below predates that correction and should not be quoted.

Does terminal-only supervision buy a larger angular BUDGET, or only a slower RATE?

This is a two-way discrimination on the report's central positive claim, and it needs no training.

§4.6 found that clamping the state's radius relocates the loop optimum without raising the ceiling,
and that the implied angular budget agrees to 0.2% across two clamp levels (0.3325 vs 0.3317). The
reading: a trained model traverses a roughly fixed angular distance of useful computation, and scale
only sets the step size. §4.3 gives the geometry -- the readout sees direction only (final_norm is
scale-invariant), and the angular step decays as ~1/t because ||h|| grows linearly.

If that budget is a property of the TRAINED MODEL, then for any checkpoint

    B = sum_{t=1..k*} || u_t - u_{t-1} ||,    u_t = h_t / ||h_t||,    k* = that model's useful depth

should be comparable across models that differ only in how fast they spend it -- and should DIFFER
for a model that genuinely has more useful computation to do.

Applied to §4.14's dense-vs-terminal pair (both seeds), this separates two readings that the CE
numbers cannot:

  B INCREASES under terminal-only  -> it buys MORE useful angular computation. Supervision changes
                                      the budget, which puts it in a different category from the
                                      radial clamp / convex gate / residual scaling nulls.
  B CONSTANT, rate falls           -> terminal-only only slows traversal. It is the same KIND of
                                      thing as the three traversal nulls, and §3.5's positive claim
                                      collapses into a fourth one.

Reports the cumulative curve too, so "where does it saturate" is visible rather than assumed.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval import load_checkpoint          # noqa: E402
from plateau import plateau, plateau_mid  # noqa: E402


def angular_path(model, x, max_loops, device):
    """Cumulative angular distance of the UNIT state, per loop. Returns (steps, cumulative)."""
    with torch.no_grad():
        out = model(x, n_loops=max_loops, supervise_idx=set(), return_states=True)
    states = out[2]                                    # list[Tensor], one per loop, detached
    u = [s.float() / s.float().norm(dim=-1, keepdim=True).clamp_min(1e-12) for s in states]
    steps = []
    for t in range(1, len(u)):
        # mean over batch and positions of the per-token angular step
        steps.append((u[t] - u[t - 1]).norm(dim=-1).mean().item())
    return np.array(steps), np.cumsum(steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--max-loops", type=int, default=32)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fixed-range", type=int, default=18,
                    help="also report B over a FIXED loop range, so the budget is not confounded "
                         "with where each arm's optimum sits (k* carries a 17%% grid sensitivity)")
    ap.add_argument("--untrained", action="store_true",
                    help="also measure a randomly-initialised model of the same shape: the control "
                         "that separates 'the budget is learned' from 'the budget is architectural'")
    ap.add_argument("--curves", default=str(ROOT / "checkpoints"))
    a = ap.parse_args()

    val = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(val) - a.seq - 2, size=a.batches)
    x = torch.from_numpy(np.stack([val[i:i + a.seq].astype(np.int64) for i in idx])).to(a.device)

    rows = []
    for cpath in a.ckpts:
        d = pathlib.Path(cpath)
        name = d.name
        model = load_checkpoint(d / "last.pt", a.device)[0]
        model.eval()
        steps, cum = angular_path(model, x, a.max_loops, a.device)

        # k* from that model's OWN stored loop curve, via the plateau statistic (argmin is retired)
        kstar, src = None, "n/a"
        for f in pathlib.Path(a.curves).glob("*results*.json"):
            try:
                j = json.load(open(f))
            except Exception:
                continue
            if isinstance(j, dict) and name in j and "history" in j[name]:
                c = {int(t): v for t, v in j[name]["history"][-1]["val_curve"].items()}
                kstar, src = plateau_mid(c), f.name
                break
        k = int(round(kstar)) if kstar else 8
        k = max(1, min(k, len(cum)))
        kf = min(a.fixed_range, len(cum))
        rows.append(dict(name=name, kstar=kstar, B=float(cum[k - 1]), k_used=k,
                         B_fixed=float(cum[kf - 1]),
                         step1=float(steps[0]), step_last=float(steps[-1]),
                         B_full=float(cum[-1]), src=src))
        print(f"{name:<22} k*={kstar if kstar else 0:>5.1f}  B(k*)={cum[k-1]:.4f}  "
              f"B(1..{kf})={cum[kf-1]:.4f}  B(all)={cum[-1]:.4f}  "
              f"step1={steps[0]:.4f} steplast={steps[-1]:.5f}")

    if a.untrained:
        # THE CONTROL THE PRIOR PROJECT LEARNED TO DEMAND: is the budget built by training, or is it
        # a property of the architecture? At init, increments should be near-orthogonal (cos ~ 0)
        # and the accumulated angular distance correspondingly different.
        import model as _m
        mdl0 = _m.LoopedTransformer(_m.Config(state_renorm=False)).to(a.device).eval()
        st0, cu0 = angular_path(mdl0, x, a.max_loops, a.device)
        kf = min(a.fixed_range, len(cu0))
        print(f"{'UNTRAINED (init)':<22} {'':>5}  {'':>14}  B(1..{kf})={cu0[kf-1]:.4f}  "
              f"B(all)={cu0[-1]:.4f}  step1={st0[0]:.4f} steplast={st0[-1]:.5f}")
        import numpy as _np
        print(f"  mean per-step angle at init: {_np.degrees(2*_np.arcsin(_np.clip(st0/2,0,1))).mean():.1f}deg")
        rows.append(dict(name="UNTRAINED", kstar=None, B=float(cu0[kf-1]), k_used=kf,
                         B_fixed=float(cu0[kf-1]), step1=float(st0[0]), step_last=float(st0[-1]),
                         B_full=float(cu0[-1]), src="init"))

    print("\n--- paired comparison (dense vs terminal at matched seed) ---")
    by = {r["name"]: r for r in rows}
    for s in (0, 1):
        dn, tm = by.get(f"sd_dense_k5_s{s}"), by.get(f"sd_terminal_k1_s{s}")
        if dn and tm:
            print(f"  seed {s}: at own k*  B {dn['B']:.4f} -> {tm['B']:.4f}  ratio={tm['B']/dn['B']:.3f}"
                  f"   |  FIXED range 1..{dn['k_used'] and a.fixed_range}  "
                  f"B {dn['B_fixed']:.4f} -> {tm['B_fixed']:.4f}  ratio={tm['B_fixed']/dn['B_fixed']:.3f}")
    print("\n  ratio ~1.0 -> terminal-only spends the SAME budget more slowly (a rate intervention,")
    print("               same category as the clamp / gate / residual-scaling nulls)")
    print("  ratio >1   -> terminal-only buys MORE useful angular computation (a budget intervention)")


if __name__ == "__main__":
    main()
