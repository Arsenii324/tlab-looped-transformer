"""Halt on CUMULATIVE angular distance rather than an instantaneous step.

§4.7 tested four label-free rule families and all failed: predictive entropy, logit margin,
successive-output KL, and the update-norm ratio ||dh||/||h||. The last of those halted at mean depth
2.03 because the instantaneous quantity collapses almost immediately.

But §4.6 says the object that governs where useful computation ends is the **angular budget** -- the
CUMULATIVE distance travelled on the unit sphere -- not the size of any one step. Two clamp levels
agreed on that budget to 0.2% (0.3325 vs 0.3317), and §4.16c since measured it rising 1.4x under
terminal-only. So the four rules were tested in the wrong coordinates: they read the derivative when
the account says the integral is what matters.

    halt(i) = min { k : sum_{t<=k} dnorm[i, t] >= tau }

Zero compute -- a cumsum over an array already on disk, plus a threshold sweep.

PREDICTION, written before running: if the angular budget is roughly constant ACROSS TOKENS the way
it is across clamp levels, a single tau should recover a large share of the per-token oracle where
every point statistic failed. If it does not, the budget varies per token -- and then the per-token
budget becomes the object to predict, which is a cleaner target than argmin-depth and is exactly what
§8.2's rate-control proposal would need anyway. Both outcomes are informative.
"""
from __future__ import annotations
import numpy as np, pathlib, sys

Z = pathlib.Path("checkpoints/full_no_state_renorm_kaggle/exitdump_full_no_state_renorm_kaggle.npz")


def main():
    z = np.load(Z)
    ce, dn = z["ce"], z["dnorm"]                      # [S, T, R]
    S, T, R = ce.shape
    ce2 = ce.reshape(-1, R)
    dn2 = dn.reshape(-1, R)
    n = ce2.shape[0]

    const = ce2.mean(0)                               # best fixed depth
    k_const = int(const.argmin())
    ce_const = float(const.min())
    ce_oracle = float(ce2.min(1).mean())              # per-token oracle
    print(f"tokens={n:,}  loops={R}")
    print(f"  best CONSTANT depth      k={k_const+1:>3}  CE={ce_const:.4f}")
    print(f"  per-token ORACLE         CE={ce_oracle:.4f}   headroom={ce_const-ce_oracle:.4f}")

    # split-half calibration: pick tau on half, score on the other -- never on the same tokens
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    cal, ev = perm[: n // 2], perm[n // 2:]
    cum = np.cumsum(dn2, axis=1)

    def score(tau, idx):
        # first index where cumulative distance reaches tau; if never, run to the end
        reached = cum[idx] >= tau
        k = np.where(reached.any(1), reached.argmax(1), R - 1)
        return float(ce2[idx, k].mean()), float((k + 1).mean())

    taus = np.quantile(cum[:, -1], np.linspace(0.02, 0.98, 49))
    best = min(((score(t, cal)[0], t) for t in taus))
    tau = best[1]
    ce_ev, depth_ev = score(tau, ev)
    ce_cal, depth_cal = score(tau, cal)

    print(f"\n  CUMULATIVE-dnorm rule, tau chosen on the calibration half, scored on the held-out half:")
    print(f"    tau={tau:.4f}   mean depth={depth_ev:.2f}   CE={ce_ev:.4f}")
    rec = (ce_const - ce_ev) / (ce_const - ce_oracle) * 100
    print(f"    vs best constant: {ce_ev-ce_const:+.4f}   oracle headroom recovered: {rec:+.1f}%")

    print(f"\n  for contrast, the INSTANTANEOUS rule §4.7 tested (halt when dnorm first drops below a threshold):")
    thr = np.quantile(dn2, np.linspace(0.02, 0.98, 49))
    def score_inst(t, idx):
        below = dn2[idx] <= t
        k = np.where(below.any(1), below.argmax(1), R - 1)
        return float(ce2[idx, k].mean()), float((k + 1).mean())
    bi = min(((score_inst(t, cal)[0], t) for t in thr))
    ce_i, d_i = score_inst(bi[1], ev)
    reci = (ce_const - ce_i) / (ce_const - ce_oracle) * 100
    print(f"    mean depth={d_i:.2f}  CE={ce_i:.4f}  vs constant {ce_i-ce_const:+.4f}  recovered {reci:+.1f}%")

    print(f"\n  per-token spread of the total angular distance (is the budget constant across tokens?):")
    tot = cum[:, -1]
    print(f"    mean={tot.mean():.4f}  sd={tot.std():.4f}  cv={tot.std()/tot.mean():.3f}  "
          f"p10={np.quantile(tot,0.1):.4f}  p90={np.quantile(tot,0.9):.4f}")
    # the budget at each token's OWN oracle depth -- the quantity that would have to be constant
    ko = ce2.argmin(1)
    bo = cum[np.arange(n), ko]
    print(f"    budget AT each token's oracle depth: mean={bo.mean():.4f} sd={bo.std():.4f} "
          f"cv={bo.std()/bo.mean():.3f}")
    print(f"    -> cv near 0 would mean one tau suffices; cv near 1 means the budget is per-token")


if __name__ == "__main__":
    main()
