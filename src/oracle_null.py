"""How much of the oracle headroom is REAL depth heterogeneity, and how much is selection on noise?

§4.7 reports oracle headroom = E_i[min_k CE(i,k)] below the best fixed depth. That statistic takes a
minimum over ~64 correlated, noisy values per token USING THE LABEL, so some of it is genuine
per-token depth preference and some is simply the minimum of noise. Calling it "optimistically
biased" is not enough -- the bias is computable from the same array.

Three calibrations, cheapest first:

  COARSE-GRID CHECK (no null needed). Compute the oracle over 7 candidate depths
  {1,2,4,8,16,32,64} instead of all 64. Selection bias grows with the number of candidates, real
  heterogeneity does not. If the 7-candidate oracle captures most of the headroom, bias is small; if
  the 64-candidate oracle is much lower, most of the excess is noise.

  NULL A (preferred): circular shift. Decompose CE(i,k) = m(k) + R(i,k) with m the population depth
  curve. Roll each token's residual by a random amount: this PRESERVES the shape of each token's
  curve (its roughness, its variance) and destroys only WHERE its minimum sits. So it tests exactly
  "tokens have depth preferences, but random ones."

  NULL B (loose): permute residuals across k. Also destroys smoothness, so it over-credits spurious
  minima and yields a CONSERVATIVE (larger) null headroom -- i.e. a lower bound on the real effect.

The same nulls are applied to the argmin DISTRIBUTION, because the dispersion is the actual finding:
is frac(argmin > 32) = 0.279 above or below what random depth preference produces?

Reported as: real_headroom = null_oracle - real_oracle. Zero GPU.
"""
from __future__ import annotations
import argparse
import numpy as np


def oracle_over(ce, cols):
    return ce[:, cols].min(1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    z = np.load(args.dump); ce = z["ce"].reshape(-1, z["ce"].shape[-1]).astype(np.float64)
    N, K = ce.shape
    rng = np.random.default_rng(args.seed)

    m = ce.mean(0)
    best_k = int(m.argmin())
    base = m[best_k]
    real_oracle = ce.min(1).mean()
    print(f"{N:,} tokens x {K} loops")
    print(f"  best fixed depth k={best_k+1}  CE {base:.4f}")
    print(f"  real oracle (all {K} candidates)  {real_oracle:.4f}   raw headroom {base-real_oracle:.4f}")

    coarse = [c for c in (1, 2, 4, 8, 16, 32, 64) if c <= K]
    ci = [c - 1 for c in coarse]
    o7 = oracle_over(ce, ci)
    print(f"\n  COARSE-GRID CHECK: oracle over {len(coarse)} candidates {coarse} = {o7:.4f}"
          f"  headroom {base-o7:.4f}")
    print(f"    fraction of the 64-candidate headroom captured by 7 candidates: "
          f"{(base-o7)/(base-real_oracle):.1%}")
    print("    (high => heterogeneity is real and coarse; low => most of the excess is selection)")

    R = ce - m
    def null(kind):
        vals, fr32, fr8 = [], [], []
        for _ in range(args.reps):
            if kind == "roll":
                sh = rng.integers(0, K, size=N)
                idx = (np.arange(K)[None, :] - sh[:, None]) % K
                Rp = np.take_along_axis(R, idx, axis=1)
            else:
                Rp = np.apply_along_axis(rng.permutation, 1, R)
            cep = m + Rp
            vals.append(cep.min(1).mean())
            am = cep.argmin(1) + 1
            fr32.append((am > 32).mean()); fr8.append((am > 8).mean())
        return np.mean(vals), np.mean(fr8), np.mean(fr32)

    am = ce.argmin(1) + 1
    print(f"\n  REAL argmin: median {int(np.median(am))}  frac>8 {(am>8).mean():.3f}  "
          f"frac>32 {(am>32).mean():.3f}  frac==1 {(am==1).mean():.3f}")
    for kind, label in (("roll", "NULL A circular shift (preferred)"),
                        ("perm", "NULL B full permutation (conservative)")):
        no, f8, f32 = null(kind)
        print(f"\n  {label}")
        print(f"    null oracle {no:.4f}   null headroom {base-no:.4f}")
        print(f"    ** REAL headroom = null_oracle - real_oracle = {no-real_oracle:+.4f} nats **")
        print(f"    null frac>8 {f8:.3f} (real {(am>8).mean():.3f})   "
              f"null frac>32 {f32:.3f} (real {(am>32).mean():.3f})")
    print("\n  Read: if REAL headroom under NULL A stays large, per-token depth demand is genuine")
    print("  heterogeneity. If it collapses toward 0, the apparent dispersion is mostly the minimum")
    print("  of noise -- itself a reportable finding, and better learned here than from a grader.")



def _persist_stdout(name, text):
    # PERSIST (traceability audit 2026-08-23): this printed its numbers and saved nothing, so
    # every claim it supports was reproducible but not traceable -- verifying one meant
    # re-running it, which only works while its inputs survive.
    import pathlib as _pl
    _dst = _pl.Path(__file__).resolve().parents[1] / "checkpoints" / f"{name}_report.txt"
    _dst.write_text(text)
    print(f"wrote {_dst}")

if __name__ == "__main__":
    import io as _io, contextlib as _cl
    _buf = _io.StringIO()
    with _cl.redirect_stdout(_buf):
        main()
    _out = _buf.getvalue()
    print(_out, end="")
    _persist_stdout("oracle_null", _out)

