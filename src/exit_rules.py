"""Turn the per-token dump into REPORTABLE early-exit numbers (calibration/test, no label leakage).

The oracle `E_token[min_k CE]` uses the label and takes a min over 64 correlated noisy values, so it
is an optimistically biased UPPER BOUND, not a score. Everything here is fit on a calibration split
and scored on a disjoint test split, using only label-free per-token signals available at loop k:
entropy, top1-top2 margin, ||h_k - h_{k-1}||/||h_k||, and KL(p_k || p_{k-1}).

Rules, in increasing order of what they're allowed to use:
  fixed-k          the baseline: one global depth chosen on calibration
  threshold(sig)   halt the first time a signal crosses tau; tau swept on calibration
  bucket(sig)      bucket tokens by their LOOP-1 value of a signal (deciles fixed on calibration),
                   pick the best fixed depth per bucket on calibration, apply on test.
                   Zero parameters, and it answers "is depth demand PREDICTABLE from a label-free
                   feature available before you spend the compute?"
  oracle           upper bound, reported with its bias stated

The key comparison is threshold/bucket vs fixed-k ON TEST. If a label-free rule beats the best fixed
depth, that is "lower perplexity obtained by spending many loops where they are wanted", which is
what the task asks for. If none does, the honest finding is that per-token depth demand is real
(the argmin spread proves it) but NOT PREDICTABLE from these signals -- which is a sharper negative
than "the curve went up", and it says exactly what an exiter would need to fix.

Usage: python src/exit_rules.py <exitdump.npz> [--calib-frac 0.5]
"""

from __future__ import annotations

import argparse
import math

import numpy as np

BYTES_PER_TOKEN = 3.3358


def bpb(ce):
    return ce / (BYTES_PER_TOKEN * math.log(2))


def summarize(name, ce_sel, depth_sel, base):
    d = ce_sel.mean() - base
    print(f"{name:34} CE {ce_sel.mean():.4f}  bpb {bpb(ce_sel.mean()):.4f}  "
          f"vs best-fixed {d:+.4f}  mean depth {depth_sel.mean():6.2f}  max {int(depth_sel.max()):3d}")
    return ce_sel.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.dump)
    ce = z["ce"]                      # [n_seq, T, R]
    n_seq, T, R = ce.shape
    sigs = {k: z[k] for k in ("entropy", "margin", "dnorm", "kl") if k in z}
    print(f"dump: {n_seq} seqs x {T} tokens x {R} loops = {n_seq*T:,} scored tokens")

    # SPLIT BY SEQUENCE, not by token: tokens inside a sequence share context and are not
    # independent, so a token-level split would leak calibration information into test.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_seq)
    n_cal = int(n_seq * args.calib_frac)
    cal, tst = perm[:n_cal], perm[n_cal:]
    print(f"split by SEQUENCE: {len(cal)} calibration / {len(tst)} test\n")

    ce_c = ce[cal].reshape(-1, R)
    ce_t = ce[tst].reshape(-1, R)

    best_k = int(ce_c.mean(0).argmin()) + 1          # chosen on calibration only
    base = ce_t[:, best_k-1].mean()
    print(f"{'best fixed depth (chosen on calib)':34} k={best_k}  TEST CE {base:.4f}  bpb {bpb(base):.4f}")
    print(f"{'  test CE at loop 1':34}   {ce_t[:,0].mean():.4f}")
    orc = ce_t.min(1).mean()
    print(f"{'ORACLE (upper bound, uses label)':34} CE {orc:.4f}  bpb {bpb(orc):.4f}  "
          f"headroom {base-orc:.4f} nats\n")

    results = {}
    for sname, S in sigs.items():
        Sc = S[cal].reshape(-1, R); St = S[tst].reshape(-1, R)
        # ---- threshold rule: halt at first k where signal crosses tau (direction chosen by which
        # side actually helps on calibration; margin rises with confidence, entropy/dnorm/kl fall)
        best = None
        for direction in (+1, -1):
            v = Sc * direction
            lo, hi = np.percentile(v[np.isfinite(v)], [1, 99])
            for tau in np.linspace(lo, hi, 40):
                hit = v >= tau
                hit[:, -1] = True                     # always terminate
                k_idx = hit.argmax(1)
                m = ce_c[np.arange(len(ce_c)), k_idx].mean()
                if best is None or m < best[0]:
                    best = (m, tau, direction)
        _, tau, direction = best
        v = St * direction
        hit = v >= tau; hit[:, -1] = True
        k_idx = hit.argmax(1)
        results[f"threshold({sname})"] = summarize(
            f"threshold({sname})", ce_t[np.arange(len(ce_t)), k_idx], k_idx + 1, base)

        # ---- bucket rule: deciles of an EARLY signal value, best depth per bucket on calibration.
        # NOT loop 1 for every signal: `dnorm` and `kl` are differences between consecutive loops and
        # are identically 0 at loop 1 by construction, so bucketing on their loop-1 value put every
        # token in one bucket (9 of 10 deciles came back empty, printing `None`). Use loop 2 for
        # those two. This was a real bug in the first run and its output was degenerate, not
        # informative -- recorded rather than quietly fixed.
        b_idx = 1 if sname in ("dnorm", "kl") else 0
        c1, t1 = Sc[:, b_idx], St[:, b_idx]
        edges = np.percentile(c1[np.isfinite(c1)], np.linspace(0, 100, 11))[1:-1]
        bc, bt = np.digitize(c1, edges), np.digitize(t1, edges)
        depth_of = {b: int(ce_c[bc == b].mean(0).argmin()) + 1 for b in range(len(edges)+1)
                    if (bc == b).any()}
        k_sel = np.array([depth_of.get(b, best_k) for b in bt])
        results[f"bucket({sname})"] = summarize(
            f"bucket({sname}) by loop-{b_idx+1} value", ce_t[np.arange(len(ce_t)), k_sel-1], k_sel, base)
        print(f"    per-decile chosen depths: {[depth_of.get(b) for b in range(len(edges)+1)]}")

    win = min(results, key=results.get)
    gain = base - results[win]
    print(f"\nbest label-free rule on TEST: {win}  CE {results[win]:.4f}  "
          f"vs best fixed {base:.4f}  ({results[win]-base:+.4f} nats)")
    # A bare `<` is not a verdict: the first run reported "BEATS" on a margin of 0.0001 nats, which
    # is far inside this eval's noise and inside the 0.006-nat spread the clamp experiment treats as
    # indistinguishable. Require the gain to clear a stated threshold AND report the oracle fraction
    # actually captured, which is the quantity a reader cares about.
    TOL = 0.01   # nats; ~1/30 of the oracle headroom, and 6x the clamp's own indistinguishable spread
    frac = gain / (base - orc) if base > orc else float("nan")
    print(f"oracle headroom captured by the best rule: {frac:6.1%}")
    if gain > TOL:
        print(f"VERDICT: a label-free rule beats the best fixed depth by {gain:.4f} nats (> {TOL})")
    else:
        print(f"VERDICT: NO label-free rule beats the best fixed depth by more than {TOL} nats.")
        print("  Depth demand IS real -- the per-token argmin spread proves it -- but it is NOT")
        print("  predictable from these signals. That is the finding: an exiter would need a signal")
        print("  these four do not carry, not merely a better threshold.")

    # PERSIST. This printed its numbers and saved nothing -- reproducible but not traceable:
    # verifying a published figure meant re-running, which only works while the inputs survive.
    # (Traceability audit, 2026-08-23: 11 load-bearing scripts had this defect.)
    import json as _json, pathlib as _pl
    _dst = (_pl.Path(__file__).resolve().parents[1] / "checkpoints" /
            f"exit_rules_{_pl.Path(args.dump).stem}.json")
    _dst.write_text(_json.dumps(results, indent=2, default=str))
    print(f"\nwrote {_dst}")


if __name__ == "__main__":
    main()
