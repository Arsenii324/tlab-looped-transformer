"""What KIND of token wants depth? Zero GPU -- pure analysis of the existing exit dump.

§4.7 found large per-token depth demand (argmin deciles [1,2,7,43,64]) that NO label-free signal
predicts. That was a search over signals available AT INFERENCE. This asks the complementary
question from the data side: is depth demand a property of the TOKEN ITSELF -- its identity, its
frequency, its position in the sequence, how surprising it is?

That matters for two reasons. If argmin depth is predictable from token identity alone, an exiter
could be a lookup on the *input* token rather than a probe on the state -- trivially cheap, and it
would say the demand is lexical rather than contextual. If it is predictable from position, the
demand is about context accumulation. If it is predictable from NEITHER, then §4.7's negative is
much stronger: depth demand is genuinely contextual and per-instance, and no cheap proxy exists.

Costs nothing: reads the npz, no model, no GPU.
"""
from __future__ import annotations
import argparse, collections, json, pathlib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--frozen", default="data/frozen_eval_set.npz")
    args = ap.parse_args()
    z = np.load(args.dump); ce = z["ce"]
    n, T, R = ce.shape
    d = np.load(args.frozen); Y = d["y"][:n]
    argmin = ce.argmin(2) + 1                     # [n, T] depth each token actually wanted
    flat_a, flat_y = argmin.reshape(-1), Y.reshape(-1)
    print(f"{n} seqs x {T} tokens; argmin depth mean {flat_a.mean():.2f} median {np.median(flat_a):.0f}")

    # 1) by POSITION in the sequence -- does more context change depth demand?
    print("\nby position decile (does context accumulation change demand?)")
    pos = np.tile(np.arange(T), n)
    for lo in range(0, T, T // 8):
        m = (pos >= lo) & (pos < lo + T // 8)
        print(f"  pos {lo:>4}-{lo+T//8-1:<4} mean argmin {flat_a[m].mean():6.2f}  "
              f"frac>8 {(flat_a[m] > 8).mean():.3f}")

    # 2) by TOKEN FREQUENCY -- is demand lexical? (frequency computed on this very sample)
    cnt = collections.Counter(flat_y.tolist())
    rank = {t: i for i, (t, _) in enumerate(cnt.most_common())}
    r = np.array([rank[t] for t in flat_y])
    print("\nby target-token frequency rank (0 = most common)")
    for lo, hi, lbl in ((0, 10, "top10"), (10, 100, "10-100"), (100, 500, "100-500"),
                        (500, 10**9, "rare")):
        m = (r >= lo) & (r < hi)
        if m.sum():
            print(f"  {lbl:>9}: n={m.sum():>7}  mean argmin {flat_a[m].mean():6.2f}  "
                  f"frac>8 {(flat_a[m] > 8).mean():.3f}  frac==1 {(flat_a[m] == 1).mean():.3f}")

    # 3) how much of the variance is EXPLAINABLE by token identity at all?
    # between-token variance of mean argmin vs total variance = an upper bound on what ANY
    # identity-only rule could capture.
    tot = flat_a.var()
    means = {}
    for t in np.unique(flat_y):
        m = flat_y == t
        if m.sum() >= 20:
            means[t] = flat_a[m].mean()
    if means:
        sizes = np.array([(flat_y == t).sum() for t in means])
        mu = np.array([means[t] for t in means])
        between = np.average((mu - flat_a.mean()) ** 2, weights=sizes)
        print(f"\nvariance of argmin depth: total {tot:.1f}")
        print(f"  between-token (identity-explainable, tokens with n>=20): {between:.1f} "
              f"= {between/tot*100:.1f}% of total")
        print("  -> an UPPER BOUND on what any rule keyed on token identity alone could capture.")

    # 4) does the token that wants depth actually GAIN from it?
    gain = ce[:, :, 0].reshape(-1) - ce.min(2).reshape(-1)
    print(f"\nper-token gain (CE@1 - CE@argmin): mean {gain.mean():.4f} nats")
    for lo, hi in ((1, 2), (2, 9), (9, 33), (33, 65)):
        m = (flat_a >= lo) & (flat_a < hi)
        if m.sum():
            print(f"  argmin in [{lo},{hi}): n={m.sum():>7} ({m.mean():5.1%})  "
                  f"mean gain {gain[m].mean():.4f}")
    print("\n(If deep-wanting tokens have LARGE gains, the headroom is concentrated and worth "
          "chasing; if their gains are tiny, the oracle headroom is spread thin and an exiter "
          "cannot realistically capture it.)")


if __name__ == "__main__":
    main()
