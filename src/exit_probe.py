"""A LEARNED exit rule, to make §4.7's negative honest.

§4.7 reports that no label-free rule beats the best fixed depth. But it only tested hand-crafted
rules -- thresholds and deciles on four single signals. That is a weak search, and a weak search
producing a negative is not the same as the negative being true. LTO (arXiv 2509.26314) reports that
a trained latent classifier reliably predicts answer correctness from latent states where simple
statistics do not, which is exactly the reason to suspect the hand-crafted family is the limitation.

So: fit a small multinomial logistic model on the EARLY-LOOP label-free features to predict the
per-token argmin depth bucket, on a calibration split, and score the resulting policy on a disjoint
test split. Features are only what a real exiter could see by loop k_obs: entropy, margin, ‖Δh‖/‖h‖
and successive KL at each loop up to k_obs, plus their deltas. No labels, no future loops.

Two honest guards:
  * split BY SEQUENCE, so tokens sharing a context cannot leak across the split;
  * the policy is scored by the CE it actually achieves, not by classification accuracy -- picking
    the right bucket is worthless if the CE difference between buckets is noise.

If this also fails to beat fixed-depth, §4.7's negative is much stronger: not "we could not find a
threshold" but "a learned probe on all four signals cannot either". If it succeeds, the exiter is
real and the report has a positive result.

Zero GPU: reads the npz produced by exit_dump.py.
Usage: python src/exit_probe.py <exitdump.npz> [--k-obs 4]
"""
from __future__ import annotations
import argparse, math
import numpy as np

BYTES_PER_TOKEN = 3.3358
bpb = lambda c: c / (BYTES_PER_TOKEN * math.log(2))


def softmax_fit(X, y, n_cls, epochs=220, lr=0.6, l2=1e-4, seed=0):
    """Plain multinomial logistic regression, written out to avoid a sklearn dependency."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, (X.shape[1], n_cls)); b = np.zeros(n_cls)
    Y = np.zeros((len(y), n_cls)); Y[np.arange(len(y)), y] = 1.0
    for ep in range(epochs):
        z = X @ W + b
        z -= z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        g = (p - Y) / len(X)
        W -= lr * (X.T @ g + l2 * W); b -= lr * g.sum(0)
    return W, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("--k-obs", type=int, default=4)
    ap.add_argument("--calib-frac", type=float, default=0.5); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    z = np.load(args.dump); ce = z["ce"]
    n_seq, T, R = ce.shape
    K = args.k_obs
    sig = [z[k] for k in ("entropy", "margin", "dnorm", "kl") if k in z]

    # features visible by loop K only
    feats = []
    for S in sig:
        feats.append(S[:, :, :K])
        feats.append(np.diff(S[:, :, :K], axis=2))          # per-loop change
    X = np.concatenate([f.reshape(n_seq, T, -1) for f in feats], axis=2).astype(np.float64)

    # depth buckets, chosen so each is a real alternative rather than a near-duplicate
    edges = [1, 2, 5, 9, 17, 33, R + 1]
    lbl = np.digitize(ce.argmin(2) + 1, edges) - 1
    reps = [1, 3, 6, 12, 24, 48]                            # representative depth per bucket

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_seq); ncal = int(n_seq * args.calib_frac)
    cal, tst = perm[:ncal], perm[ncal:]
    Xc, Xt = X[cal].reshape(-1, X.shape[2]), X[tst].reshape(-1, X.shape[2])
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-8
    Xc, Xt = (Xc - mu) / sd, (Xt - mu) / sd
    yc = lbl[cal].reshape(-1)
    cec, cet = ce[cal].reshape(-1, R), ce[tst].reshape(-1, R)

    best_k = int(cec.mean(0).argmin()) + 1
    base = cet[:, best_k - 1].mean()
    orc = cet.min(1).mean()
    print(f"{n_seq} seqs, split by sequence: {len(cal)} cal / {len(tst)} test; features from loops 1..{K}")
    print(f"  best fixed depth (calib) k={best_k}   TEST CE {base:.4f}  bpb {bpb(base):.4f}")
    print(f"  oracle (label-using bound)            TEST CE {orc:.4f}  bpb {bpb(orc):.4f}  "
          f"headroom {base-orc:.4f}")

    W, b = softmax_fit(Xc, yc, len(reps), seed=args.seed)
    pred = (Xt @ W + b).argmax(1)
    depth = np.array([reps[i] for i in pred])
    ce_policy = cet[np.arange(len(cet)), depth - 1].mean()
    acc = (pred == lbl[tst].reshape(-1)).mean()
    print(f"\n  LEARNED PROBE: TEST CE {ce_policy:.4f}  bpb {bpb(ce_policy):.4f}  "
          f"vs fixed {ce_policy-base:+.4f} nats")
    print(f"    bucket accuracy {acc:.3f}  mean depth {depth.mean():.2f}  "
          f"depth histogram {np.bincount(depth, minlength=49)[[1,3,6,12,24,48]].tolist()}")
    frac = (base - ce_policy) / (base - orc) if base > orc else float("nan")
    print(f"    oracle headroom captured: {frac:6.1%}")
    TOL = 0.01
    if base - ce_policy > TOL:
        print(f"  VERDICT: the learned probe BEATS fixed depth by {base-ce_policy:.4f} nats (> {TOL})")
    else:
        print(f"  VERDICT: even a learned probe on all four signals does NOT beat fixed depth "
              f"by more than {TOL} nats.")
        print("    This strengthens §4.7: the negative is not 'we failed to find a threshold', it is")
        print("    'the information required to choose a per-token depth is not present in these")
        print("    signals at all'. An exiter would need a different observable.")


if __name__ == "__main__":
    main()
