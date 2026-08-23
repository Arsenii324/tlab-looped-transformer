"""Can a STATIC mixture over loop depths beat the best single depth? Zero training.

WHY. §4.7 measures per-token oracle depth headroom that is real and reliable (0.2008/0.2032 nats;
split-half +0.866 against a null of +0.0007) and UNREACHABLE by five label-free rule families. §4.7b
gives the structural reason: the rules condition on total path length, whose cross-token cv is 0.068,
while the ANGULAR BUDGET AT each token's own oracle depth has cv 0.798 -- they read a quantity that
barely varies in order to fire at one that varies 12x more. (Said precisely: 0.798 is
`cum[i, k_oracle(i)]` from cumulative_exit.py, NOT the cv of the oracle depth, which is larger.)

A rule has to COMMIT to one depth. A mixture does not: it can weight {h_t} and let the readout see a
combination. So the natural question after §4.7's negative is whether the information is recoverable
without selecting at all.

TEST THIS AS A READOUT FIRST, WITH NO TRAINING. If no STATIC weighting beats the best single depth,
a learned gate conditioned on the same states is very unlikely to, and the idea dies for free. If
some static mixture does beat it, the learned version becomes a §8 proposal WITH a measurement
behind it rather than an unrun idea.

PRE-REGISTERED READ (written before running, CLAUDE.md §1):
  * BASELINE is the best SINGLE depth on the same batch -- not loop 1, and not the oracle, which
    uses the label and is an upper bound rather than a competitor.
  * A mixture "wins" only if it beats that baseline by more than the same-config replicate floor
    measured in §4.15 (|dCE_best| 0.0202 / 0.0326 / 0.0527 at n=3, so use the LARGEST, 0.0527).
    Anything smaller is not resolvable and is reported as null.
  * FALSIFIER, and the expected outcome: no static mixture clears 0.0527. Then §4.7's negative
    extends from "no label-free RULE can reach the headroom" to "no static COMBINATION can either",
    which is materially stronger, and the learned-gate proposal is withdrawn before it is made.
  * This lands in §8 as a direction, or in §4.7 as a strengthened negative. NEVER in §3.5.

Mixtures swept (all convex, all label-free -- none may consult the target):
  single depths (the baseline family) | uniform over a contiguous window [a,b] |
  exponentially-tilted over all depths | best PAIR of depths at equal weight

Usage: python src/depth_mixture.py <ckpt_dir> [--loops 32] [--batches 8]
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig, LoopedTransformer  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
FLOOR = 0.0527   # §4.15, largest pairwise |dCE_best| across the n=3 same-config MPS replicates


@torch.no_grad()
def ce_of(model, states, w, y, normalize=False):
    """CE of the readout applied to sum_t w_t * h_t.

    `normalize`: divide each h_t by its own norm BEFORE mixing. THIS IS NOT COSMETIC. ||h|| grows
    18-26x across depth (sec4.3), so a raw uniform mixture over [1,16] is dominated by h_16 and is
    effectively the deepest single state wearing a mixture's clothes -- which would make a null
    result an artifact of the parameterisation rather than a fact about depth mixing. Both are
    reported; if they disagree, the disagreement is the finding."""
    h = torch.zeros_like(states[0])
    for wt, ht in zip(w, states):
        if wt:
            h = h + wt * (ht / ht.float().norm(dim=-1, keepdim=True).clamp_min(1e-8) if normalize else ht)
    lg = model.readout(h)
    return F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--loops", type=int, default=32)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq", type=int, default=256)
    args = ap.parse_args()

    cp = pathlib.Path(args.checkpoint)
    cp = cp / "last.pt" if cp.is_dir() else cp
    d = torch.load(cp, map_location="cpu", weights_only=False)
    m = LoopedTransformer(ModelConfig(**d["model_cfg"]))
    m.load_state_dict(d["model"]); m.eval()

    val = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    R = args.loops
    acc: dict[str, list[float]] = {}

    def note(tag, v):
        acc.setdefault(tag, []).append(v)

    NORM = True

    for _ in range(args.batches):
        ix = rng.integers(0, len(val) - args.seq - 1, size=args.batch_size)
        x = torch.from_numpy(np.stack([val[i:i + args.seq] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([val[i + 1:i + args.seq + 1] for i in ix]).astype(np.int64))
        with torch.no_grad():
            _, _, st = m(x, n_loops=R, return_all_loops=False, supervise_idx=set(),
                          return_states=True)
        for t in range(1, R + 1):                                   # singles
            w = [0.0] * R; w[t - 1] = 1.0
            note(f"single@{t}", ce_of(m, st, w, y))
        for a in (1, 2, 4, 6, 8, 12):                               # uniform windows
            for b in (8, 12, 16, 24, 32):
                if b <= a or b > R: continue
                w = [0.0] * R
                for t in range(a, b + 1): w[t - 1] = 1.0 / (b - a + 1)
                note(f"uniform[{a},{b}]", ce_of(m, st, w, y)); note(f"NORM_uniform[{a},{b}]", ce_of(m, st, w, y, True))
        for tau in (2.0, 4.0, 8.0, 16.0):                           # exponential tilt toward deep
            raw = np.exp(np.arange(1, R + 1) / tau); raw /= raw.sum()
            note(f"exp_tilt_tau{tau:g}", ce_of(m, st, raw.tolist(), y)); note(f"NORM_exp_tilt_tau{tau:g}", ce_of(m, st, raw.tolist(), y, True))
        for p, q in itertools.combinations((4, 8, 12, 16, 24, 32), 2):   # equal-weight pairs
            if q > R: continue
            w = [0.0] * R; w[p - 1] = w[q - 1] = 0.5
            note(f"pair({p},{q})", ce_of(m, st, w, y)); note(f"NORM_pair({p},{q})", ce_of(m, st, w, y, True))

    mean = {k: float(np.mean(v)) for k, v in acc.items()}
    singles = {k: v for k, v in mean.items() if k.startswith("single@")}
    base_k = min(singles, key=singles.get); base = singles[base_k]
    mixes = {k: v for k, v in mean.items() if not k.startswith("single@")}
    best_k = min(mixes, key=mixes.get); best = mixes[best_k]

    print(f"checkpoint {cp.parent.name}   {args.batches}x{args.batch_size}x{args.seq} tokens, R={R}")
    print(f"  BASELINE best single depth : {base_k:<18} CE {base:.4f}")
    print(f"  BEST static mixture        : {best_k:<18} CE {best:.4f}")
    print(f"  delta                      : {best - base:+.4f}   (floor {FLOOR}, §4.15 n=3)")
    print(f"  VERDICT: {'MIXTURE WINS' if base - best > FLOOR else 'NULL -- no static mixture clears the floor'}")
    print("\n  top 8 mixtures:")
    for k in sorted(mixes, key=mixes.get)[:8]:
        print(f"    {k:<20} {mixes[k]:.4f}  ({mixes[k]-base:+.4f} vs best single)")

    dst = ROOT / "checkpoints" / f"depth_mixture_{cp.parent.name}.json"
    dst.write_text(json.dumps(dict(baseline=base_k, baseline_ce=base, best_mix=best_k,
                                    best_mix_ce=best, delta=best - base, floor=FLOOR,
                                    all_means=mean), indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
