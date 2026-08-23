"""Per-token, per-loop dump: everything an early-exit rule could possibly key on.

The saturation result in §4.2/§4.6 is about an AVERAGE: min_k E_token[CE(token,k)] = 4.0071 at k=8.
But the per-token argmin depth is a DISTRIBUTION. If tokens differ in how much depth they want, then

    E_token[ min_k CE(token,k) ]   <   min_k E_token[ CE(token,k) ]

strictly, unless every token has the same argmin. So "the fixed-depth curve saturates at 8" does NOT
by itself imply "loops stop being useful past 8" -- it implies a single GLOBAL depth cannot extract
what is there. The gap between those two quantities is the entire headroom available to early exit,
and measuring it costs one forward pass over a checkpoint that already exists.

This script only DUMPS. It computes, for every scored token and every loop k:
    ce        cross-entropy of the true next token           (label-dependent -> oracle only)
    entropy   predictive entropy of the loop-k distribution  (label-free)
    margin    top1 - top2 logit gap                          (label-free)
    dnorm     ||h_k - h_{k-1}|| / ||h_k||                    (label-free; LoopMDM's halting signal)
    kl        KL(p_k || p_{k-1})                             (label-free; successive-output change)
The label-free four are what a deployable rule may use; `ce` is kept ONLY to compute the oracle
upper bound and to fit rules on a calibration split. Keeping them in one file makes the
calibration/test discipline checkable rather than promised.

Memory note, learned the hard way (report §6): materializing [B,T,V] logits for all 64 loops at once
is 6.4GB and OOMs. Logits are computed one loop at a time and discarded immediately; only the small
per-token scalars survive.

Usage: python src/exit_dump.py <ckpt_dir_or_pt> [--max-loops 64] [--out out.npz]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--max-loops", type=int, default=64)
    ap.add_argument("--frozen", type=str, default=str(ROOT / "data" / "frozen_eval_set.npz"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    from model import Config as ModelConfig, LoopedTransformer

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    if dev == "mps":
        torch.mps.set_per_process_memory_fraction(10.0e9 / torch.mps.recommended_max_memory())
    p = pathlib.Path(args.checkpoint)
    if p.is_dir():
        p = p / "last.pt"
    ck = torch.load(p, map_location=dev, weights_only=False)
    cfg = ModelConfig(**ck["model_cfg"])
    model = LoopedTransformer(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    d = np.load(args.frozen)
    X, Y = d["x"], d["y"]
    n, T = X.shape
    R = args.max_loops
    print(f"device={dev} ckpt={p} tokens={ck.get('tokens')} seqs={n} seq_len={T} loops={R}", flush=True)

    keys = ("ce", "entropy", "margin", "dnorm", "kl")
    acc = {k: np.zeros((n, T, R), dtype=np.float32) for k in keys}

    with torch.no_grad():
        for b0 in range(0, n, args.batch_size):
            xb = torch.from_numpy(X[b0:b0 + args.batch_size]).to(dev)
            yb = torch.from_numpy(Y[b0:b0 + args.batch_size]).to(dev)
            B = xb.shape[0]
            _, _, states = model(xb, n_loops=R, return_all_loops=False,
                                 supervise_idx=set(), return_states=True)
            cos, sin = model.rope(T, xb.device, states[0].dtype)
            prev_logp = None
            for r in range(R):
                h = states[r].float()
                lg = model.readout(states[r], cos, sin).float()
                logp = F.log_softmax(lg, dim=-1)
                acc["ce"][b0:b0+B, :, r] = (-logp.gather(-1, yb.unsqueeze(-1)).squeeze(-1)).cpu().numpy()
                acc["entropy"][b0:b0+B, :, r] = (-(logp.exp() * logp).sum(-1)).cpu().numpy()
                top2 = lg.topk(2, dim=-1).values
                acc["margin"][b0:b0+B, :, r] = (top2[..., 0] - top2[..., 1]).cpu().numpy()
                if r > 0:
                    dh = h - states[r-1].float()
                    acc["dnorm"][b0:b0+B, :, r] = (dh.norm(dim=-1) /
                                                    h.norm(dim=-1).clamp_min(1e-8)).cpu().numpy()
                    acc["kl"][b0:b0+B, :, r] = ((logp.exp() * (logp - prev_logp)).sum(-1)).cpu().numpy()
                prev_logp = logp
                del lg, logp
            if (b0 // args.batch_size) % 20 == 0:
                print(f"  {b0+B}/{n}", flush=True)

    out = args.out or str(p.parent / f"exitdump_{p.parent.name}.npz")
    np.savez_compressed(out, tokens=ck.get("tokens"), step=ck.get("step"), max_loops=R, **acc)

    ce = acc["ce"].reshape(-1, R)
    fixed = ce.mean(0)
    best_fixed = int(fixed.argmin()) + 1
    oracle = ce.min(1).mean()
    argmin = ce.argmin(1) + 1
    print(f"\nwrote {out}")
    print(f"best FIXED depth: {best_fixed}  CE {fixed[best_fixed-1]:.4f}   (CE@1 {fixed[0]:.4f})")
    print(f"ORACLE per-token min CE: {oracle:.4f}   headroom {fixed[best_fixed-1]-oracle:.4f} nats")
    print("  (oracle uses the label and takes a min over 64 correlated noisy values -- an "
          "optimistically biased UPPER BOUND on what any rule can reach, not a score)")
    q = np.percentile(argmin, [10, 25, 50, 75, 90])
    print(f"per-token argmin depth: median {int(np.median(argmin))}  "
          f"deciles {q.tolist()}  frac at depth 1: {(argmin==1).mean():.3f}  "
          f"frac >8: {(argmin>8).mean():.3f}  frac >32: {(argmin>32).mean():.3f}")


if __name__ == "__main__":
    main()
