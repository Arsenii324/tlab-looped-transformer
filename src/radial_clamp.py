"""Does controlling the state's SCALE recover depth, or does scale growth protect the model?

Setup. §4.3 established that in the winning config the state escapes radially: ‖h‖ grows ~linearly
while the tangential step stays roughly constant, so the readout-visible angular step decays as 1/t.
Two readings of that are both live, and they make opposite predictions:

  (A) Dilution is the binding constraint. The loop still computes; the readout just stops seeing it
      because each increment is a smaller rotation. Then clamping ‖h‖ back down should RESTORE the
      angular step and recover depth-dependence.
  (B) Norm growth is accidental annealing. Past loop 8 this model's CE rises monotonically all the
      way to loop 105 while consecutive increments stay aligned at cos = 0.9999 -- the direction of
      travel is HARMFUL. Then the 1/t decay is the only thing keeping the damage small, and clamping
      should make things WORSE past the optimum, sharply.

This is a clean fork and it costs no training: rescale each token's state to a target RMS after every
loop, before both the readout and the next recurrence, then re-measure the per-loop CE curve.

Quantitative prediction, written before running (CLAUDE.md §1). The measured tangential step at loop
8 is 0.0249 * 6630 ~ 165. Clamped to s = s1 = 1655, the per-loop angular step becomes ~165/1655 ~
0.10 rad, so loops 8->64 would accumulate ~5.6 rad of rotation instead of the ~0.4 rad they
currently do. Under (B) that predicts a large CE INCREASE at high loop counts under the tightest
clamp, monotone in clamp tightness. Under (A) it predicts the opposite. Either way the answer is
legible in one table.

Clamp levels are read from the model's OWN measured trajectory (‖h1‖, ‖h8‖, ‖h16‖ on this
checkpoint) rather than round numbers, so "clamp to loop-1 scale" means exactly that.

Usage: python src/radial_clamp.py <ckpt_dir> [--max-loops 64]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig, LoopedTransformer  # noqa: E402
from eval import BYTES_PER_TOKEN, load_checkpoint  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

if torch.backends.mps.is_available():
    torch.mps.set_per_process_memory_fraction(12.0e9 / torch.mps.recommended_max_memory())


@torch.no_grad()
def clamped_curve(model, val, max_loops, seq_len, n_batches, batch_size, device,
                   target_rms: float | None, seed: int = 0, per_token_out=None):
    """Re-implements the loop with a radial clamp inserted. `target_rms=None` is the unclamped
    control and MUST reproduce eval.py's curve -- checked by the caller, since a control that does
    not match means the re-implementation is the thing being measured."""
    rng = np.random.default_rng(seed)
    ce_sum = {r: 0.0 for r in range(1, max_loops + 1)}
    n_tok = 0
    for _ in range(n_batches):
        ix = rng.integers(0, len(val) - seq_len - 1, size=batch_size)
        x = torch.from_numpy(np.stack([val[i:i + seq_len] for i in ix]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([val[i + 1:i + seq_len + 1] for i in ix]).astype(np.int64)).to(device)
        B, T = x.shape
        e = model.embed(x)
        cos, sin = model.rope(T, x.device, e.dtype)
        for layer in model.prelude:
            e = layer(e, cos, sin)
        h = model.h0.expand(B, T, -1) + e
        for t in range(max_loops):
            h_in = model._inject(h, e) if t > 0 else h
            h = model.block(h_in, cos, sin)
            if model.loop_norm is not None:
                h = model.loop_norm(h)
            if target_rms is not None:
                # per-token radial rescale to a fixed RMS; direction untouched, applied before BOTH
                # the readout and the next recurrence (so it is a change to the dynamics, not a
                # cosmetic change to the decode path).
                cur = h.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
                h = (h.float() * (target_rms / cur)).to(h.dtype)
            logits = model.readout(h, cos, sin)
            per_tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                       reduction="none")
            ce_sum[t + 1] += per_tok.sum().item()
            if per_token_out is not None:
                per_token_out[t].append(per_tok.view(B, T).cpu().float().numpy())
        n_tok += x.numel()
    return {r: v / n_tok for r, v in ce_sum.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str)
    ap.add_argument("--max-loops", type=int, default=64)
    ap.add_argument("--n-batches", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--val", type=str, default=str(ROOT / "data" / "val.bin"))
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt_path = pathlib.Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "last.pt"
    model, cfg, ckpt = load_checkpoint(ckpt_path, device)
    seq_len = ckpt["train_cfg"]["seq_len"]
    val = np.memmap(args.val, dtype=np.uint16, mode="r")

    # clamp levels from this checkpoint's own measured trajectory, converted from L2 norm to RMS
    dyn_path = ckpt_path.parent / f"dynamics_{ckpt_path.parent.name}.json"
    H = cfg.hidden_size
    if dyn_path.exists():
        sn = json.load(dyn_path.open())["state_norm"]
        levels = {f"h{k}": sn[k - 1] / math.sqrt(H) for k in (1, 8, 16)}
    else:
        print("no dynamics json; falling back to measured-on-the-fly norms", flush=True)
        levels = {}
    print(f"checkpoint {ckpt_path}  tokens={ckpt.get('tokens')}  H={H}")
    print("clamp levels (RMS): " + ", ".join(f"{k}={v:.2f}" for k, v in levels.items()), flush=True)

    out = {}
    base = clamped_curve(model, val, args.max_loops, seq_len, args.n_batches, args.batch_size,
                          device, None)
    out["unclamped"] = base
    pub = ckpt_path.parent / f"eval_{ckpt_path.parent.name}.json"
    if pub.exists():
        p = json.load(pub.open())["val_ce"]
        d = max(abs(base[r] - p[str(r)]) for r in range(1, min(args.max_loops, 64) + 1))
        print(f"CONTROL vs published eval.py curve: max|diff| = {d:.3e} "
              f"{'OK' if d < 1e-4 else '<-- RE-IMPLEMENTATION DIFFERS, results below are suspect'}",
              flush=True)

    for name, rms in levels.items():
        out[f"clamp_{name}"] = clamped_curve(model, val, args.max_loops, seq_len, args.n_batches,
                                              args.batch_size, device, rms)
        print(f"  done clamp_{name}", flush=True)

    cols = list(out)
    print("\n" + f"{'loop':>5}" + "".join(f"{c:>14}" for c in cols))
    for r in [1, 2, 4, 8, 11, 16, 24, 32, 48, 64]:
        if r > args.max_loops:
            continue
        print(f"{r:>5}" + "".join(f"{out[c][r]:>14.4f}" for c in cols))
    print(f"\n{'variant':>14} {'best_loop':>10} {'best_CE':>9} {'CE@1':>8} {'loop_gain':>10} {'bpb':>7}")
    summary = {}
    for c in cols:
        b = min(out[c], key=out[c].get)
        summary[c] = dict(best_loop=b, best_ce=out[c][b], ce_at_1=out[c][1],
                           loop_gain=out[c][1] - out[c][b],
                           bpb=out[c][b] / (BYTES_PER_TOKEN * math.log(2)))
        s = summary[c]
        print(f"{c:>14} {b:>10} {s['best_ce']:>9.4f} {s['ce_at_1']:>8.4f} "
              f"{s['loop_gain']:>10.4f} {s['bpb']:>7.4f}")

    op = ckpt_path.parent / f"clamp_{ckpt_path.parent.name}.json"
    op.write_text(json.dumps(dict(checkpoint=str(ckpt_path), tokens=ckpt.get("tokens"),
                                   levels_rms=levels, curves={k: {str(r): v for r, v in c.items()}
                                                               for k, c in out.items()},
                                   summary=summary), indent=2))
    print(f"\nwrote {op}")


if __name__ == "__main__":
    main()
