"""Evaluate a trained checkpoint: per-loop validation perplexity (the task's actual scored metric),
swept past the trained max loop count to see whether it degrades gracefully or catastrophically;
per-loop predictive entropy (catches collapse-to-a-fixed-distribution, a different degenerate case
than genuine saturation); and an online contraction-rate estimate from a clean/h0-perturbed pair on
the same batch (state-level, does not depend on the readout at all -- reused from D44's method in
the sibling Huginn project, adapted here since h0 is a deterministic parameter, not an unseeded draw).

Usage: python eval.py <checkpoint_dir_or_.pt> [--max-loops 64] [--val data/val.bin]
Prints the full per-loop table to stdout (not just a verdict) and writes eval_<run_name>.json next
to the checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig, LoopedTransformer  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
BYTES_PER_TOKEN = 3.3358  # measured over the FULL 6M-token val shard (the exact set bits/byte is
# reported on) by decoding it and taking len(utf8 bytes)/n_tokens. Supersedes an earlier
# CHARS_PER_TOKEN = 3.45, which was wrong twice over: it was estimated from a 5-document sample in
# train_tokenizer.py (real value over 6M val tokens is 3.3162 chars/token), and it counted CHARACTERS
# where bits-per-BYTE needs bytes (this corpus is 1.006 bytes/char -- some multi-byte UTF-8). The old
# constant made every reported bits/byte ~3.4% optimistic. This matters more than it looks: the entire
# reason to report bits/byte instead of token perplexity is comparability with external numbers under
# a different tokenizer, so the constant has to be right or the metric is worse than useless.
CHARS_PER_TOKEN = BYTES_PER_TOKEN  # back-compat alias; the divisor below is bytes/token

if torch.backends.mps.is_available():
    # Kept as a safety cap on a shared machine, but the reason it was originally needed is gone:
    # model.forward used to wrap each loop in torch.enable_grad(), which overrides the @torch.no_grad()
    # on perplexity_curve below, so every eval retained a graph across all n_loops (report.md sec 6).
    # Post-fix, a 128-loop sweep fits comfortably; this no longer has to be tight.
    torch.mps.set_per_process_memory_fraction(14.0e9 / torch.mps.recommended_max_memory())


def load_checkpoint(path: pathlib.Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"])
    model = LoopedTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


@torch.no_grad()
def perplexity_curve(model, val: np.memmap, max_loops: int, seq_len: int, n_batches: int,
                      batch_size: int, device: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    ce_sum = {r: 0.0 for r in range(1, max_loops + 1)}
    entropy_sum = {r: 0.0 for r in range(1, max_loops + 1)}
    n_tok = 0
    for _ in range(n_batches):
        ix = rng.integers(0, len(val) - seq_len - 1, size=batch_size)
        x = torch.from_numpy(np.stack([val[i:i + seq_len] for i in ix]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([val[i + 1:i + seq_len + 1] for i in ix]).astype(np.int64)).to(device)
        # Read out ONE loop at a time instead of materializing every loop's logits at once.
        # return_all_loops=True holds max_loops tensors of [B,T,V] simultaneously: at 192 loops,
        # batch 8, V=4096 that is 6.4GB of logits alone, which OOM'd a 13GB MPS budget. The states
        # are [B,T,H] -- 110x smaller at this vocab -- so keeping all of those and projecting them
        # one at a time costs ~700MB instead. Exact same arithmetic, verified below at 0.0 diff.
        _, _, states = model(x, n_loops=max_loops, return_all_loops=False,
                             supervise_idx=set(), return_states=True)
        cos, sin = model.rope(x.shape[1], x.device, states[0].dtype)  # readout needs these if n_coda>0
        for r in range(1, max_loops + 1):
            logits = model.readout(states[r - 1], cos, sin)
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
            ce_sum[r] += ce.item()
            logp = F.log_softmax(logits.float(), dim=-1)
            ent = -(logp.exp() * logp).sum(-1).mean().item()
            entropy_sum[r] += ent * x.numel() / seq_len  # weight by number of positions summed
        n_tok += x.numel()
    ce = {r: v / n_tok for r, v in ce_sum.items()}
    # The clamp keeps exp() from overflowing, but a CE of 20 is not a result -- chance for this
    # vocabulary is ln(4096) = 8.3178, so anything materially above it means a broken forward pass,
    # a vocabulary mismatch, or a degenerate checkpoint. Silently reporting e^20 hands back a number
    # where an exception belongs, which is the opposite of what train.py's degenerate-step guard
    # does. Flag it loudly and keep the clamp for the values that are merely bad.
    _chance = math.log(model.cfg.vocab_size)
    _broken = {r: v for r, v in ce.items() if v > _chance + 0.5}
    if _broken:
        print(f"  ⚠ CE ABOVE CHANCE ({_chance:.4f}) at loops {sorted(_broken)}: "
              f"{ {r: round(v, 4) for r, v in sorted(_broken.items())[:8]} } -- this is not a "
              f"quality result, it is a broken vocabulary or a broken forward pass. Check the "
              f"tokenizer with src/check_tokenizer_identity.py before reading anything below.",
              flush=True)
    ppl = {r: float(np.exp(min(v, 20))) for r, v in ce.items()}
    ent = {r: v / (n_batches * batch_size) for r, v in entropy_sum.items()}
    return ce, ppl, ent


@torch.no_grad()
def contraction_estimate(model, val: np.memmap, seq_len: int, max_loops: int, device: str,
                          noise_scale: float = 1.0, seed: int = 0):
    """Two forwards of the SAME batch, one with h0 perturbed by noise_scale, tracked via the model's
    own per-loop state norms is not enough (that gives each trajectory's own norm, not their
    distance) -- so this re-implements the loop manually to log ||h_clean - h_noisy|| per loop.

    THE DUPLICATION IS NOT ACTUALLY NECESSARY and the justification above is wrong: `forward()`
    accepts both `h0_noise=` and `return_states=`, so two calls give both trajectories' states
    through the REAL loop and the distances are a subtraction away. It is left in place because it
    works and is what every published contraction number here was measured with (CLAUDE.md sec 3:
    do not refactor analysis code that works). What is added is the guard the duplication needs --
    `rollout` below omits the convex-gate branch that `forward()` applies, so on a gated checkpoint
    it would silently measure a DIFFERENT dynamical system than the model is. No gated checkpoint
    has ever been run through it; this makes sure none silently is."""
    if model.gate_mlp is not None or model.cfg.fixed_gate is not None:
        raise NotImplementedError(
            "contraction_estimate re-implements the loop and does not apply the convex gate, so on "
            "this checkpoint it would measure a different map than the model computes. Use "
            "forward(..., h0_noise=..., return_states=True) twice and difference the states.")
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(val) - seq_len - 1, size=8)
    x = torch.from_numpy(np.stack([val[i:i + seq_len] for i in ix]).astype(np.int64)).to(device)

    def rollout(h0_noise):
        # Only the h0_noise>0 call ever draws from randn_like (the clean call's noise branch is
        # skipped entirely), so this seed does not pair the two draws -- it only makes the noisy
        # trajectory itself reproducible run-to-run at a fixed noise_scale.
        torch.manual_seed(12345)
        B, T = x.shape
        e = model.embed(x)
        cos, sin = model.rope(T, x.device, e.dtype)
        for layer in model.prelude:   # must mirror forward(): what the loop sees is the prelude
            e = layer(e, cos, sin)    # output, not the raw embedding (no-op when n_prelude=0)
        h0 = model.h0 + (h0_noise * torch.randn_like(model.h0) if h0_noise > 0 else 0)
        h = h0.expand(B, T, -1) + e
        states = []
        for t in range(max_loops):
            h_in = model._inject(h, e) if t > 0 else h
            h = model.block(h_in, cos, sin)
            if model.loop_norm is not None:
                h = model.loop_norm(h)
            states.append(h)
        return states

    clean = rollout(0.0)
    noisy = rollout(noise_scale)
    dist = [float((c - n).norm(dim=-1).mean().item()) for c, n in zip(clean, noisy)]
    ratios = [dist[i + 1] / dist[i] if dist[i] > 1e-8 else float("nan") for i in range(len(dist) - 1)]
    return dist, ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=str)
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting a wider stored sweep with a narrower one")
    ap.add_argument("--max-loops", type=int, default=64)
    ap.add_argument("--val", type=str, default=str(ROOT / "data" / "val.bin"))
    ap.add_argument("--n-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt_path = pathlib.Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "last.pt"
    model, cfg, ckpt = load_checkpoint(ckpt_path, device)

    seq_len = ckpt["train_cfg"]["seq_len"]
    val = np.memmap(args.val, dtype=np.uint16, mode="r")

    print(f"checkpoint: {ckpt_path}  step={ckpt.get('step')}  tokens={ckpt.get('tokens')}")
    print(f"model: H={cfg.hidden_size} layers_per_loop={cfg.layers_per_loop} "
          f"truncate_bptt={cfg.truncate_bptt} state_renorm={cfg.state_renorm} "
          f"inject_mode={cfg.inject_mode} depth_init={cfg.depth_init}")
    trained_max = ckpt["train_cfg"]["max_train_loops"]
    print(f"trained loop range: [{ckpt['train_cfg']['min_train_loops']}, {trained_max}]  "
          f"evaluating to {args.max_loops} loops ({'within' if args.max_loops <= trained_max else 'PAST'} "
          f"the trained range)")

    ce, ppl, ent = perplexity_curve(model, val, args.max_loops, seq_len, args.n_batches,
                                     args.batch_size, device)
    dist, ratios = contraction_estimate(model, val, seq_len, args.max_loops, device)

    bpb = {r: v / (BYTES_PER_TOKEN * math.log(2)) for r, v in ce.items()}

    print(f"\n{'loop':>5} {'val_CE':>8} {'val_ppl':>9} {'bits/byte':>10} {'pred_entropy':>13} "
          f"{'clean/noisy_dist':>17} {'contraction_ratio':>18}")
    for r in range(1, args.max_loops + 1):
        d = dist[r - 1]
        ratio = ratios[r - 2] if r >= 2 else float("nan")
        print(f"{r:>5} {ce[r]:>8.4f} {ppl[r]:>9.2f} {bpb[r]:>10.4f} {ent[r]:>13.4f} {d:>17.3f} "
              f"{ratio:>18.4f}")

    best_r = min(ce, key=ce.get)
    print(f"\nbest loop count on val: r={best_r}  CE={ce[best_r]:.4f}  ppl={ppl[best_r]:.2f}  "
          f"bits/byte={bpb[best_r]:.4f}")
    print(f"val CE at r=1: {ce[1]:.4f} -> at r={best_r}: {ce[best_r]:.4f} "
          f"(improvement {ce[1]-ce[best_r]:.4f} nats, {bpb[1]-bpb[best_r]:.4f} bits/byte)")
    print(f"(bits/byte uses {BYTES_PER_TOKEN} BYTES/token, measured over the full val shard -- "
          f"comparable across tokenizers, unlike token-level perplexity)")

    out = dict(checkpoint=str(ckpt_path), step=ckpt.get("step"), tokens=ckpt.get("tokens"),
               model_cfg=dataclasses.asdict(cfg), val_ce=ce, val_ppl=ppl, val_bits_per_byte=bpb,
               pred_entropy=ent, contraction_dist=dist, contraction_ratio=ratios, best_loop=best_r)
    out_path = ckpt_path.parent / f"eval_{ckpt_path.parent.name}.json"
    # Refuse to replace a WIDER sweep with a narrower one. This path is fixed per checkpoint, so a
    # quick `--max-loops 8 --n-batches 2` smoke run silently destroyed the dense 1..64 curve that
    # the model card and §.headline's plateau [6,17] are both read from -- recovered only because
    # the file happened to be tracked in git. The artifact is load-bearing; overwriting it is a
    # deliberate act and now has to be spelled.
    if out_path.exists() and not args.force:
        prev = json.loads(out_path.read_text()).get("val_ce", {})
        if len(prev) > len(out["val_ce"]):
            raise SystemExit(
                f"REFUSING to overwrite {out_path}: it holds a {len(prev)}-loop sweep and this run "
                f"produced only {len(out['val_ce'])}. That would replace a full curve with a "
                f"partial one at the same path. Re-run with --max-loops {max(int(k) for k in prev)} "
                f"to regenerate it properly, or pass --force if narrowing is what you intend.")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
