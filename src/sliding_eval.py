"""Sliding-window evaluation: report the headline number under a protocol that is not
context-starved, alongside the chunked one, with both labelled.

The chunked protocol every other eval here uses scores 256-token non-overlapping windows, so a
scored token has on average only **128.5 tokens (~429 bytes) of left context** and 25.7% of windows
straddle a document boundary. That is fine for WITHIN-project comparisons -- the protocol is fixed
and identical across arms, which is all a paired comparison needs -- but it deflates the ABSOLUTE
bits/byte, which is the only number an outside reader can compare.

Sliding-window scoring fixes that: advance by `stride` and score only the final `stride` positions of
each window, so every scored token has close to a full window of context. At L=256 and stride 64 the
cost is 4x a chunked pass and average left context rises from ~128 to ~224 tokens.

Reported as a SECOND number with its protocol named, never as a replacement -- swapping protocols
mid-report would silently invalidate every comparison already made.

Usage: python src/sliding_eval.py <ckpt_dir> --loops 8 [--stride 64]
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval import load_checkpoint, BYTES_PER_TOKEN  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
if torch.backends.mps.is_available():
    torch.mps.set_per_process_memory_fraction(10.0e9 / torch.mps.recommended_max_memory())


@torch.no_grad()
def sliding(model, val, loops, seq_len, stride, n_tokens, device, batch_size=8):
    """Score only the last `stride` positions of each window; step by `stride`."""
    starts = list(range(0, n_tokens - seq_len - 1, stride))
    tot_ce, tot_n = 0.0, 0
    for b0 in range(0, len(starts), batch_size):
        chunk = starts[b0:b0 + batch_size]
        x = torch.from_numpy(np.stack([val[i:i + seq_len] for i in chunk]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([val[i + 1:i + seq_len + 1] for i in chunk]).astype(np.int64)).to(device)
        _, _, states = model(x, n_loops=loops, return_all_loops=False,
                             supervise_idx=set(), return_states=True)
        cos, sin = model.rope(seq_len, x.device, states[0].dtype)
        lg = model.readout(states[loops - 1], cos, sin)
        # score ONLY the trailing `stride` positions -> each has >= seq_len-stride tokens of context
        lg, yy = lg[:, -stride:, :], y[:, -stride:]
        tot_ce += F.cross_entropy(lg.reshape(-1, lg.size(-1)), yy.reshape(-1),
                                  reduction="sum").item()
        tot_n += yy.numel()
        if (b0 // batch_size) % 25 == 0:
            print(f"  {b0}/{len(starts)} windows", flush=True)
    return tot_ce / tot_n, tot_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("--loops", type=int, required=True)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--n-tokens", type=int, default=400_000)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cp = pathlib.Path(args.checkpoint)
    model, cfg, ck = load_checkpoint(cp / "last.pt" if cp.is_dir() else cp, dev)
    seq_len = ck["train_cfg"]["seq_len"]
    val = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")

    ce_s, n_s = sliding(model, val, args.loops, seq_len, args.stride, args.n_tokens, dev,
                        args.batch_size)
    # chunked control on the SAME token range, so the two differ only by protocol
    ce_c, n_c = sliding(model, val, args.loops, seq_len, seq_len, args.n_tokens, dev,
                        args.batch_size)
    b = lambda c: c / (BYTES_PER_TOKEN * math.log(2))
    avg_ctx_s = seq_len - args.stride / 2
    print(f"\ncheckpoint {cp.name} at loop {args.loops}, {ck.get('tokens'):,} training tokens")
    print(f"  CHUNKED   (stride {seq_len}): CE {ce_c:.4f}  bpb {b(ce_c):.4f}  "
          f"n={n_c:,}  avg left ctx ~{seq_len/2:.0f} tok")
    print(f"  SLIDING   (stride {args.stride}): CE {ce_s:.4f}  bpb {b(ce_s):.4f}  "
          f"n={n_s:,}  avg left ctx ~{avg_ctx_s:.0f} tok")
    print(f"  protocol effect: {ce_s-ce_c:+.4f} nats = {b(ce_s)-b(ce_c):+.4f} bits/byte")
    out = cp / f"sliding_{cp.name}.json"
    out.write_text(json.dumps(dict(loops=args.loops, stride=args.stride,
                                    chunked_ce=ce_c, sliding_ce=ce_s,
                                    chunked_bpb=b(ce_c), sliding_bpb=b(ce_s),
                                    n_scored_sliding=n_s, n_scored_chunked=n_c,
                                    tokens=ck.get("tokens")), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
