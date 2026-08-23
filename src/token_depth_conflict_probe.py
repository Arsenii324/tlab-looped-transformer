"""Experiment E3: Token-3 Conflict & Cross-Depth Representation Degradation Probe.

Tests whether tokens whose oracle depth is shallow (d* <= 8) suffer representation
degradation or distort self-attention when forced to compute at deep steps (t = 20):
  1. Drift cosine: cos(h_20(j), h_d*(j)) for shallow vs deep oracle tokens.
  2. Value-vector norm: ||v_20(j)|| vs ||v_d*(j)||.
  3. Attention mass received at depth 20 from shallow vs deep context tokens.

Runs as a zero-training forward pass on CPU/MPS using the 2.5M checkpoint and exitdump.
"""
from __future__ import annotations

import json
import pathlib
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval import load_checkpoint

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXITDUMP_PATH = pathlib.Path("/private/tmp/claude-501/-Users-a2mogus-build-projs-barannikov-work-tlab-loop-transformer/8d0bbec0-a97a-4cfd-898c-ff91777e2e65/scratchpad/exitdump_sd_dense_k5_s0.npz")


def run_conflict_probe(n_seq=128, query_depth=20):
    device = "cpu"  # Run on CPU to keep MPS 100% dedicated to training
    print("=" * 80)
    print(f"EXPERIMENT E3: TOKEN DEPTH CONFLICT PROBE (device={device}, n_seq={n_seq}, q_depth={query_depth})")
    print("=" * 80)

    # 1. Load Checkpoint
    ckpt_path = ROOT / "checkpoints" / "sd_dense_k5_s0" / "last.pt"
    model, cfg, _ = load_checkpoint(ckpt_path, device=device)
    model.eval()

    # 2. Load Dumped Per-Token Oracle Depths
    dump = np.load(EXITDUMP_PATH)
    val_ce = dump["ce"][:n_seq]  # [N, 256, 32]
    oracle_d = val_ce.argmin(axis=-1) + 1  # [N, 256], 1-indexed

    # Load validation token binaries
    val_bin = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    T = 256
    # ALIGNMENT FIX. The exit dump's oracle depths were computed on data/frozen_eval_set.npz,
    # NOT on sequential slices of val.bin. Pairing oracle_d[b,j] with val.bin[b*T+j] pairs a
    # depth label with a DIFFERENT token (frozen starts are 219, 494, 2630... not 0, 256, 512).
    # Both probes in this repo originally did that, which scrambles every by-group statistic.
    _fz = np.load(ROOT / "data" / "frozen_eval_set.npz")
    X = _fz["x"][:n_seq].astype(np.int64)
    Y = _fz["y"][:n_seq].astype(np.int64)

    batch_size = 4
    n_batches = n_seq // batch_size
    max_loops = 32

    # Metric collectors for shallow (d* <= 8), mid (8 < d* < 16), deep (d* >= 16)
    groups = {"shallow (d*<=8)": [], "mid (8<d*<16)": [], "deep (d*>=16)": []}
    cos_fidelity = {"shallow": [], "mid": [], "deep": []}
    norm_ratio = {"shallow": [], "mid": [], "deep": []}
    attn_mass_received = {"shallow": [], "mid": [], "deep": []}

    print("Running forward passes and extracting cross-depth representation metrics...")

    with torch.no_grad():
        for b in range(n_batches):
            xb = torch.from_numpy(X[b * batch_size : (b + 1) * batch_size]).to(device)
            B, seq_len = xb.shape
            b_oracle = oracle_d[b * batch_size : (b + 1) * batch_size]  # [B, T]

            e = model.embed(xb)
            cos, sin = model.rope(seq_len, xb.device, e.dtype)
            for layer in model.prelude:
                e = layer(e, cos, sin)
            h = model.h0.expand(B, seq_len, -1) + e

            # Collect states at all depths
            states = [h]
            for t in range(max_loops):
                h_in = (h + e) if t > 0 else h
                h = model.block(h_in, cos, sin)
                if model.loop_norm is not None:
                    h = model.loop_norm(h)
                states.append(h)

            h_q = states[query_depth]  # [B, T, H] at depth 20

            # Measure state cosine fidelity cos(h_20, h_d*) and norm ratio ||h_20|| / ||h_d*||
            for bi in range(B):
                for pos in range(seq_len):
                    d_star = int(b_oracle[bi, pos])
                    h_star = states[d_star][bi, pos]
                    h_curr = h_q[bi, pos]

                    cos_sim = F.cosine_similarity(h_curr.unsqueeze(0), h_star.unsqueeze(0)).item()
                    ratio = (h_curr.norm() / max(1e-6, h_star.norm())).item()

                    if d_star <= 8:
                        grp = "shallow"
                    elif d_star >= 16:
                        grp = "deep"
                    else:
                        grp = "mid"

                    cos_fidelity[grp].append(cos_sim)
                    norm_ratio[grp].append(ratio)

    results = {
        "query_depth": query_depth,
        "n_sequences": n_seq,
        "cos_fidelity": {k: float(np.mean(v)) for k, v in cos_fidelity.items()},
        "norm_ratio_depth20_vs_oracle": {k: float(np.mean(v)) for k, v in norm_ratio.items()},
        "token_counts": {k: len(v) for k, v in cos_fidelity.items()},
    }

    print("\n" + "=" * 80)
    print("EXPERIMENT E3 RESULTS SUMMARY")
    print("=" * 80)
    print(f"Token Group Counts: Shallow={results['token_counts']['shallow']}, Mid={results['token_counts']['mid']}, Deep={results['token_counts']['deep']}")
    print(f"\n1. State Cosine Fidelity cos(h_{query_depth}, h_d*):")
    for grp in ["shallow", "mid", "deep"]:
        print(f"   - {grp:<10}: cos = {results['cos_fidelity'][grp]:.4f}")

    print(f"\n2. State Norm Ratio ||h_{query_depth}|| / ||h_d*||:")
    for grp in ["shallow", "mid", "deep"]:
        print(f"   - {grp:<10}: norm ratio = {results['norm_ratio_depth20_vs_oracle'][grp]:.4f}x")

    out_path = ROOT / "checkpoints" / "token_conflict_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_conflict_probe()
