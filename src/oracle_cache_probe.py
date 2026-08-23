"""Experiment B: Oracle-depth KV Cache and Two-depth KV Cache Probing.

Evaluates whether depth heterogeneity can pay in the KV cache without full generation overhead:
  1. Uniform-depth cache baseline across depths k in {1, 2, 4, 8, 16, 24, 32}.
  2. Oracle-depth cache: each prefix token's KV is cached at its own oracle exit depth d*(j).
  3. Two-depth cache: concatenate KV from depths 4 and 16 into the attended context set.

Uses dumped arrays from exitdump_sd_dense_k5_s0.npz and the 2.5M checkpoint.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval import load_checkpoint
from model import apply_rope

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXITDUMP_PATH = pathlib.Path("/private/tmp/claude-501/-Users-a2mogus-build-projs-barannikov-work-tlab-loop-transformer/8d0bbec0-a97a-4cfd-898c-ff91777e2e65/scratchpad/exitdump_sd_dense_k5_s0.npz")


def run_cache_probe(ckpt_name="sd_dense_k5_s0", n_seq=128, query_depth=20):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("=" * 80)
    print(f"EXPERIMENT B: ORACLE-DEPTH & TWO-DEPTH KV CACHE PROBE (device={device})")
    print("=" * 80)

    # 1. Load Model
    ckpt_path = ROOT / "checkpoints" / ckpt_name / "last.pt"
    if not ckpt_path.exists():
        print(f"Error: checkpoint {ckpt_path} not found")
        return
    model, cfg, _ = load_checkpoint(ckpt_path, device=device)
    model.eval()

    # 2. Load Tokens and Oracle Depths from exit dump if present, or compute
    dump = np.load(EXITDUMP_PATH)
    val_ce = dump["ce"][:n_seq]  # [N, 256, 32]
    oracle_d = val_ce.argmin(axis=-1) + 1  # [N, 256], 1-indexed

    # Load validation tokens
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
    query_loops = [8, 16, 20, 24, 32]

    # Metrics accumulators
    uniform_loss = {k: 0.0 for k in [1, 2, 4, 8, 16, 20, 24, 32]}
    oracle_loss = {q: 0.0 for q in query_loops}
    twodepth_4_16_loss = {q: 0.0 for q in query_loops}
    twodepth_8_24_loss = {q: 0.0 for q in query_loops}
    total_tokens = 0

    print(f"Evaluating {n_seq} sequences (T={T}) across cache configurations...")

    with torch.no_grad():
        for b in range(n_batches):
            xb = torch.from_numpy(X[b * batch_size : (b + 1) * batch_size]).to(device)
            yb = torch.from_numpy(Y[b * batch_size : (b + 1) * batch_size]).to(device)
            B, seq_len = xb.shape
            b_oracle = oracle_d[b * batch_size : (b + 1) * batch_size]  # [B, T]

            e = model.embed(xb)
            cos, sin = model.rope(seq_len, xb.device, e.dtype)
            for layer in model.prelude:
                e = layer(e, cos, sin)
            h = model.h0.expand(B, seq_len, -1) + e

            # Collect states and per-layer inputs at all depths
            states = [h]
            layer_inputs_by_depth = []  # list of [L=3 tensors of shape [B, T, H]]
            for t in range(max_loops):
                h_in = (h + e) if t > 0 else h
                col = []
                h = model.block(h_in, cos, sin, collect=col)
                if model.loop_norm is not None:
                    h = model.loop_norm(h)
                layer_inputs_by_depth.append(col)
                states.append(h)

            def compute_ce_at(hh):
                lg = model.readout(hh, cos, sin)
                return F.cross_entropy(lg.reshape(-1, lg.size(-1)), yb.reshape(-1), reduction="sum").item()

            # (A) Uniform Caches at Query Depth 20
            q_depth = 20
            h_prev = states[q_depth - 1]
            h_in_q = (h_prev + e) if q_depth > 1 else h_prev

            for k in uniform_loss:
                hh_uniform = model.block(h_in_q, cos, sin, kv_sources=layer_inputs_by_depth[k - 1])
                if model.loop_norm is not None:
                    hh_uniform = model.loop_norm(hh_uniform)
                uniform_loss[k] += compute_ce_at(hh_uniform)

            # (B) Oracle-Depth Cache
            # For each layer l, construct ragged KV source: kv_oracle[b, pos, :] = layer_inputs[d*(b, pos)-1][l][b, pos, :]
            for q in query_loops:
                h_prev_q = states[q - 1]
                h_in_curr = (h_prev_q + e) if q > 1 else h_prev_q
                
                oracle_kv_sources = []
                for l_idx in range(len(model.block.layers)):
                    l_src = torch.zeros(B, seq_len, cfg.hidden_size, device=device, dtype=e.dtype)
                    for bi in range(B):
                        for pos in range(seq_len):
                            d_star = int(b_oracle[bi, pos])
                            d_idx = min(max(d_star - 1, 0), max_loops - 1)
                            l_src[bi, pos] = layer_inputs_by_depth[d_idx][l_idx][bi, pos]
                    oracle_kv_sources.append(l_src)

                hh_oracle = model.block(h_in_curr, cos, sin, kv_sources=oracle_kv_sources)
                if model.loop_norm is not None:
                    hh_oracle = model.loop_norm(hh_oracle)
                oracle_loss[q] += compute_ce_at(hh_oracle)

            total_tokens += xb.numel()

    # Summaries
    print("\n" + "=" * 80)
    print("EXPERIMENT B RESULTS SUMMARY")
    print("=" * 80)
    print(f"1. Uniform Cache vs Depth (Query Loop = 20):")
    best_uniform_k = None
    best_uniform_ce = 1e9
    for k in sorted(uniform_loss.keys()):
        ce_val = uniform_loss[k] / total_tokens
        if ce_val < best_uniform_ce:
            best_uniform_ce = ce_val
            best_uniform_k = k
        print(f"   - Cache Depth k={k:2d}: CE = {ce_val:.4f}")

    print(f"\n2. Oracle-Depth Ragged Cache:")
    for q in query_loops:
        ce_val = oracle_loss[q] / total_tokens
        print(f"   - Query Depth q={q:2d}: Oracle-Cache CE = {ce_val:.4f}")

    out_data = {
        "uniform_cache_q20": {k: uniform_loss[k] / total_tokens for k in uniform_loss},
        "oracle_cache": {q: oracle_loss[q] / total_tokens for q in oracle_loss},
        "best_uniform_k": best_uniform_k,
        "best_uniform_ce": best_uniform_ce,
    }
    out_path = ROOT / "checkpoints" / "oracle_cache_results.json"
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"\nResults saved to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_cache_probe()
