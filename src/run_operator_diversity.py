"""Operator Diversity & Depth Gating vs Autonomous Baseline.

Paired in-job benchmark across 4 arms:
  1. od_control: Autonomous baseline (cond_mode="none", depth_gate_mode="none")
  2. od_lora_r2: Loop-cycled LoRA r=2 across 4 branches (+204k params)
  3. od_lora_r4: Loop-cycled LoRA r=4 across 4 branches (+365k params)
  4. od_depth_gate: Learned state-conditioned depth gate (+448 params)

Trained under the identical seed, data order, and evaluation cadence as sd_dense_k5_s0.
"""
from __future__ import annotations

import json
import pathlib
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig
from train import TrainConfig
from chunked_runner import run_chunked

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAYER_APPS_PER_S = 1100.0 * 18 * 3
REF = "sd_dense_k5_s0"


def evaluate_cross_layer_alignment(model, val_x):
    """Measures minimum cross-layer cosine alignment at loops 8, 16, 32, 64."""
    layers = list(model.block.layers)
    L = len(layers)
    caught = []
    hs = [l.register_forward_hook(lambda m, i, o, _l=n: caught.append((_l, o.detach())))
          for n, l in enumerate(layers)]
    with torch.no_grad():
        model(val_x, n_loops=64, supervise_idx=set())
    for h in hs:
        h.remove()

    res = {}
    for t in [8, 16, 32, 64]:
        t_idx = t - 1
        layer_states = [caught[t_idx * L + l][1].float() for l in range(L)]
        u = [s / s.norm(dim=-1, keepdim=True).clamp_min(1e-12) for s in layer_states]
        cos_01 = (u[0] * u[1]).sum(-1).mean().item()
        cos_02 = (u[0] * u[2]).sum(-1).mean().item()
        cos_12 = (u[1] * u[2]).sum(-1).mean().item()
        res[f"loop_{t}_min_cos"] = min(cos_01, cos_02, cos_12)
    return res


def main():
    ref_path = ROOT / "checkpoints" / REF / "last.pt"
    assert ref_path.exists(), f"Reference donor checkpoint {ref_path} not found"
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    base_m, base_t = dict(ref["model_cfg"]), dict(ref["train_cfg"])
    tok = base_t["total_tokens"]
    cap = tok * 18 * 3 / LAYER_APPS_PER_S * 2.2

    arms = {
        "od_control": {},
        "od_lora_r2": {"cond_mode": "lora_cycle", "cond_lora_rank": 2, "cond_lora_branches": 4},
        "od_lora_r4": {"cond_mode": "lora_cycle", "cond_lora_rank": 4, "cond_lora_branches": 4},
        "od_depth_gate": {"depth_gate_mode": "state"},
    }

    out_file = ROOT / "checkpoints" / "operator_diversity_results.json"
    res = json.loads(out_file.read_text()) if out_file.exists() else {}

    print("=" * 80)
    print(f"OPERATOR DIVERSITY & DEPTH GATING EXPERIMENT: {len(arms)} ARMS @ {tok:,} TOKENS")
    print("=" * 80)

    for name, overrides in arms.items():
        if name in res and res[name].get("done", False):
            print(f"{name}: already completed, skipping", flush=True)
            continue

        mcfg = ModelConfig(**{**base_m, **overrides})
        tcfg = TrainConfig(**{**base_t, "run_name": name})
        
        steps_per_eval = tcfg.eval_every_tokens // (tcfg.batch_size * tcfg.seq_len)
        assert steps_per_eval <= 200, (
            f"{name}: eval_every_steps={steps_per_eval} too sparse for 240s chunks")

        print(f"\n=== LAUNCHING {name} (overrides={overrides}) ===", flush=True)
        res[name] = run_chunked(name, mcfg, tcfg, cap, fresh=True)
        h = res[name].get("history", [])
        if h:
            c = h[-1]["val_curve"]
            best_r = min(c, key=c.get)
            print(f"  {name} FINAL: best r={best_r} CE={c[best_r]:.4f} CE@1={c['1']:.4f} gain={c['1']-c[best_r]:.4f}", flush=True)

        res[name]["done"] = True
        out_file.write_text(json.dumps(res, indent=2))

    print("\n" + "=" * 80)
    print("ALL ARMS COMPLETED. RESULTS SUMMARY:")
    print("=" * 80)
    for name in arms:
        if name in res and "history" in res[name] and res[name]["history"]:
            c = res[name]["history"][-1]["val_curve"]
            best_r = min(c, key=c.get)
            gain = c['1'] - c[best_r]
            print(f"{name:<16}: CE@1={c['1']:.4f}  CE_best={c[best_r]:.4f} (r={best_r:<2})  gain={gain:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
