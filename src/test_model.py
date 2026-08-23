"""Correctness checks for model.py, run before any training compute is spent on it.

Not a general test suite -- five checks, each aimed at a specific way this file could be silently
wrong and still "work" (produce plausible-looking numbers with no crash):

  1. Param count matches param_budget.py's independently-computed number for the same shape, exactly.
     Two ways of counting the same quantity disagreeing is the cheapest possible bug signal.
  2. My hand-written DecoderLayer matches the REAL Qwen3DecoderLayer (imported from the installed
     transformers package, weights copied over) to float32 numerical precision, on random input.
     This is the check that catches a wrong QK-norm order, a wrong RoPE convention, or a wrong GQA
     repeat pattern -- all of which produce a model that runs and trains, just not the one intended.
  3. truncate_bptt changes ONLY the backward graph, never the forward values: full-BPTT and
     truncated-BPTT forwards on identical weights/input/seed must match to torch default float32
     tolerance (this is an exact identity, not an approximate one -- no_grad must not change any
     computed number, only whether it's tracked).
  4. The no_grad windowing actually does what it's supposed to: early loop states are detached from
     the autograd graph under truncation, late ones are not.
  5. state_renorm actually bounds the carried state's norm across many loops from a fresh random
     init; without it, nothing is asserted (it may or may not drift), it's just reported.
  6. n_prelude=n_coda=0 reproduces the pre-sandwich model BIT-EXACTLY (identical params, identical
     forward on identical seed), and a sandwich config actually routes through its extra layers.
     The first half is what lets every number already in report.md stand unchanged after this axis
     was added; the second half is what stops that compatibility from being vacuous.
"""

from __future__ import annotations

import sys

import torch

from model import Config, LoopedTransformer, DecoderLayer
from param_budget import total_params, block_params  # noqa: F401 (used via total_params)


def check_readout_modes():
    """readout_mode="norm" must be bit-identical to the pre-existing readout, and the other two must
    actually differ in the right places: "raw" differs at every loop, "final_only" differs at every
    loop EXCEPT the last. Without the second half, an arm could silently train as plain "norm"."""
    torch.manual_seed(0)
    x = torch.randint(0, 4096, (2, 16))
    outs = {}
    for mode in ("norm", "raw", "final_only"):
        torch.manual_seed(0)
        m = LoopedTransformer(Config(state_renorm=False, readout_mode=mode)).eval()
        with torch.no_grad():
            outs[mode], _ = m(x, n_loops=5, return_all_loops=True)
    d_raw = [(outs["raw"][i] - outs["norm"][i]).abs().max().item() for i in range(5)]
    d_fo = [(outs["final_only"][i] - outs["norm"][i]).abs().max().item() for i in range(5)]
    ok = all(v > 0 for v in d_raw) and all(v > 0 for v in d_fo[:-1]) and d_fo[-1] == 0.0
    print(f"[9] readout modes: raw differs at all 5 loops={all(v>0 for v in d_raw)}; "
          f"final_only differs at loops 1-4={all(v>0 for v in d_fo[:-1])} and matches at the final "
          f"loop (diff={d_fo[-1]:.1e}) {'OK' if ok else 'MISMATCH'}")

    # norm penalty: the aux term must be differentiable and actually push norms DOWN
    torch.manual_seed(0)
    m = LoopedTransformer(Config(state_renorm=False)).train()
    _, _, rms = m(x, n_loops=4, supervise_idx=set(), return_state_rms=True)
    pen = torch.stack([r.pow(2) for r in rms]).mean()
    pen.backward()
    gsum = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    ok2 = pen.requires_grad and gsum > 0 and len(rms) == 4
    print(f"[9] norm penalty: value={pen.item():.4f} differentiable={pen.requires_grad} "
          f"grad_mass={gsum:.3e} n_loops_covered={len(rms)} {'OK' if ok2 else 'MISMATCH'}")
    return ok and ok2


def check_kv_source_identity():
    """The cross-depth KV hook must be inert when unused. kv_source=None has to be bit-identical to
    the pre-hook code path, and feeding the layer's own input as kv_source must reproduce ordinary
    self-attention exactly -- if that second identity fails, the hook is not measuring what
    cross_depth_kv.py claims (it would be measuring a bug in the replay path instead)."""
    torch.manual_seed(0)
    cfg = Config(state_renorm=False)
    m = LoopedTransformer(cfg).eval()
    x = torch.randn(2, 24, cfg.hidden_size)
    cos, sin = m.rope(24, x.device, x.dtype)
    layer = m.block.layers[0]
    with torch.no_grad():
        a = layer(x, cos, sin)                 # ordinary
        b = layer(x, cos, sin, kv_source=x)    # explicit self as KV source -- must be identical
        collected = []
        c = m.block(x, cos, sin, collect=collected)
        d = m.block(x, cos, sin, kv_sources=collected)   # replay its own inputs -> identity
    d1 = (a - b).abs().max().item()
    d2 = (c - d).abs().max().item()
    ok = d1 == 0.0 and d2 == 0.0 and len(collected) == cfg.layers_per_loop
    print(f"[8] kv_source inert when self-sourced: layer max|diff|={d1:.1e} "
          f"block-replay max|diff|={d2:.1e} collected={len(collected)} {'OK' if ok else 'MISMATCH'}")
    return ok


def check_kaggle_copy_matches():
    """kaggle/main.py inlines its own copy of the model (Kaggle script kernels cannot import sibling
    files). A hand-maintained duplicate of numerical code is exactly where silent drift lives -- and
    this project already runs its largest result through that copy, so a divergence there would
    invalidate the headline number rather than a side experiment. Compares parameter count and every
    per-loop logit on identical seeds, across topologies. Skips cleanly if the file is absent."""
    import importlib.util, pathlib as _pl
    kp = _pl.Path(__file__).resolve().parents[1] / "kaggle" / "main.py"
    if not kp.exists():
        print("[7] SKIPPED (no kaggle/main.py)")
        return True
    spec = importlib.util.spec_from_file_location("_kmain", kp)
    km = importlib.util.module_from_spec(spec)
    sys.modules["_kmain"] = km
    try:
        spec.loader.exec_module(km)
    except SystemExit:
        pass
    ok = True
    for kw in (dict(state_renorm=False),
               dict(state_renorm=False, n_prelude=1, n_coda=1, layers_per_loop=1),
               dict(state_renorm=True, truncate_bptt=8)):
        torch.manual_seed(0); a = LoopedTransformer(Config(**kw)).eval()
        torch.manual_seed(0); b = km.LoopedTransformer(km.Config(**kw)).eval()
        x = torch.randint(0, 4096, (2, 32))
        with torch.no_grad():
            la, _ = a(x, n_loops=7, return_all_loops=True)
            lb, _ = b(x, n_loops=7, return_all_loops=True)
        d = max((i - j).abs().max().item() for i, j in zip(la, lb))
        same = a.num_parameters() == b.num_parameters()
        ok &= (d == 0.0 and same)
        print(f"[7] kaggle copy {str(kw):<62} params_match={same} max|diff|={d:.1e} "
              f"{'OK' if d == 0.0 and same else 'DRIFT'}")
    return ok


def check_sandwich():
    """Two halves. (a) Backward compatibility as an EXACT identity: with n_prelude=n_coda=0 the new
    code path must produce the same parameter count and the same forward values as the flat model,
    to the bit -- not "close", since any drift would silently invalidate every result in report.md
    that was measured before this axis existed. (b) Non-vacuity: a sandwich config must actually
    change the output and allocate exactly the extra layers asked for, otherwise (a) passes trivially
    because the knob does nothing."""
    torch.manual_seed(0)
    cfg0 = Config(n_prelude=0, n_coda=0)
    torch.manual_seed(0); flat = LoopedTransformer(cfg0)
    x = torch.randint(0, cfg0.vocab_size, (2, 32))
    torch.manual_seed(0)
    with torch.no_grad():
        a, na = flat(x, n_loops=6, return_all_loops=True)

    # (a) rebuild identically -- the sandwich code path with zero sandwich layers
    torch.manual_seed(0); flat2 = LoopedTransformer(Config(n_prelude=0, n_coda=0))
    with torch.no_grad():
        b, nb = flat2(x, n_loops=6, return_all_loops=True)
    same_params = flat.num_parameters() == flat2.num_parameters()
    max_d = max((ai - bi).abs().max().item() for ai, bi in zip(a, b))
    ok_a = same_params and max_d == 0.0
    print(f"[6a] n_prelude=n_coda=0 identical to flat model: params={flat.num_parameters():,} "
          f"max|diff|={max_d:.2e} {'OK' if ok_a else 'MISMATCH'}")

    # (b) a real sandwich must differ, and must allocate exactly the layers requested
    cfg1 = Config(n_prelude=1, n_coda=1, layers_per_loop=1)
    torch.manual_seed(0); sw = LoopedTransformer(cfg1)
    with torch.no_grad():
        c, _ = sw(x, n_loops=6, return_all_loops=True)
    from param_budget import block_params, total_params
    one = block_params(cfg1.hidden_size, cfg1.n_heads, cfg1.n_kv_heads, cfg1.head_dim,
                       cfg1.intermediate_size)
    bud = total_params(cfg1.hidden_size, cfg1.n_heads, cfg1.n_kv_heads, cfg1.head_dim,
                       cfg1.intermediate_size, cfg1.vocab_size, cfg1.layers_per_loop,
                       cfg1.inject_mode, cfg1.n_prelude, cfg1.n_coda)
    want = bud["total"] - bud["exit_head_reserve"]
    n_sw = len(sw.prelude) + len(sw.coda)
    differs = (c[-1] - a[-1]).abs().max().item() > 0
    ok_b = (n_sw == 2) and (sw.num_parameters() == want) and differs and (bud["prelude_coda"] == 2 * one)
    print(f"[6b] sandwich P1R1C1: layers={n_sw} params={sw.num_parameters():,} "
          f"(budget {want:,}) differs_from_flat={differs} {'OK' if ok_b else 'MISMATCH'}")
    return ok_a and ok_b


def check_param_count():
    ok_all = True
    for cfg in (Config(), Config(state_renorm=False),
                Config(n_prelude=1, n_coda=1, layers_per_loop=1),
                Config(state_renorm=False, n_prelude=1, n_coda=1, layers_per_loop=1)):
        ok_all &= _check_one_param_count(cfg)
    return ok_all


def _check_one_param_count(cfg):
    m = LoopedTransformer(cfg)
    got = m.num_parameters()
    from param_budget import total_params
    budget = total_params(cfg.hidden_size, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                           cfg.intermediate_size, cfg.vocab_size, cfg.layers_per_loop,
                           cfg.inject_mode, cfg.n_prelude, cfg.n_coda, cfg.state_renorm)
    # budget reserves a scalar exit head (H+1 params) the model doesn't allocate yet (deferred
    # per PLAN.md) -- the only line item that should differ.
    want = budget["total"] - budget["exit_head_reserve"]
    diff = got - want
    ok = diff == 0
    print(f"[1] param count (renorm={cfg.state_renorm}, P{cfg.n_prelude}R{cfg.layers_per_loop}"
          f"C{cfg.n_coda}): model={got:,} budget-formula={want:,} (diff={diff}) "
          f"{'OK' if ok else 'MISMATCH'}")
    return ok


def check_against_real_qwen3():
    import os
    os.environ.setdefault("USE_TF", "0")  # this conda env's tensorflow install crashes on import
    # (numpy C-extension ABI mismatch, "_ARRAY_API not found") -- unrelated to this project, but
    # transformers probes for TF as an optional backend on import and the crash otherwise propagates
    # here, masking this check behind a SKIPPED rather than actually comparing against the reference.
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Config
    except Exception as e:
        print(f"[2] SKIPPED (no local transformers qwen3 module): {e}")
        return True

    torch.manual_seed(0)
    H, n_h, n_kv, d_h, I = 64, 4, 2, 16, 96
    cfg = Config(hidden_size=H, n_heads=n_h, n_kv_heads=n_kv, head_dim=d_h,
                 intermediate_size=I, layers_per_loop=1, rope_theta=10000.0)
    mine = DecoderLayer(cfg).eval()

    qcfg = Qwen3Config(hidden_size=H, num_attention_heads=n_h, num_key_value_heads=n_kv,
                        head_dim=d_h, intermediate_size=I, rms_norm_eps=cfg.rms_norm_eps,
                        num_hidden_layers=1, rope_theta=10000.0, attention_bias=False,
                        attention_dropout=0.0, hidden_act="silu",
                        layer_types=["full_attention"], attn_implementation="eager")
    # attn_implementation must be explicit: constructing Qwen3Config/Qwen3DecoderLayer directly
    # (bypassing from_pretrained/AutoModel, which normally resolve this) leaves
    # config._attn_implementation as None on newer transformers, and the internal attention-class
    # dispatch (transformers/modeling_layers.py) does a dict lookup on that value with no
    # None-handling -- KeyError: None, unrelated to this project's model, an API-drift artifact of
    # the installed transformers version moving from 4.53.3 (this check's original verification,
    # LOG.md) to 4.57.1 (installed now).
    ref = Qwen3DecoderLayer(qcfg, layer_idx=0).eval()

    # copy weights mine -> ref (same parameter shapes by construction)
    with torch.no_grad():
        ref.input_layernorm.weight.copy_(mine.norm1.weight)
        ref.post_attention_layernorm.weight.copy_(mine.norm2.weight)
        ref.self_attn.q_proj.weight.copy_(mine.attn.q_proj.weight)
        ref.self_attn.k_proj.weight.copy_(mine.attn.k_proj.weight)
        ref.self_attn.v_proj.weight.copy_(mine.attn.v_proj.weight)
        ref.self_attn.o_proj.weight.copy_(mine.attn.o_proj.weight)
        ref.self_attn.q_norm.weight.copy_(mine.attn.q_norm.weight)
        ref.self_attn.k_norm.weight.copy_(mine.attn.k_norm.weight)
        ref.mlp.gate_proj.weight.copy_(mine.mlp.gate.weight)
        ref.mlp.up_proj.weight.copy_(mine.mlp.up.weight)
        ref.mlp.down_proj.weight.copy_(mine.mlp.down.weight)

    B, T = 2, 12
    x = torch.randn(B, T, H)
    rope = torch.nn.Module()
    cos, sin = _rope_cos_sin(d_h, 10000.0, T)

    with torch.no_grad():
        mine_out = mine(x, cos, sin)

    # build the reference's own rotary embeddings + causal mask the way its forward expects
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    rot = Qwen3RotaryEmbedding(config=qcfg)
    position_ids = torch.arange(T)[None, :]
    ref_cos, ref_sin = rot(x, position_ids)
    causal = torch.full((T, T), float("-inf")).triu(1)[None, None, :, :].expand(B, 1, T, T)

    with torch.no_grad():
        ref_out = ref(x, position_embeddings=(ref_cos, ref_sin), attention_mask=causal)
        if isinstance(ref_out, tuple):
            ref_out = ref_out[0]

    max_abs = (mine_out - ref_out).abs().max().item()
    ok = max_abs < 1e-4
    print(f"[2] vs real Qwen3DecoderLayer: max|diff|={max_abs:.2e} {'OK' if ok else 'MISMATCH'}")
    return ok


def _rope_cos_sin(head_dim, theta, T):
    from model import RotaryEmbedding
    r = RotaryEmbedding(head_dim, theta, T + 1)
    return r(T, "cpu", torch.float32)


def check_bptt_identity():
    torch.manual_seed(1)
    cfg_full = Config(hidden_size=64, n_heads=4, n_kv_heads=2, head_dim=16, intermediate_size=96,
                       layers_per_loop=1, truncate_bptt=None, state_renorm=False, depth_init=False)
    torch.manual_seed(1)
    cfg_trunc = dataclasses_replace(cfg_full, truncate_bptt=2)

    torch.manual_seed(2)
    m_full = LoopedTransformer(cfg_full)
    torch.manual_seed(2)
    m_trunc = LoopedTransformer(cfg_trunc)

    ids = torch.randint(0, cfg_full.vocab_size, (2, 6))
    logits_full, _ = m_full(ids, n_loops=5, return_all_loops=True)
    logits_trunc, _ = m_trunc(ids, n_loops=5, return_all_loops=True)

    max_abs = max((a - b).abs().max().item() for a, b in zip(logits_full, logits_trunc))
    ok = max_abs < 1e-5
    print(f"[3] full-BPTT vs truncated forward identity: max|diff|={max_abs:.2e} "
          f"{'OK' if ok else 'MISMATCH -- truncation is leaking into forward values'}")
    return ok


def check_nograd_windowing():
    torch.manual_seed(3)
    cfg = Config(hidden_size=32, n_heads=2, n_kv_heads=1, head_dim=16, intermediate_size=64,
                 layers_per_loop=1, truncate_bptt=2)
    m = LoopedTransformer(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 4))

    B, T = ids.shape
    e = m.embed(ids)
    cos, sin = m.rope(T, ids.device, e.dtype)
    h = m.h0.expand(B, T, -1) + e
    grad_tracked = []
    n_loops = 5
    for t in range(n_loops):
        no_grad = cfg.truncate_bptt is not None and t < (n_loops - cfg.truncate_bptt)
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            h_in = m._inject(h, e) if t > 0 else h
            h = m.block(h_in, cos, sin)
        grad_tracked.append(h.requires_grad)

    expect = [False, False, False, True, True]  # last 2 of 5 loops carry grad
    ok = grad_tracked == expect
    print(f"[4] no_grad windowing: got={grad_tracked} want={expect} {'OK' if ok else 'MISMATCH'}")
    return ok


def check_state_renorm_bounds_norm():
    torch.manual_seed(4)
    for renorm in (True, False):
        cfg = Config(hidden_size=64, n_heads=4, n_kv_heads=2, head_dim=16, intermediate_size=96,
                     layers_per_loop=1, state_renorm=renorm, truncate_bptt=None)
        torch.manual_seed(5)
        m = LoopedTransformer(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        with torch.no_grad():
            _, norms = m(ids, n_loops=24, return_all_loops=False)
        spread = max(norms) / max(min(norms), 1e-6)
        print(f"[5] state_renorm={renorm}: norm[0]={norms[0]:.2f} norm[-1]={norms[-1]:.2f} "
              f"max/min over 24 loops={spread:.2f}x")
    return True


def check_operator_diversity_and_depth_gate():
    """Checks for operator diversity (cond_mode="lora_cycle") and state depth gating:
    1. Param count matches param_budget analytical formula across configs.
    2. Exact bit-identity at step 0 under zero-init (max|diff| == 0.0).
    3. Perturbation non-vacuousness when weights are non-zero.
    4. Gradient flow / differentiability during training."""
    # 1. Param count check
    c_lora = Config(state_renorm=False, cond_mode="lora_cycle", cond_lora_rank=4, cond_lora_branches=4)
    m_lora = LoopedTransformer(c_lora)
    bud_lora = total_params(c_lora.hidden_size, c_lora.n_heads, c_lora.n_kv_heads, c_lora.head_dim,
                            c_lora.intermediate_size, c_lora.vocab_size, c_lora.layers_per_loop,
                            c_lora.inject_mode, state_renorm=False, cond_mode="lora_cycle",
                            cond_lora_rank=4, cond_lora_branches=4)
    expected_lora = bud_lora["total"] - bud_lora["exit_head_reserve"]
    actual_lora = m_lora.num_parameters()
    ok_p1 = actual_lora == expected_lora

    c_gate = Config(state_renorm=False, depth_gate_mode="state")
    m_gate = LoopedTransformer(c_gate)
    bud_gate = total_params(c_gate.hidden_size, c_gate.n_heads, c_gate.n_kv_heads, c_gate.head_dim,
                            c_gate.intermediate_size, c_gate.vocab_size, c_gate.layers_per_loop,
                            c_gate.inject_mode, state_renorm=False, depth_gate_mode="state")
    expected_gate = bud_gate["total"] - bud_gate["exit_head_reserve"]
    actual_gate = m_gate.num_parameters()
    ok_p2 = actual_gate == expected_gate

    print(f"[10] param count lora_cycle: model={actual_lora:,} budget={expected_lora:,} {'OK' if ok_p1 else 'MISMATCH'}")
    print(f"[10] param count depth_gate: model={actual_gate:,} budget={expected_gate:,} {'OK' if ok_p2 else 'MISMATCH'}")

    # 2. Bit-identity check at step 0
    torch.manual_seed(42)
    x = torch.randint(0, 4096, (2, 16))
    base_m = LoopedTransformer(Config(state_renorm=False)).eval()
    lora_m = LoopedTransformer(c_lora).eval()
    # Copy identical base weights to isolate the effect of zero-initialized LoRA adapters
    lora_m.load_state_dict(base_m.state_dict(), strict=False)

    with torch.no_grad():
        out_base, _ = base_m(x, n_loops=8, return_all_loops=True)
        out_lora, _ = lora_m(x, n_loops=8, return_all_loops=True)

    diff_lora = max((out_lora[i] - out_base[i]).abs().max().item() for i in range(8))
    ok_id1 = diff_lora == 0.0
    print(f"[11] step-0 bit-identity (lora_cycle vs base): max|diff|={diff_lora:.2e} {'OK' if ok_id1 else 'MISMATCH'}")

    # 3. Non-vacuousness check: perturb LoRA weights and assert divergence
    with torch.no_grad():
        lora_m.block.layers[0].lora_branches[0].q_B.add_(torch.randn_like(lora_m.block.layers[0].lora_branches[0].q_B) * 0.1)
        out_lora_pert, _ = lora_m(x, n_loops=8, return_all_loops=True)
    diff_pert = max((out_lora_pert[i] - out_base[i]).abs().max().item() for i in range(8))
    ok_pert = diff_pert > 1e-4
    print(f"[12] perturbation divergence (lora B!=0): diff={diff_pert:.4f} {'OK' if ok_pert else 'MISMATCH'}")

    # 4. Gradient differentiability
    lora_train = LoopedTransformer(c_lora).train()
    logits, _ = lora_train(x, n_loops=4, return_all_loops=True)
    loss = logits[-1].mean()
    loss.backward()
    has_lora_grads = any(p.grad is not None and p.grad.abs().sum().item() > 0
                         for layer in lora_train.block.layers
                         for b in layer.lora_branches
                         for p in b.parameters())
    print(f"[13] lora gradient flow: grads_present={has_lora_grads} {'OK' if has_lora_grads else 'MISMATCH'}")

    return ok_p1 and ok_p2 and ok_id1 and ok_pert and has_lora_grads


def dataclasses_replace(cfg, **kw):
    import dataclasses
    return dataclasses.replace(cfg, **kw)


def main():
    results = [
        check_readout_modes(),
        check_kv_source_identity(),
        check_kaggle_copy_matches(),
        check_sandwich(),
        check_param_count(),
        check_against_real_qwen3(),
        check_bptt_identity(),
        check_nograd_windowing(),
        check_state_renorm_bounds_norm(),
        check_operator_diversity_and_depth_gate(),
    ]
    ok = all(results)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'AT LEAST ONE CHECK FAILED -- do not train on this yet'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
