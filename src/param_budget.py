"""Search (hidden, heads, kv_heads, head_dim, intermediate, vocab, layers_per_loop) for a config
that hits the <=10M total-parameter budget with room to spare for an injection adapter.

Why this needs its own script rather than eyeballing it: a looped transformer's param count does
NOT grow with loop count (one block, reused), so the whole budget can go into a single block's
width -- the arithmetic isn't "how many layers fit" (a normal transformer question), it's "how wide
can one block be." Getting this wrong either wastes most of the budget on an oversized vocab
embedding or leaves the block too narrow to be worth looping.

Param count, one Qwen3-style block (bias=False everywhere, matching the reference implementation
read from transformers/models/qwen3/modeling_qwen3.py):
    attn:  q(H*nh*dh) + k(H*nkv*dh) + v(H*nkv*dh) + o(nh*dh*H) + q_norm(dh) + k_norm(dh)
    norms: input_layernorm(H) + post_attention_layernorm(H)
    mlp:   gate(H*I) + up(H*I) + down(I*H)
Embedding (tied with lm_head): V*H. Final norm: H. Optional injection adapter: 2*H*H (concat mode).
"""

from __future__ import annotations

import itertools


def block_params(H: int, n_h: int, n_kv: int, d_h: int, I: int) -> int:
    attn = H * n_h * d_h + 2 * (H * n_kv * d_h) + n_h * d_h * H + 2 * d_h
    norms = 2 * H
    mlp = 3 * H * I
    return attn + norms + mlp


def lora_adapter_params(H: int, n_h: int, n_kv: int, d_h: int, I: int, rank: int) -> int:
    q = H * rank + (n_h * d_h) * rank
    k = H * rank + (n_kv * d_h) * rank
    v = H * rank + (n_kv * d_h) * rank
    o = H * rank + H * rank
    gate = H * rank + I * rank
    up = H * rank + I * rank
    down = H * rank + I * rank
    return q + k + v + o + gate + up + down


def total_params(H, n_h, n_kv, d_h, I, V, layers_per_loop, inject_mode,
                  n_prelude: int = 0, n_coda: int = 0, state_renorm: bool = True,
                  cond_mode: str = "none", cond_lora_rank: int = 4, cond_lora_branches: int = 4,
                  depth_gate_mode: str = "none") -> dict:
    blk = block_params(H, n_h, n_kv, d_h, I) * layers_per_loop
    lora_params = (lora_adapter_params(H, n_h, n_kv, d_h, I, cond_lora_rank) * cond_lora_branches * layers_per_loop
                   if cond_mode == "lora_cycle" else 0)
    embed = V * H  # tied
    final_norm = H            # readout norm, applied before the LM head
    # Separate RMSNorm confining the carried state. Only allocated when state_renorm=True -- this
    # used to be counted unconditionally, which silently overcounted every state_renorm=False config
    # by exactly H. Not caught earlier because test_model.py check [1] only exercised the default.
    loop_norm = H if state_renorm else 0
    h0 = H                     # learned initial state, decoupled from content
    adapter = 2 * H * H if inject_mode == "concat" else 0
    exit_head_reserve = H + 1  # tiny scalar halting head, reserved even if unused in v1
    depth_gate = H if depth_gate_mode == "state" else 0
    # Unshared prelude/coda layers (sandwich topology). These are NOT multiplied by loop count -- they
    # run once -- which is exactly why they are expensive here: they consume budget that would
    # otherwise sit in the block the loop reuses r times. At H=448 one layer is 2.41M of a 10M ceiling.
    once = block_params(H, n_h, n_kv, d_h, I) * (n_prelude + n_coda)
    total = blk + lora_params + embed + final_norm + loop_norm + h0 + adapter + exit_head_reserve + depth_gate + once
    return dict(block=blk, lora=lora_params, embed=embed, final_norm=final_norm, loop_norm=loop_norm, h0=h0,
                adapter=adapter, exit_head_reserve=exit_head_reserve, depth_gate=depth_gate, prelude_coda=once, total=total)


def search(budget=10_000_000, margin_low=8_500_000, margin_high=9_800_000):
    candidates = []
    for H in (192, 224, 256, 288, 320, 384, 448, 512, 576, 640):
        for n_h in (2, 4, 6, 8, 12, 16):
            if H % n_h != 0:
                continue
            d_h = H // n_h
            if d_h < 16 or d_h > 128:
                continue
            for n_kv in (1, 2, n_h):
                if n_kv > n_h or n_h % n_kv != 0:
                    continue
                for mlp_ratio in (2.0, 8 / 3, 3.0, 4.0):
                    I = int(round(H * mlp_ratio / 8) * 8)  # round to multiple of 8
                    for V in (3072, 4096, 6144, 8192):
                        for layers_per_loop in (1, 2, 3):
                            for inject_mode in ("additive", "concat"):
                                d = total_params(H, n_h, n_kv, d_h, I, V, layers_per_loop, inject_mode)
                                if margin_low <= d["total"] <= margin_high:
                                    candidates.append(dict(
                                        H=H, n_h=n_h, n_kv=n_kv, d_h=d_h, I=I, V=V,
                                        layers_per_loop=layers_per_loop, inject_mode=inject_mode,
                                        **d))
    return candidates


def main():
    cands = search()
    print(f"{len(cands)} configs in [8.5M, 9.8M] budget window")
    # Prefer: most of the budget in the reusable block, not the vocab table -- a fixed-size lookup
    # table is exactly the kind of thing the task warns will stop mattering at scale, and it's the
    # opposite of what a looped model is supposed to spend its params on. Then additive injection
    # (cheaper), then layers_per_loop=1 (purest "one block looped r times" reading of the task).
    def score(c):
        return (-c["block"] / c["total"], c["inject_mode"] != "additive", c["layers_per_loop"] != 1)
    cands.sort(key=score)
    print(f"\n{'H':>4} {'n_h':>4} {'n_kv':>4} {'d_h':>4} {'I':>5} {'V':>6} {'L':>2} {'inject':>9} "
          f"{'block':>9} {'embed':>9} {'total':>10} {'block/total':>12}")
    for c in cands[:15]:
        print(f"{c['H']:>4} {c['n_h']:>4} {c['n_kv']:>4} {c['d_h']:>4} {c['I']:>5} {c['V']:>6} "
              f"{c['layers_per_loop']:>2} {c['inject_mode']:>9} {c['block']:>9,} {c['embed']:>9,} "
              f"{c['total']:>10,} {c['block']/c['total']:>11.1%}")
    best = cands[0]
    print(f"\nSELECTED: H={best['H']} n_h={best['n_h']} n_kv={best['n_kv']} d_h={best['d_h']} "
          f"I={best['I']} V={best['V']} layers_per_loop={best['layers_per_loop']} "
          f"total={best['total']:,} params ({best['block']/best['total']:.1%} in the reusable block)")
    return best


if __name__ == "__main__":
    main()
