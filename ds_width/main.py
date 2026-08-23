"""DataSphere driver: duo-causal attention + scale-invariant depth gate, seed 0.
In-job paired. RECURRENCE-side (duo-causal W=2,3) vs READOUT-side (gate) vs control.
"""
from __future__ import annotations
import contextlib, dataclasses, json, math, os, subprocess, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

T0 = time.time()
MAX_SWEEP_SECONDS = 3.0 * 3600
OUT_DIR = "."
RESULTS_PATH = "results.json"

def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


# === MODEL DEFINITION ===
"""A weight-tied looped transformer, inner block derived from Qwen3 (read directly from the
installed transformers==4.53.3 reference, not from memory -- see PLAN.md sec 3 for the confirmed
mechanics: pre-norm RMSNorm, QK-norm per head on head_dim before RoPE, GQA, SwiGLU, no biases).

The outer loop -- one block of `layers_per_loop` Qwen3-style layers, applied `r` times with shared
weights -- is the actual modification the task asks for. Five toggles, each grounded in a specific
prior result (PLAN.md sec 3), are implemented here as plain config flags, not a plugin system:

  truncate_bptt   None = full backprop through every loop; int k = only the last k loops carry
                  gradient (torch.no_grad() around the rest), reproducing Huginn's own training
                  recipe for direct comparison.
  state_renorm    if True, RMSNorm the full state after each loop before re-injection (Huginn's
                  explicit sphere confinement); if False, the state is only implicitly normed at
                  each sublayer's input, standard pre-norm residual style.
  inject_mode     "none" | "additive" | "concat" -- how the input embedding re-enters at every loop.
  depth_init      if True, output projections (o_proj, down_proj) are scaled by an extra
                  1/sqrt(2*n_loop_eff) at init, matching Huginn's depth-aware initialization.
  (loop-count randomization lives in train.py, not here -- it's a sampling policy over `r`, not a
  model structure choice.)
  n_prelude /    unshared layers applied once before / after the loop -- the "sandwich" topology
  n_coda         (Huginn, Ouro, Parcae all have one; this model originally had neither, a pure flat
                 loop with a learned h0). Both default to 0, which reproduces the flat model EXACTLY
                 (test_model.py check [6] pins this as a bit-exact identity, not an approximation).
                 Note the budget arithmetic, because it is the whole design tension at this scale:
                 one DecoderLayer at H=448 is 2,409,568 params, so a naive P1 R3 C1 costs 13.88M
                 against a 10M ceiling. A prelude and coda are near-free at 730M params and are NOT
                 free here -- they must be paid for out of the recurrent block, which is the thing
                 loops multiply. Any prelude/coda config in this project therefore trades logical
                 depth-per-loop for unshared input/output specialization, and that trade is the
                 hypothesis under test, not a free win.

Every loop's state is also read out through the (tied) LM head, not just the final one -- this is
what makes "how much does each loop help" a directly trainable and measurable quantity, and it's
also the reason a Readout Blind Spot (arXiv 2606.24898) is a real risk here: a dense per-loop loss
through a scale-invariant readout does not constrain the *raw* state's scale, only its normalized
direction. `state_renorm` is the fix; `forward()` also returns the raw per-loop state norm so
train.py can watch for it happening anyway when the toggle is off.
"""





@dataclasses.dataclass
class Config:
    vocab_size: int = 4096
    hidden_size: int = 448
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 112
    intermediate_size: int = 1344
    layers_per_loop: int = 3
    n_prelude: int = 0                     # unshared layers applied ONCE before the loop
    n_coda: int = 0                        # unshared layers applied ONCE at each readout
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 512

    # the five ablation axes
    readout_mode: str = "norm"             # "norm" | "raw" | "final_only" -- see readout()
    convex_gate: bool = False              # h_t = (1-g)h_{t-1} + g*block(...); see _gate()
    explore_noise: float = 0.0             # sigma for stochastic exploration during loops; see below
    explore_anneal: bool = True            # scale sigma by 1/sqrt(t) (Langevin-style) if True
    fixed_gate: float | None = None        # if set, g is this CONSTANT instead of learned. Turns the
    # gate into a swept parameter with g=1.0 being exactly the ungated model, so the arm series reads
    # as a monotone trend rather than a single pairwise A/B (METHODS.md rule 2). Costs 0 params.
    truncate_bptt: int | None = None       # None = full BPTT
    state_renorm: bool = True
    inject_mode: str = "additive"          # "none" | "additive" | "concat" | "gated" -- see _inject()
    depth_init: bool = True
    residual_scale: float | None = None    # if set, lambda in eps = lambda/(N*sqrt(L)); see below
    scale_clock: bool = False              # feed log||h|| back into the block's input; see _clock()
    gate_alpha_init: float = 0.874         # inject_mode="gated": INITIAL per-channel carry decay.
    # Swept rather than fixed, following the `fixed_gate` precedent (METHODS.md rule 2), because the
    # first screen showed this value decides whether the mechanism is testable at all -- see _inject().
    n_loop_eff: int = 24                   # for depth_init scaling; ~ mean of train-time loop sampler

    # Operator diversity & Depth gating extensions (Gemini Antigravity Agent)
    cond_mode: str = "none"                # "none" | "lora_cycle" -- loop-cycled LoRA adapters
    cond_lora_rank: int = 4                # rank r for LoRA branches (default: 4)
    cond_lora_branches: int = 4            # number of cycled branches N (default: 4)
    kv_untie_buckets: int = 1              # >1: give W_K this many DISTINCT matrices,
    #   assigned by loop index (t % nb). Bucket 0 reuses the shared k_proj, so the added
    #   cost is exactly (nb-1)*H*(n_kv*d_h). sec4.28 prices the rank this buys.
    depth_gate_mode: str = "none"          # "none" | "state" | "state_norm" -- learned depth gate
    kv_window: int = 1                     # DUO-CAUSAL attention window. 1 = ordinary self-attention.
    #   W > 1: at loop t every layer attends over the K/V of its own inputs from loops t-W+1..t,
    #   concatenated along the key axis under a token-causal mask replicated across depths. This is
    #   Think-at-Hard's duo-causal attention (arXiv 2511.08577, verified from tarball: "across both
    #   previous positions and shallower iteration depths", "enforced via a modified additive
    #   attention mask, requiring no custom CUDA kernels"). Positions are encoded by TOKEN index only,
    #   depth-invariant, exactly as they specify. ZERO added parameters.
    #
    #   Why this axis and not another: every instrument this project has aimed at the per-token depth
    #   headroom -- label-free rules (§4.7), static readout mixtures (§4.7c), the annealed retest
    #   (§4.7a), the oracle-depth cache (§4.8b), the learned gate (§4.22) -- is READOUT-side. All five
    #   read the finished trajectory and select or blend it; none changes what the block SEES at loop
    #   t. This does. §4.3's anchor result says the forcing bias exceeds the model's own per-step
    #   motion from ~loop 2, so the update is largely history-independent -- and history is precisely
    #   what the block cannot compute from h_t alone.
    #
    #   Cost at these shapes (T=256): attention-score FLOPs scale as W, and scores are ~0.46 of ~5.3
    #   MFLOP per token per layer, so W=2 is ~+9% and W=3 ~+17% per loop. Memory holds W-1 past
    #   per-layer inputs, [B,T,H] each -- ~11 MB at W=3, B=8, unlike the depth gate's O(r).


# ---------------------------------------------------------------------------------------- primitives

class RMSNorm(nn.Module):
    """Bit-identical to Qwen3RMSNorm (modeling_qwen3.py): float32 upcast, weight applied after cast
    back to input dtype. Verified against the real module below, not assumed."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float, max_pos: int):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_pos).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device, dtype):
        return (self.cos[:seq_len].to(device=device, dtype=dtype),
                self.sin[:seq_len].to(device=device, dtype=dtype))


def apply_rope(q, k, cos, sin):
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_h = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.d_h = cfg.head_dim
        self.groups = self.n_h // self.n_kv
        self.scale = 1.0 / math.sqrt(self.d_h)
        H = cfg.hidden_size
        self.q_proj = nn.Linear(H, self.n_h * self.d_h, bias=False)
        self.k_proj = nn.Linear(H, self.n_kv * self.d_h, bias=False)
        self.v_proj = nn.Linear(H, self.n_kv * self.d_h, bias=False)
        self.o_proj = nn.Linear(self.n_h * self.d_h, H, bias=False)
        self.q_norm = RMSNorm(self.d_h, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.d_h, cfg.rms_norm_eps)
        nb = max(1, int(getattr(cfg, "kv_untie_buckets", 1)))
        self.k_buckets = nb
        self.k_proj_extra = (nn.ModuleList(nn.Linear(H, self.n_kv * self.d_h, bias=False)
                                           for _ in range(nb - 1)) if nb > 1 else None)

    def forward(self, x, cos, sin, kv_source=None, lora=None, kv_extra=None, bucket=None):
        """`kv_source`, when given, supplies keys and values while `x` still supplies queries. This
        is the one hook needed to ask what an early-exit KV cache costs: under teacher forcing every
        position is processed at every depth, so "the context exited at depth k while this token is
        computing at depth t" is exactly q from the depth-t stream against k/v from the depth-k one.
        Defaults to None = ordinary self-attention, bit-identical (test_model.py check [8])."""
        B, T, _ = x.shape
        kv = x if kv_source is None else kv_source
        q_raw = self.q_proj(x)
        kp = self.k_proj
        if self.k_proj_extra is not None and bucket is not None:
            b = bucket % self.k_buckets
            kp = self.k_proj if b == 0 else self.k_proj_extra[b - 1]
        k_raw = kp(kv)
        v_raw = self.v_proj(kv)
        if lora is not None:
            q_raw = q_raw + F.linear(F.linear(x, lora.q_A), lora.q_B)
            k_raw = k_raw + F.linear(F.linear(kv, lora.k_A), lora.k_B)
            v_raw = v_raw + F.linear(F.linear(kv, lora.v_A), lora.v_B)

        q_pre = self.q_norm(q_raw.view(B, T, self.n_h, self.d_h)).transpose(1, 2)
        k_pre = self.k_norm(k_raw.view(B, T, self.n_kv, self.d_h)).transpose(1, 2)
        v = v_raw.view(B, T, self.n_kv, self.d_h).transpose(1, 2)
        q, k = apply_rope(q_pre, k_pre, cos, sin)
        if kv_extra:
            # DUO-CAUSAL: concatenate the K/V of this layer's inputs from earlier loops along the KEY
            # axis. RoPE uses the same cos/sin for every depth -- positions are token-indexed and
            # depth-invariant, which is what makes the extra keys addressable at all.
            ks, vs = [], []
            for e in kv_extra:
                ek_raw, ev_raw = kp(e), self.v_proj(e)
                if lora is not None:
                    ek_raw = ek_raw + F.linear(F.linear(e, lora.k_A), lora.k_B)
                    ev_raw = ev_raw + F.linear(F.linear(e, lora.v_A), lora.v_B)
                ek = self.k_norm(ek_raw.view(B, T, self.n_kv, self.d_h)).transpose(1, 2)
                ev = ev_raw.view(B, T, self.n_kv, self.d_h).transpose(1, 2)
                _, ek = apply_rope(q_pre, ek, cos, sin)  # q_pre discarded; apply_rope is stateless
                ks.append(ek); vs.append(ev)
            k = torch.cat(ks + [k], dim=2)   # current depth LAST, so W=1 is the untouched path
            v = torch.cat(vs + [v], dim=2)
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            mask = causal.repeat(1, len(kv_extra) + 1)  # same token-causality at every depth
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
        else:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, T, self.n_h * self.d_h)
        o_out = self.o_proj(out)
        if lora is not None:
            o_out = o_out + F.linear(F.linear(out, lora.o_A), lora.o_B)
        return o_out


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x, lora=None):
        gate_out = self.gate(x)
        up_out = self.up(x)
        if lora is not None:
            gate_out = gate_out + F.linear(F.linear(x, lora.gate_A), lora.gate_B)
            up_out = up_out + F.linear(F.linear(x, lora.up_A), lora.up_B)
        act = F.silu(gate_out) * up_out
        down_out = self.down(act)
        if lora is not None:
            down_out = down_out + F.linear(F.linear(act, lora.down_A), lora.down_B)
        return down_out


class LoRALayerAdapter(nn.Module):
    """Parameter-efficient LoRA branch for a single DecoderLayer (MoDr lineage).
    Adds low-rank adapters with B initialized to zero, ensuring exact step-0 bit-identity."""

    def __init__(self, cfg: Config, rank: int = 4):
        super().__init__()
        H = cfg.hidden_size
        I = cfg.intermediate_size
        self.rank = rank

        # In PyTorch F.linear(x, W): W is (out_features, in_features).
        # A projects in_dim -> rank: shape is (rank, in_dim).
        # B projects rank -> out_dim: shape is (out_dim, rank).
        self.q_A = nn.Parameter(torch.randn(rank, H) / math.sqrt(H))
        self.q_B = nn.Parameter(torch.zeros(cfg.n_heads * cfg.head_dim, rank))

        self.k_A = nn.Parameter(torch.randn(rank, H) / math.sqrt(H))
        self.k_B = nn.Parameter(torch.zeros(cfg.n_kv_heads * cfg.head_dim, rank))

        self.v_A = nn.Parameter(torch.randn(rank, H) / math.sqrt(H))
        self.v_B = nn.Parameter(torch.zeros(cfg.n_kv_heads * cfg.head_dim, rank))

        self.o_A = nn.Parameter(torch.randn(rank, cfg.n_heads * cfg.head_dim) / math.sqrt(cfg.n_heads * cfg.head_dim))
        self.o_B = nn.Parameter(torch.zeros(H, rank))

        self.gate_A = nn.Parameter(torch.randn(rank, H) / math.sqrt(H))
        self.gate_B = nn.Parameter(torch.zeros(I, rank))

        self.up_A = nn.Parameter(torch.randn(rank, H) / math.sqrt(H))
        self.up_B = nn.Parameter(torch.zeros(I, rank))

        self.down_A = nn.Parameter(torch.randn(rank, I) / math.sqrt(I))
        self.down_B = nn.Parameter(torch.zeros(H, rank))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)
        if cfg.cond_mode == "lora_cycle":
            self.lora_branches = nn.ModuleList(
                LoRALayerAdapter(cfg, cfg.cond_lora_rank) for _ in range(cfg.cond_lora_branches)
            )
        else:
            self.lora_branches = None

    def forward(self, x, cos, sin, kv_source=None, branch_idx: int | None = None, kv_extra=None, bucket=None):
        # kv_source is normed by THIS layer's norm1, matching how it would have been normed on the
        # pass that produced it -- norm1 is the same module either way, so a depth-k cache entry is
        # the same tensor it would have been at depth k.
        kv = None if kv_source is None else self.norm1(kv_source)
        lora = self.lora_branches[branch_idx % len(self.lora_branches)] if (self.lora_branches is not None and branch_idx is not None) else None
        xe = None if not kv_extra else [self.norm1(e) for e in kv_extra]
        x = x + self.attn(self.norm1(x), cos, sin, kv, lora=lora, kv_extra=xe, bucket=bucket)
        x = x + self.mlp(self.norm2(x), lora=lora)
        return x


# --------------------------------------------------------------------------------- the looped model

class LoopedBlock(nn.Module):
    """`layers_per_loop` distinct DecoderLayers, applied as one unit, weights shared across loops."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.layers_per_loop))

    def forward(self, x, cos, sin, kv_sources=None, collect=None, branch_idx: int | None = None,
                kv_extra_per_layer=None, bucket=None):
        """`kv_sources`: per-layer KV inputs (list, len == layers_per_loop) for cross-depth attention.
        `collect`: if a list is passed, each layer's INPUT is appended to it -- that is what a later
        cross-depth pass needs as its kv_sources, so the two are captured and replayed by the same
        code path rather than by a second transcription."""
        for i, layer in enumerate(self.layers):
            if collect is not None:
                collect.append(x)
            x = layer(x, cos, sin, None if kv_sources is None else kv_sources[i],
                      branch_idx=branch_idx, bucket=bucket,
                      kv_extra=None if kv_extra_per_layer is None else kv_extra_per_layer[i])
        return x


class LoopedTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        self.embed = nn.Embedding(cfg.vocab_size, H)
        self.h0 = nn.Parameter(torch.zeros(H))  # learned initial state, decoupled from content
        self.block = LoopedBlock(cfg)
        # Unshared prelude/coda. Empty ModuleLists when n_prelude/n_coda are 0, so they allocate no
        # parameters and the forward path below reduces to the flat model with no branch.
        self.prelude = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.n_prelude))
        self.coda = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.n_coda))
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings)
        self.final_norm = RMSNorm(H, cfg.rms_norm_eps)  # readout norm only, applied in readout()
        self.loop_norm = (RMSNorm(H, cfg.rms_norm_eps) if cfg.state_renorm else None)  # separate
        # weight: confining the CARRIED state to a sphere is a different job from normalizing the
        # state right before the LM head, and they must not share a learnable scale.
        self.lm_head_weight = self.embed.weight  # tied

        if cfg.convex_gate and cfg.fixed_gate is None:
            # Learned convex update: h_t = (1-g_t)*h_{t-1} + g_t*block(h_in).
            #
            # Why this exists, stated as the hypothesis it tests. Every scale-control mechanism this
            # project has tried bounds ||h|| by shrinking per-step progress: state_renorm contracts
            # (§4.3), residual_scale scales the step down by 1/N, and doing nothing lets ||h|| grow
            # so the readout-visible angular step decays as 1/t (§4.3). A CONVEX combination bounds
            # ||h_t|| <= max(||h_{t-1}||, ||block(...)||) WITHOUT scaling the branch down -- the new
            # term enters at full magnitude and only its weight is reduced. That is the one
            # structural property the other three lack, and it is the minimal two-term version of
            # the depth-mixing / hyper-connection family (softmax attention over ALL past loop
            # states), which costs 64 stored states and risks the routing collapse reported in
            # arXiv 2606.22325 (top source 0.643 vs 0.245 uniform, piling onto two hubs).
            #
            # g is generated from the loop index via a sinusoidal encoding + MLP -- a FUNCTION of t,
            # not a table over t, so it stays defined outside the trained loop range (§3.4's rule)
            # -- and from the current state, so the gate can be state-dependent.
            # Also note this is exactly the damped forward-Euler sub-step of arXiv 2605.23872, with
            # the damping LEARNED rather than fixed at 1/N, which is the only difference from
            # `residual_scale`. If the ceiling does not move here, the depth-mixing family is
            # unlikely to be rescued by adding more sources to choose from.
            self.gate_mlp = nn.Sequential(nn.Linear(64 + H, 128), nn.SiLU(), nn.Linear(128, 1))
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.constant_(self.gate_mlp[-1].bias, 2.0)   # sigmoid(2)=0.88: starts near the
            # ungated model, so the gate must LEARN to damp rather than starting damped.
        else:
            self.gate_mlp = None
        if cfg.fixed_gate is not None:
            assert cfg.convex_gate, "fixed_gate requires convex_gate=True (it sets g, not whether to gate)"

        # Zero-init: the clock is off at step 0 and the model is bit-identical to scale_clock=False
        # until the optimizer chooses otherwise. A strictly-larger hypothesis class, not a bet.
        self.clock_w = nn.Parameter(torch.zeros(H)) if cfg.scale_clock else None

        if cfg.inject_mode == "concat":
            self.adapter = nn.Linear(2 * H, H, bias=False)
        else:
            self.adapter = None

        if cfg.inject_mode == "gated":
            # Diagonal state-space write (see _inject). Two per-channel vectors = 2*H = 896 params.
            #   softplus(inj_b) = 1  <=  inj_b = log(e - 1) = 0.5413
            #   alpha = exp(-delta * exp(inj_a))  =>  inj_a = log(-log(alpha)) at delta = 1
            #
            # THE INIT IS NOT COSMETIC, AND THE FIRST SCREEN OF THIS ARM FAILED ON IT. The obvious
            # choice was to start AT the additive model (alpha -> 1, inj_a = -12) so the arm would be
            # a strictly-larger hypothesis class. Measured consequence: d(alpha)/d(inj_a) = 6.1e-06
            # there, against d(delta)/d(inj_b) = 0.63 -- delta is **~103,000x more reachable**. After
            # 2.5M tokens delta had moved 1.0 -> 1.1137 while alpha moved < 5e-6 and 0/448 channels
            # fell below 0.99. That run did NOT show the model declining the decay; it showed alpha
            # was untrainable from that point, so the mechanism was never exercised.
            #
            # The trade is unavoidable: any smooth map into (0,1) is flat where it approaches 1, so
            # "decay off at init" and "decay reachable" cannot both hold. **The decay has to be
            # initialised ON.** At the default alpha = 0.874 the gradient is only ~5x below delta's
            # and the model can move alpha in EITHER direction -- which is what lets the run express
            # a preference, rather than only being able to crawl one way.
            #   alpha=0.982 -> 35x worse gradient   alpha=0.874 -> 5x   alpha=0.692 -> 2x
            # Cost: the arm no longer starts at the control, so this is a different model at init and
            # the comparison is additive-vs-decayed rather than a superset test. Stated, not hidden.
            a0 = math.log(-math.log(cfg.gate_alpha_init))
            self.inj_a = nn.Parameter(torch.full((H,), a0))
            self.inj_b = nn.Parameter(torch.full((H,), math.log(math.e - 1.0)))
        else:
            self.inj_a = self.inj_b = None

        if cfg.depth_gate_mode in ("state", "state_norm"):
            # Learned state-dependent depth gating (AttnRes / Depth-Mixture family)
            # Scores each loop state h_t and forms a soft convex mixture before readout.
            # Zero-initialized weights guarantee exact uniform weighting (1/n_loops) at step 0.
            #
            # "state"      -- logits = w . h_t on the RAW state. MEASURED BROKEN (§4.22): ||h_t|| grows
            #                 1.8-4.0x within a forward pass and ~1e3 over training, so the softmax
            #                 temperature is effectively zero and the gate saturates onto ONE loop
            #                 (effective loops mixed = 1.01-1.05 of r, 95-98% of tokens at top-w>0.99).
            #                 It is a hard selector, not a mixture, and cannot express the hypothesis.
            # "state_norm" -- logits = tau * (w . h_t/||h_t||). Scale-invariant, exactly as the readout
            #                 is (RMSNorm before the tied head). `tau` is a single learned scalar,
            #                 init 0 => temperature 1, so the model CHOOSES its own sharpness instead of
            #                 having it forced to infinity by the state norm. +1 param over "state".
            #
            # PREDICTION, written before running (§4.7c's null is a LOWER bound on this, not an upper
            # one -- a global weighting cannot reach a per-token signal, a learned per-token gate can):
            # if the per-token depth headroom (0.2008-0.2032 nats, split-half 0.866) is reachable at
            # all, this is the instrument that reaches it. Falsifier: effective-loops-mixed stays ~1.0
            # (it saturated anyway, and scale was not the binding constraint), or stays ~r with no CE
            # gain (it declined to discriminate and this reduces to §4.7c's static-mixture null).
            self.depth_gate_head = nn.Linear(H, 1, bias=False)
            nn.init.zeros_(self.depth_gate_head.weight)
            self.depth_gate_logtau = (nn.Parameter(torch.zeros(1))
                                      if cfg.depth_gate_mode == "state_norm" else None)
        else:
            self.depth_gate_head = None
            self.depth_gate_logtau = None

        self.apply(self._init_weights)
        # Re-zero the clock, gate, depth_gate, and LoRA B weights if present, ensuring exact init semantics
        if self.clock_w is not None:
            nn.init.zeros_(self.clock_w)
        if self.depth_gate_head is not None:
            nn.init.zeros_(self.depth_gate_head.weight)
        if self.gate_mlp is not None:
            nn.init.zeros_(self.gate_mlp[-1].weight)
            nn.init.constant_(self.gate_mlp[-1].bias, 2.0)
        if cfg.cond_mode == "lora_cycle":
            for layer in self.block.layers:
                if layer.lora_branches is not None:
                    for b in layer.lora_branches:
                        nn.init.zeros_(b.q_B)
                        nn.init.zeros_(b.k_B)
                        nn.init.zeros_(b.v_B)
                        nn.init.zeros_(b.o_B)
                        nn.init.zeros_(b.gate_B)
                        nn.init.zeros_(b.up_B)
                        nn.init.zeros_(b.down_B)

        if cfg.residual_scale is not None:
            # mutually exclusive with depth_init: both rescale the same two matrices, and stacking
            # them would silently apply 1/sqrt(2*n_loop_eff) * lambda/(N*sqrt(L)) -- a constant
            # nobody chose. Asserted rather than silently preferred.
            assert not cfg.depth_init, ("residual_scale replaces depth_init; enabling both would "
                                        "compose two different prescriptions for the same scaling")
            self._apply_residual_scale()
        elif cfg.depth_init:
            self._apply_depth_init()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = math.sqrt(2.0 / (5.0 * self.cfg.hidden_size))  # Huginn-style width-aware std
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _apply_residual_scale(self):
        """eps = lambda / (N * sqrt(L)) applied to every residual BRANCH OUTPUT, replacing depth_init.

        Two differences from `depth_init`, both deliberate. depth_init scales o_proj/down_proj at
        INIT only, by 1/sqrt(2*n_loop_eff) -- a one-time change the optimizer immediately starts
        undoing. This is a persistent multiplier on the branch output, so the residual stream's
        per-step increment stays O(eps) throughout training. And the constant is right: N is the
        loop count and L the layers per loop, so eps*N = O(1/sqrt(L)) bounds the TOTAL displacement
        from h0 regardless of N, which is the property that makes the optimal LR transfer across
        loop counts (the reported reason to prefer 1/N over 1/sqrt(N)).

        Recorded objection, because it is not obviously a win here: eps*N = O(1) bounding total
        displacement is *also* this project's dilution problem (§4.3) arriving by a different road --
        if the state can only move a bounded distance no matter how many loops it takes, the angular
        budget is bounded by construction. §4.6 measured that the achievable CE is invariant to how
        fast that budget is spent. So this may reproduce the same ceiling with a cleaner LR story
        rather than lifting it. That is the experiment.

        Implemented as a forward-time multiplier registered on the branch modules, so it is exactly
        `x + eps*branch(norm(x))` with no change to the branch itself.
        """
        lam = self.cfg.residual_scale
        N, L = self.cfg.n_loop_eff, self.cfg.layers_per_loop
        eps = lam / (N * math.sqrt(L))
        for layer in list(self.block.layers) + list(self.prelude) + list(self.coda):
            layer.attn.o_proj.weight.data.mul_(eps)
            layer.mlp.down.weight.data.mul_(eps)
        self._residual_eps = eps

    def _apply_depth_init(self):
        scale = 1.0 / math.sqrt(2.0 * self.cfg.n_loop_eff)
        for layer in self.block.layers:
            layer.attn.o_proj.weight.data.mul_(scale)
            layer.mlp.down.weight.data.mul_(scale)
        # Prelude/coda run ONCE, not n_loop_eff times, so they get the ordinary 1/sqrt(2*depth)
        # scaling for their own depth rather than the loop's -- scaling them by the loop's factor
        # would shrink a once-applied layer as if it were composed 24 times with itself.
        n_once = len(self.prelude) + len(self.coda)
        if n_once:
            once_scale = 1.0 / math.sqrt(2.0 * n_once)
            for layer in list(self.prelude) + list(self.coda):
                layer.attn.o_proj.weight.data.mul_(once_scale)
                layer.mlp.down.weight.data.mul_(once_scale)

    def _loop_pos_enc(self, t: int, device, dtype):
        """Sinusoidal encoding of the loop index. A function of t, defined for every t, so a model
        trained to 32 loops can still be evaluated at 128 (a table over t could not)."""
        half = 32
        freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32)
                          * -(math.log(10000.0) / half))
        a = torch.tensor(float(t), device=device) * freqs
        return torch.cat([a.sin(), a.cos()]).to(dtype)

    def _gate(self, h_prev, h_new, t):
        if self.cfg.fixed_gate is not None:
            g = self.cfg.fixed_gate
            return (1.0 - g) * h_prev + g * h_new, g
        pe = self._loop_pos_enc(t, h_new.device, h_new.dtype)
        pe = pe.expand(*h_new.shape[:-1], pe.shape[-1])
        g = torch.sigmoid(self.gate_mlp(torch.cat([pe, h_new], dim=-1)))
        return (1.0 - g) * h_prev + g * h_new, g

    def _clock(self, h_in, h, log_rms0):
        """Multiply the block's input by `1 + w * (log rms(h_t) - log rms(h_1))`, per token.

        THE MOTIVATION, AND IT IS NOT THE ONE THIS STARTED AS. The original proposal was to break a
        fixed point of the induced sphere map `G(u) = F(u)/||F(u)||`: RMSNorm is scale-invariant and
        ||e||/||h|| ~ 1e-4, so the block's input is a function of DIRECTION alone, and if `u` settles
        the loop computes a constant forever. **That premise was tested and is false**
        (`src/angular_convergence.py`): `u` does not converge, it drifts logarithmically -- aligned
        steps of size C/t summing to C*ln(T/t), which diverges, R2 0.986 against a power law's 0.748.
        There is no fixed point here to break, and an intervention sold as breaking one is aimed at
        nothing.

        What survives the correction is a weaker and more honest motivation. Logarithmic drift means
        the READOUT-VISIBLE progress per loop vanishes as 1/t even though the trajectory never
        settles -- that is §4.3's dilution, restated. The block cannot see ||h||, so it has no way to
        know how deep it is or to behave differently late than early; every loop is handed a
        direction and nothing else. This gives it the one coordinate normalization throws away.

        Precedent that a TRAINING-TIME scale intervention can move the learned path where an
        inference-time one cannot: §4.6's radial clamp relocates the optimum without raising the
        ceiling, while §4.6b's norm penalty -- trained -- is the only arm whose rho crosses below 1
        and whose drift constant falls (C 0.154 -> 0.102). This is a trained scale coupling, so it is
        in the second class, not the first.

        Cost: `hidden_size` parameters (448, 0.005% of the block), zero extra FLOPs, and `w` is
        ZERO-INITIALISED so the model is bit-identical to the current one at step 0 and must learn to
        use the clock rather than starting coupled. `s` is referenced to the token's OWN loop-1 scale
        rather than a batch statistic, so it is per-token, batch-independent, and 0 at t=1.

        §3.4 compliance: this is a FUNCTION of the state, not a table over `t`. It is defined at every
        loop count, extrapolates past the trained range, and has nothing to outgrow -- unlike an
        iteration embedding or a per-iteration norm table, which §3.4 rejects for that reason."""
        s = torch.log(h.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)) - log_rms0
        return h_in * (1.0 + self.clock_w * s.to(h_in.dtype))

    def _inject(self, h, e):
        """How the input re-enters at every loop t > 0. (At t = 0 the state is `h0 + e`
        unconditionally -- see forward(); none of these modes can change that.)

        `gated` is the diagonal state-space write used by Parcae and by *Looped Transformers Done
        Right*, and it is the cell §4.1's ablation never tested:

            delta = softplus(inj_b)                    learned per-channel WRITE strength
            alpha = exp(-delta * exp(inj_a))           learned per-channel CARRY decay, in (0,1)
            h_in  = alpha * h + delta * e

        WHY IT MATTERS HERE, AND IT IS NOT A FOURTH ARBITRARY OPTION. §4.1 swept the normalisation
        axis as {hard RMSNorm between loops (`state_renorm=True`), nothing (`False`)} and found the
        second worth -0.744 nats -- the largest effect in the project. Both reference implementations
        choose NEITHER: a soft per-channel decay, which bounds the state WITHOUT projecting it onto a
        sphere. That is the missing third option, and it bears directly on this project's own
        findings:

          * §4.3 measures ||e||/||h|| = 1.3e-3 falling to 7e-5 -- the re-injected input is drowned.
            That is a CONSEQUENCE OF PLAIN ADDITION: `e` has fixed magnitude, ||h|| grows without
            bound, and nothing in `h + e` controls the ratio. Under this form alpha < 1 bounds the
            state at delta*||e||/(1-alpha), so the write ratio is a learned quantity instead of an
            accident of how far the state has drifted.
          * §4.3 also shows the state does not converge -- it drifts logarithmically. A carry decay
            is the one mechanism on this axis that can stop that without the contraction-to-inertness
            that `state_renorm=True` produces (§4.3).

        Parameter cost is 2*H = 896 (0.0099%). A projection `W_in` on the write (as in the reference
        implementations) would add 200,704 more; it is deliberately NOT included, so that any effect
        measured here is the DECAY MECHANISM rather than 200k extra parameters. The projected variant
        is untested and is stated as such.
        """
        if self.cfg.inject_mode == "none":
            return h
        if self.cfg.inject_mode == "additive":
            return h + e
        if self.cfg.inject_mode == "gated":
            delta = F.softplus(self.inj_b)
            alpha = torch.exp(-delta * torch.exp(self.inj_a))
            return alpha * h + delta * e
        return self.adapter(torch.cat([h, e], dim=-1))  # concat

    def readout(self, h, cos=None, sin=None, is_final: bool = True):
        """`readout_mode` implements three of the four scale-control interventions in
        arXiv 2606.24898 Table 1 (the fourth, inter-loop normalization, is `state_renorm`):
          "norm"       -- RMSNorm before the tied head. Scale-INVISIBLE to CE (their Lemma 1:
                          <grad_H L, H> = 0), which is the blind spot itself.
          "raw"        -- no final_norm. Scale becomes visible to CE by breaking scale invariance.
          "final_only" -- raw at intermediate loops, normed at the final loop: exposes scale through
                          the intermediate exits while keeping a normalized final interface.
        `is_final` only matters for "final_only". Default "norm" reproduces the previous readout
        exactly (check [9]).

        The coda runs HERE, not once after the loop, because every loop is read out and each
        readout is a real exit point -- a coda applied only after the last loop would leave every
        intermediate exit un-decoded, which is the opposite of what the sandwich topology is for.
        Cost: with n_coda>0 a dense every-loop eval runs the coda n_loops times. cos/sin are required
        whenever n_coda>0 (the coda contains attention); they stay optional so that the n_coda=0 path,
        and any existing caller, is untouched."""
        if self.coda:
            if cos is None:
                raise ValueError("readout() needs cos/sin when n_coda > 0 (the coda has attention)")
            for layer in self.coda:
                h = layer(h, cos, sin)
        m = self.cfg.readout_mode
        normed = m == "norm" or (m == "final_only" and is_final)
        return F.linear(self.final_norm(h) if normed else h, self.lm_head_weight)

    def forward(self, input_ids: torch.Tensor, n_loops: int, return_all_loops: bool = True,
                h0_noise: float = 0.0, supervise_idx=None, return_states: bool = False,
                return_state_rms: bool = False):
        """Returns (logits_per_loop: list[Tensor|None] len n_loops, state_norms: list[float]), or
        (..., states) when `return_states` -- the raw per-loop hidden states, detached. That third
        return exists so state_dynamics.py can measure the loop map's behaviour using THIS loop
        rather than a fourth transcription of it (eval.py already carries one; see the exact-identity
        check in state_dynamics.py that pins the two together).
        `logits_per_loop[-1]` is always populated. `supervise_idx`, if given, restricts readout to
        exactly those loop indices (all others are None) -- this is not just an optimisation: a
        [B,T,V] float32 logits tensor at every one of up to 32 loops is real memory (measured: this
        is what pushed a single forward over an 8GB self-imposed cap, see LOG.md), and holding 27
        unused ones for every 5 actually supervised was pure waste. `return_all_loops=True` with
        `supervise_idx=None` keeps the old dense behaviour, used by eval.py where every loop's
        logits genuinely are wanted (inference-only, no backward graph to worry about).

        `h0_noise`: adds N(0, h0_noise) to the initial state for this call only, nothing else about
        the model is touched. Exists so eval.py can run a clean/perturbed pair from the same batch
        and measure how fast the two trajectories reconverge -- an online contraction-rate estimate
        that needs no separate instrument, since h0 is a deterministic learned parameter here (unlike
        Huginn's unseeded h0 draw, this architecture has no natural source of trajectory divergence
        to measure against otherwise)."""
        B, T = input_ids.shape
        e = self.embed(input_ids)
        cos, sin = self.rope(T, input_ids.device, e.dtype)
        for layer in self.prelude:      # unshared, runs once; empty ModuleList when n_prelude=0
            e = layer(e, cos, sin)
        # NB: `e` is rebound to the prelude OUTPUT, so what gets re-injected at every loop is the
        # encoded input, not the raw embedding. That is the point of a prelude -- injecting the raw
        # table lookup at every step would make the prelude a once-only detour the loop never sees.
        # With n_prelude=0 this is a no-op and `e` is the embedding exactly as before.

        h0 = self.h0
        if h0_noise > 0:
            h0 = h0 + h0_noise * torch.randn_like(h0)
        h = h0.expand(B, T, -1) + e  # first injection always happens, regardless of inject_mode
        logits_per_loop, state_norms, states, state_rms = [], [], [], []
        k = self.cfg.truncate_bptt

        log_rms0 = None
        all_h_states = []
        W = max(1, int(getattr(self.cfg, 'kv_window', 1)))
        kv_hist = []  # duo-causal: last W-1 loops' per-layer INPUTS, oldest first
        for t in range(n_loops):
            no_grad = k is not None and t < (n_loops - k)
            # nullcontext, NOT torch.enable_grad(): enable_grad() overrides an OUTER no_grad(), so
            # the original `else torch.enable_grad()` silently re-enabled autograd inside every
            # @torch.no_grad() caller whenever truncate_bptt is None -- i.e. for the default and for
            # the winning config. Every eval in this project was therefore building a full graph
            # across all n_loops (this is what forced eval batch_size down to 4 on Kaggle and
            # motivated the 14GB MPS guard in eval.py; see LOG.md). During training grad is already
            # enabled by the caller, so nullcontext is exactly equivalent there -- forward VALUES
            # are unchanged either way, which test_model.py check [3] pins at max|diff|=0.
            ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
            with ctx:
                h_in = self._inject(h, e) if t > 0 else h
                if self.clock_w is not None:
                    if log_rms0 is None:
                        log_rms0 = torch.log(
                            h.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-8)).detach()
                    h_in = self._clock(h_in, h, log_rms0)
                branch_idx = t if self.cfg.cond_mode == "lora_cycle" else None
                cur_inputs = [] if W > 1 else None
                extras = ([[past[i] for past in kv_hist] for i in range(len(self.block.layers))]
                          if (W > 1 and kv_hist) else None)
                h_new = self.block(h_in, cos, sin, collect=cur_inputs, branch_idx=branch_idx,
                                   bucket=t,
                                   kv_extra_per_layer=extras)
                if W > 1:
                    kv_hist.append(cur_inputs)
                    if len(kv_hist) > W - 1:
                        kv_hist.pop(0)
                if self.gate_mlp is not None or self.cfg.fixed_gate is not None:
                    h_new, _g = self._gate(h, h_new, t)
                h = h_new
                if self.loop_norm is not None:
                    h = self.loop_norm(h)
            if self.depth_gate_head is not None:
                all_h_states.append(h)
            if self.cfg.explore_noise > 0.0 and self.training:
                # STOCHASTIC EXPLORATION DURING LOOPS -- the task names this directly ("exploration
                # во время лупов") and EBT, one of its three cited exemplars, does exactly this:
                # its loop is Langevin descent, y_{i+1} = y_i - a*grad E + eta_i.
                #
                # Motivated here by this project's own measurement rather than by analogy. §4.3
                # found consecutive increments aligned at cos(du_t, du_{t-1}) -> 0.9999: the state
                # travels an almost perfectly straight ray. That is maximal coherence, i.e. ZERO
                # exploration -- every loop pushes in the direction the last one did. If depth is
                # wasted because the trajectory commits early and never deviates, breaking that
                # coherence is the intervention the geometry actually suggests.
                #
                # Noise is scaled RELATIVE to ||h|| per token, because §4.3 also showed the readout
                # sees only direction (final_norm is scale-invariant): absolute noise on a state
                # whose norm grows 18x would vanish from the readout's view exactly when depth
                # matters most. Annealed as 1/sqrt(t) by default, the Langevin convention.
                # Train-time only -- at eval the map stays deterministic, so every number in this
                # report remains comparable.
                sig = self.cfg.explore_noise / math.sqrt(t + 1) if self.cfg.explore_anneal \
                    else self.cfg.explore_noise
                rms = h.detach().float().pow(2).mean(-1, keepdim=True).sqrt()
                h = h + (sig * rms * torch.randn_like(h.float())).to(h.dtype)
            state_norms.append(h.detach().float().norm(dim=-1).mean().item())
            if return_states:
                states.append(h.detach())
            if return_state_rms:
                # DIFFERENTIABLE per-loop mean RMS scale -- one scalar per loop, so the norm penalty
                # costs no meaningful memory (unlike retaining states). E||H_k||^2_rms in the
                # notation of arXiv 2606.24898's norm-penalty intervention.
                state_rms.append(h.float().pow(2).mean(-1).sqrt().mean())
            wanted = (t == n_loops - 1) or (supervise_idx is not None and t in supervise_idx) or \
                     (supervise_idx is None and return_all_loops)
            logits_per_loop.append(
                self.readout(h, cos, sin, is_final=(t == n_loops - 1)) if wanted else None)

        if self.depth_gate_head is not None and len(all_h_states) == n_loops:
            H_stack = torch.stack(all_h_states, dim=2)  # [B, T, n_loops, H]
            if self.depth_gate_logtau is not None:
                # SCALE-INVARIANT gate: score the DIRECTION, not the raw state. The readout is
                # scale-invariant by construction and this makes the gate match it (§4.22).
                gate_in = F.normalize(H_stack.float(), dim=-1).to(H_stack.dtype)
                gate_logits = (self.depth_gate_head(gate_in).squeeze(-1)
                               * self.depth_gate_logtau.exp())
            else:
                gate_logits = self.depth_gate_head(H_stack).squeeze(-1)  # [B, T, n_loops]
            gate_weights = F.softmax(gate_logits, dim=-1)           # [B, T, n_loops]
            h_weighted = (H_stack * gate_weights.unsqueeze(-1)).sum(dim=2)  # [B, T, H]
            logits_per_loop[-1] = self.readout(h_weighted, cos, sin, is_final=True)

        if return_states and return_state_rms:
            return logits_per_loop, state_norms, states, state_rms
        if return_state_rms:
            return logits_per_loop, state_norms, state_rms
        if return_states:
            return logits_per_loop, state_norms, states
        return logits_per_loop, state_norms

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

# === TRAINING ROUTINES ===

def train_tokenizer():
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from datasets import load_dataset
    log("loading FineWeb sample for tokenizer...")
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    def text_iterator():
        for i, item in enumerate(ds):
            if i >= 5000: break
            yield item["text"]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=4096, special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"], show_progress=False)
    tok.train_from_iterator(text_iterator(), trainer=trainer)
    log("tokenizer trained, vocab_size=4096")
    def stream():
        for item in ds: yield item["text"]
    return tok, stream()


def pack_from_stream(stream_iter, tok, n_train_tokens=10_000_000, n_val_tokens=3_000_000):
    log(f"packing tokens: {n_train_tokens} train, {n_val_tokens} val...")
    buf = []
    for text in stream_iter:
        ids = tok.encode(text).ids
        buf.extend(ids)
        if len(buf) >= n_train_tokens + n_val_tokens:
            break
    arr = np.array(buf, dtype=np.uint16)
    train_shard = arr[:n_train_tokens]
    val_shard = arr[n_train_tokens:n_train_tokens+n_val_tokens]
    log(f"tokens packed: train {len(train_shard)}, val {len(val_shard)}")
    return train_shard, val_shard


def get_batch(data, batch_size, seq_len, device, rng):
    max_idx = len(data) - seq_len - 1
    ix = rng.integers(0, max_idx, size=(batch_size,))
    x = np.stack([data[i:i+seq_len].astype(np.int64) for i in ix])
    y = np.stack([data[i+1:i+1+seq_len].astype(np.int64) for i in ix])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def lr_at(step, total_steps, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + math.cos(math.pi * progress))


def sample_supervise_idx(n_loops, k, rng):
    if k == 1: return [n_loops - 1]
    if k >= n_loops: return list(range(n_loops))
    intermediate = list(rng.choice(n_loops - 1, size=k - 1, replace=False))
    return sorted(intermediate + [n_loops - 1])


def effective_k(step, total_steps, cfg):
    if cfg.supervise_k_final is None or cfg.supervise_switch_frac is None:
        return cfg.supervise_k
    frac = step / max(1, total_steps)
    return cfg.supervise_k if frac < cfg.supervise_switch_frac else cfg.supervise_k_final


def per_loop_ce(logits_per_loop, y, supervise_idx):
    losses = []
    y_flat = y.reshape(-1)
    for idx in supervise_idx:
        lg = logits_per_loop[idx].reshape(-1, logits_per_loop[idx].size(-1))
        losses.append(F.cross_entropy(lg, y_flat))
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, val_data, train_cfg, rng):
    model.eval()
    losses = {r: [] for r in train_cfg.eval_loop_sweep}
    n_eval_seqs = 64
    eval_bs = 4
    for b in range(0, n_eval_seqs, eval_bs):
        x, y = get_batch(val_data, eval_bs, train_cfg.seq_len, train_cfg.device, rng)
        y_flat = y.reshape(-1)
        for r in train_cfg.eval_loop_sweep:
            lg, _ = model(x, n_loops=r, supervise_idx={r - 1})
            loss = F.cross_entropy(lg[r - 1].reshape(-1, lg[r - 1].size(-1)), y_flat)
            losses[r].append(loss.item())
    model.train()
    return {r: float(np.mean(losses[r])) for r in train_cfg.eval_loop_sweep}


@dataclasses.dataclass
class TrainConfig:
    run_name: str = "run"
    batch_size: int = 8
    seq_len: int = 256
    total_tokens: int = 2_500_000
    lr: float = 3e-3
    min_lr: float = 7.5e-5
    warmup_tokens: int = 250_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    min_train_loops: int = 4
    max_train_loops: int = 32
    supervise_k: int = 5
    supervise_k_final: int | None = None
    supervise_switch_frac: float | None = None
    fixed_train_loops: int | None = None
    norm_penalty: float = 0.0
    eval_every_tokens: int = 500_000
    eval_loop_sweep: tuple = (1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64)
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    warmup_steps: int = 0


def run_arm(model_cfg, train_cfg, train_shard, val_shard, results):
    log(f"=== LAUNCHING {train_cfg.run_name} ===")
    arm_t0 = time.time()
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    rng = np.random.default_rng(train_cfg.seed)

    model = LoopedTransformer(model_cfg).to(train_cfg.device)
    log(f"Model parameters: {model.num_parameters():,}")

    tokens_per_step = train_cfg.batch_size * train_cfg.seq_len
    total_steps = train_cfg.total_tokens // tokens_per_step
    train_cfg.warmup_steps = train_cfg.warmup_tokens // tokens_per_step
    eval_every = max(1, train_cfg.eval_every_tokens // tokens_per_step)

    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": train_cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ], lr=train_cfg.lr, betas=(0.9, 0.95))

    arm_history = []
    model.train()

    for step in range(total_steps):
        lr = lr_at(step, total_steps, train_cfg)
        for g in opt.param_groups: g["lr"] = lr
        x, y = get_batch(train_shard, train_cfg.batch_size, train_cfg.seq_len, train_cfg.device, rng)
        n_loops = int(rng.integers(train_cfg.min_train_loops, train_cfg.max_train_loops + 1))
        sup_idx = sample_supervise_idx(n_loops, effective_k(step, total_steps, train_cfg), rng)
        logits_per_loop, state_norms = model(x, n_loops=n_loops, supervise_idx=set(sup_idx))
        loss = per_loop_ce(logits_per_loop, y, sup_idx)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()

        if step % 50 == 0 or step == total_steps - 1:
            tok_s = tokens_per_step * (step + 1) / max(1e-3, time.time() - arm_t0)
            log(f"  step {step}/{total_steps} loss={loss.item():.4f} n_loops={n_loops} gnorm={gnorm:.2f} {tok_s:.0f} tok/s")

        if (step % eval_every == 0 and step > 0) or step == total_steps - 1:
            curve = evaluate(model, val_shard, train_cfg, rng)
            log(f"  EVAL step {step}: " + " ".join(f"r{r}={v:.4f}" for r, v in curve.items()))
            arm_history.append(dict(step=step, val_curve=curve))
            results[train_cfg.run_name] = dict(
                model_cfg=dataclasses.asdict(model_cfg),
                train_cfg=dataclasses.asdict(train_cfg),
                history=arm_history,
                params=model.num_parameters(),
                elapsed_s=time.time() - arm_t0
            )
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)
            ckpt_path = f"{train_cfg.run_name}_last.pt"
            torch.save(dict(model=model.state_dict(), model_cfg=dataclasses.asdict(model_cfg),
                            train_cfg=dataclasses.asdict(train_cfg), step=step), ckpt_path)

    log(f"=== ARM {train_cfg.run_name} finished in {time.time()-arm_t0:.1f}s ===")
    return results


def main():
    log(f"CUDA available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    if not torch.cuda.is_available():
        log("ERROR: CUDA not available on GPU node, exiting to avoid wasted compute")
        sys.exit(1)

    tok, stream_it = train_tokenizer()
    train_shard, val_shard = pack_from_stream(stream_it, tok, n_train_tokens=10_000_000, n_val_tokens=3_000_000)

    SWEEP = tuple(sorted({1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64}))
    M25 = 2_500_000

    def make_tcfg(name):
        return TrainConfig(run_name=name, batch_size=8, total_tokens=M25, eval_every_tokens=500_000,
                           supervise_k=5, min_train_loops=4, max_train_loops=32, seed=0, eval_loop_sweep=SWEEP)

    # sec4.28's causal test. dg_norm showed a WORKING depth mixture gains nothing over a depth-key
    # stream of rank ~1.6. If that rank is the binding constraint, raising it should let the same
    # gate act. W_K in 4 loop-index buckets raises measured key rank 1.603 -> 5.742 (+903,168 params,
    # 9,967,776 total, under the 10M cap). Three arms, in-job, so the drift measured at sec4.27
    # (0.0914 cross-job) cannot touch the comparison.
    # sec4.31 measured the depth-key rank collapse as width-INDEPENDENT at initialisation across a
    # 12x parameter range. sec4.30 measured what TRAINING does to it (8.818 -> 1.74 with four
    # projections) at ONE width. This trains the same tied control at three widths so the trained
    # collapse can be compared across them. Head_dim held near-constant so width is what varies.
    arms = [
        ("wd_224_s0", Config(state_renorm=False, hidden_size=224, n_heads=2, n_kv_heads=1,
                             head_dim=112, intermediate_size=672), make_tcfg("wd_224_s0")),
        ("wd_448_s0", Config(state_renorm=False), make_tcfg("wd_448_s0")),
        ("wd_896_s0", Config(state_renorm=False, hidden_size=896, n_heads=8, n_kv_heads=4,
                             head_dim=112, intermediate_size=2688), make_tcfg("wd_896_s0")),
    ]

    results = {}
    results = {}
    for name, mcfg, tcfg in arms:
        if time.time() - T0 > MAX_SWEEP_SECONDS:
            log(f"Time cutoff reached before {name}, stopping")
            break
        results = run_arm(mcfg, tcfg, train_shard, val_shard, results)

    log("=== DUO-CAUSAL SWEEP COMPLETED ===")
    for name, r in results.items():
        if r["history"]:
            last = r["history"][-1]["val_curve"]
            best_r = min(last, key=last.get)
            log(f"  {name:<18} best: r={best_r:>2} CE={last[best_r]:.4f} r1={last[1]:.4f}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"ALL DONE in {time.time()-T0:.1f}s")


if __name__ == "__main__":
    main()
