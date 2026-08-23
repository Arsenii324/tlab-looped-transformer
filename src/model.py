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

from __future__ import annotations

import contextlib
import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    inject_mode: str = "additive"          # "none" | "additive" | "concat"
    depth_init: bool = True
    residual_scale: float | None = None    # if set, lambda in eps = lambda/(N*sqrt(L)); see below
    n_loop_eff: int = 24                   # for depth_init scaling; ~ mean of train-time loop sampler


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
        self.n_h, self.n_kv, self.d_h = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.groups = self.n_h // self.n_kv
        self.scale = self.d_h ** -0.5
        H = cfg.hidden_size
        self.q_proj = nn.Linear(H, self.n_h * self.d_h, bias=False)
        self.k_proj = nn.Linear(H, self.n_kv * self.d_h, bias=False)
        self.v_proj = nn.Linear(H, self.n_kv * self.d_h, bias=False)
        self.o_proj = nn.Linear(self.n_h * self.d_h, H, bias=False)
        self.q_norm = RMSNorm(self.d_h, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.d_h, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, kv_source=None):
        """`kv_source`, when given, supplies keys and values while `x` still supplies queries. This
        is the one hook needed to ask what an early-exit KV cache costs: under teacher forcing every
        position is processed at every depth, so "the context exited at depth k while this token is
        computing at depth t" is exactly q from the depth-t stream against k/v from the depth-k one.
        Defaults to None = ordinary self-attention, bit-identical (test_model.py check [8])."""
        B, T, _ = x.shape
        kv = x if kv_source is None else kv_source
        q = self.q_norm(self.q_proj(x).view(B, T, self.n_h, self.d_h)).transpose(1, 2)
        k = self.k_norm(self.k_proj(kv).view(B, T, self.n_kv, self.d_h)).transpose(1, 2)
        v = self.v_proj(kv).view(B, T, self.n_kv, self.d_h).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, T, self.n_h * self.d_h)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, kv_source=None):
        # kv_source is normed by THIS layer's norm1, matching how it would have been normed on the
        # pass that produced it -- norm1 is the same module either way, so a depth-k cache entry is
        # the same tensor it would have been at depth k.
        kv = None if kv_source is None else self.norm1(kv_source)
        x = x + self.attn(self.norm1(x), cos, sin, kv)
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------------- the looped model

class LoopedBlock(nn.Module):
    """`layers_per_loop` distinct DecoderLayers, applied as one unit, weights shared across loops."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.layers_per_loop))

    def forward(self, x, cos, sin, kv_sources=None, collect=None):
        """`kv_sources`: per-layer KV inputs (list, len == layers_per_loop) for cross-depth attention.
        `collect`: if a list is passed, each layer's INPUT is appended to it -- that is what a later
        cross-depth pass needs as its kv_sources, so the two are captured and replayed by the same
        code path rather than by a second transcription."""
        for i, layer in enumerate(self.layers):
            if collect is not None:
                collect.append(x)
            x = layer(x, cos, sin, None if kv_sources is None else kv_sources[i])
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

        if cfg.inject_mode == "concat":
            self.adapter = nn.Linear(2 * H, H, bias=False)
        else:
            self.adapter = None

        self.apply(self._init_weights)
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

    def _inject(self, h, e):
        if self.cfg.inject_mode == "none":
            return h
        if self.cfg.inject_mode == "additive":
            return h + e
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
                h_new = self.block(h_in, cos, sin)
                if self.gate_mlp is not None or self.cfg.fixed_gate is not None:
                    h_new, _g = self._gate(h, h_new, t)
                h = h_new
                if self.loop_norm is not None:
                    h = self.loop_norm(h)
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

        if return_states and return_state_rms:
            return logits_per_loop, state_norms, states, state_rms
        if return_state_rms:
            return logits_per_loop, state_norms, state_rms
        if return_states:
            return logits_per_loop, state_norms, states
        return logits_per_loop, state_norms

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
