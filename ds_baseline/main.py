"""T-Lab looped-transformer: full-budget run of the screening winner, on a Kaggle T4. Self-contained
(Kaggle script kernels can't import sibling local files, learned the hard way earlier in this
workspace's history: an instrument had to be inlined after a cross-directory import broke a prior
kernel). This file is model.py + a condensed train_tokenizer.py + data.py + train.py + a driver,
mechanically kept consistent with the already-verified local source (5 correctness checks passed
locally, see LOG.md) rather than re-derived here.

HISTORY: this file started as the screening-sweep kernel (2026-08-13), never actually run that
session -- the account's weekly Kaggle GPU quota was already exhausted by unrelated prior use, so
screening ran locally on MPS instead (see LOG.md/report.md for those results and the local
throughput/memory findings). Screening is done and its winner known (no_state_renorm, by a wide
margin, report.md sec 4.1); local full-budget runs of it reached 14.60M tokens before local compute
ran out. Repurposed (2026-08-16) to push that same winning config much further using this Kaggle
GPU-hour budget instead: one arm, no_state_renorm, targeting up to 90M tokens, governed by
MAX_SWEEP_SECONDS rather than the token target (stops gracefully at the wall-clock cutoff with
whatever real progress was made, same pattern as the local chunked runs).

Writes results.json (per-loop val CE curve, swept past training range to loop 64) and a model
checkpoint (no_state_renorm_last.pt) to /kaggle/working/, pulled back via `kaggle kernels output`.
Also prints everything to stdout as it happens -- the run log is the record, not just the final JSON,
per this workspace's own standing "raw output over printed verdict" practice.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()
MAX_SWEEP_SECONDS = 10.8 * 3600  # Kaggle's hard GPU-session ceiling is 12h; this stops gracefully
# well before it. Sized from the PREVIOUS run's measured throughput rather than a guess: that run did
# 45,975,552 tokens in 5.29h = 2,414 tok/s, so total_tokens=90M needs ~10.36h. The 5.5h value this
# replaces is the entire reason the headline run stopped at 46.0M tokens (46% of the task's budget) --
# it was sized for a check-in window, not for the token target, and the token target was never the
# binding constraint. By the run's own measured scaling (0.398 nats per e-fold of tokens), finishing
# the budget is worth ~0.25-0.31 nats, which is larger than the entire measured loop gain.
if torch.backends.mps.is_available():
    torch.mps.set_per_process_memory_fraction(8.0e9 / torch.mps.recommended_max_memory())
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
RESULTS_PATH = os.path.join(OUT_DIR, "results.json")


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


# ============================================================================================
# MODEL -- verbatim from src/model.py (verified locally: matches real Qwen3DecoderLayer to
# 2.38e-07, exact-identity checks on BPTT truncation and no_grad windowing, state_renorm bounds
# norm exactly). Not re-derived here, copied.
# ============================================================================================

@dataclasses.dataclass
class Config:
    vocab_size: int = 4096
    hidden_size: int = 448
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 112
    intermediate_size: int = 1344
    layers_per_loop: int = 3
    readout_mode: str = "norm"
    n_prelude: int = 0
    n_coda: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 512
    truncate_bptt: int | None = None
    state_renorm: bool = True
    inject_mode: str = "additive"
    depth_init: bool = True
    n_loop_eff: int = 24


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta, max_pos):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_pos).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, seq_len, device, dtype):
        return (self.cos[:seq_len].to(device=device, dtype=dtype),
                self.sin[:seq_len].to(device=device, dtype=dtype))


def apply_rope(q, k, cos, sin):
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


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

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.n_h, self.d_h)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_kv, self.d_h)).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.d_h).transpose(1, 2)
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

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class LoopedBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.layers_per_loop))

    def forward(self, x, cos, sin):
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x


class UntiedStack(nn.Module):
    """N distinct decoder layers applied once each -- the compute-matched non-looped control.

    Exists to settle report.md §4.4's negative result. That section reports that a 33-layer untied
    stack could not be trained stably at all (NaN between steps 13 and 411 across six configurations),
    and uses it to argue that weight tying acts as an implicit regulariser. **Every one of those
    attempts ran on MPS, the backend this project documents as silently producing zeros and
    NaN-shaped output under sustained load (§6).** So the negative is confounded with the hardware,
    and it is the most attackable claim in the report. This runs the SAME configurations on CUDA.
    """
    def __init__(self, cfg, n_layers):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        self.embed = nn.Embedding(cfg.vocab_size, H)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(n_layers))
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings)
        self.final_norm = RMSNorm(H, cfg.rms_norm_eps)
        self.apply(self._init_weights)
        # same depth-aware init the looped model uses, keyed to the REAL depth here
        sc = 1.0 / math.sqrt(2.0 * len(self.layers))
        for l in self.layers:
            l.attn.o_proj.weight.data.mul_(sc); l.mlp.down.weight.data.mul_(sc)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, math.sqrt(2.0 / (5.0 * self.cfg.hidden_size)))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x, n_loops=None, return_all_loops=False, supervise_idx=None,
                h0_noise=0.0, return_state_rms=False):
        e = self.embed(x); h = e
        cos, sin = self.rope(x.shape[1], x.device, e.dtype)
        norms = []
        for l in self.layers:
            h = l(h, cos, sin)
            norms.append(h.detach().float().norm(dim=-1).mean().item())
        logits = F.linear(self.final_norm(h), self.embed.weight)
        return [logits], norms

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class LoopedTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        self.embed = nn.Embedding(cfg.vocab_size, H)
        self.h0 = nn.Parameter(torch.zeros(H))
        self.block = LoopedBlock(cfg)
        self.prelude = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.n_prelude))
        self.coda = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.n_coda))
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings)
        self.final_norm = RMSNorm(H, cfg.rms_norm_eps)
        self.loop_norm = (RMSNorm(H, cfg.rms_norm_eps) if cfg.state_renorm else None)
        self.lm_head_weight = self.embed.weight
        self.adapter = nn.Linear(2 * H, H, bias=False) if cfg.inject_mode == "concat" else None
        self.apply(self._init_weights)
        if cfg.depth_init:
            self._apply_depth_init()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = math.sqrt(2.0 / (5.0 * self.cfg.hidden_size))
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _apply_depth_init(self):
        scale = 1.0 / math.sqrt(2.0 * self.cfg.n_loop_eff)
        for layer in self.block.layers:
            layer.attn.o_proj.weight.data.mul_(scale)
            layer.mlp.down.weight.data.mul_(scale)
        n_once = len(self.prelude) + len(self.coda)   # run once, not n_loop_eff times
        if n_once:
            once_scale = 1.0 / math.sqrt(2.0 * n_once)
            for layer in list(self.prelude) + list(self.coda):
                layer.attn.o_proj.weight.data.mul_(once_scale)
                layer.mlp.down.weight.data.mul_(once_scale)

    def _inject(self, h, e):
        if self.cfg.inject_mode == "none":
            return h
        if self.cfg.inject_mode == "additive":
            return h + e
        return self.adapter(torch.cat([h, e], dim=-1))

    def readout(self, h, cos=None, sin=None, is_final=True):
        for layer in self.coda:          # coda runs at EVERY readout -- each loop is a real exit
            h = layer(h, cos, sin)
        m = self.cfg.readout_mode
        normed = m == "norm" or (m == "final_only" and is_final)
        return F.linear(self.final_norm(h) if normed else h, self.lm_head_weight)

    def forward(self, input_ids, n_loops, return_all_loops=True, h0_noise=0.0, supervise_idx=None,
                return_state_rms=False):
        B, T = input_ids.shape
        e = self.embed(input_ids)
        cos, sin = self.rope(T, input_ids.device, e.dtype)
        for layer in self.prelude:       # what the loop re-injects is the prelude output, not the
            e = layer(e, cos, sin)       # raw embedding (no-op when n_prelude=0)
        h0 = self.h0
        if h0_noise > 0:
            h0 = h0 + h0_noise * torch.randn_like(h0)
        h = h0.expand(B, T, -1) + e
        logits_per_loop, state_norms, state_rms = [], [], []
        k = self.cfg.truncate_bptt
        for t in range(n_loops):
            no_grad = k is not None and t < (n_loops - k)
            # nullcontext, NOT enable_grad(): enable_grad() overrides an outer no_grad(), which
            # made every eval retain a graph across all loops (report.md sec 6).
            ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
            with ctx:
                h_in = self._inject(h, e) if t > 0 else h
                h = self.block(h_in, cos, sin)
                if self.loop_norm is not None:
                    h = self.loop_norm(h)
            state_norms.append(h.detach().float().norm(dim=-1).mean().item())
            if return_state_rms:
                state_rms.append(h.float().pow(2).mean(-1).sqrt().mean())
            wanted = (t == n_loops - 1) or (supervise_idx is not None and t in supervise_idx) or                      (supervise_idx is None and return_all_loops)
            logits_per_loop.append(
                self.readout(h, cos, sin, is_final=(t == n_loops - 1)) if wanted else None)
        if return_state_rms:
            return logits_per_loop, state_norms, state_rms
        return logits_per_loop, state_norms

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================================
# DATA -- tokenizer trained fresh (deterministic given the same streamed corpus, ~15s, avoids
# needing to upload a Kaggle dataset asset), then packed to in-memory uint16 arrays.
# ============================================================================================

VOCAB_SIZE = 4096
TOKENIZER_CHAR_BUDGET = 60_000_000
SKIP_DOCS_FOR_TOKENIZER = 20_000  # rounded up from the local run's 19,319 docs / 60M chars


def train_tokenizer():
    from datasets import load_dataset
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    log("streaming FineWeb for tokenizer training...")
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    it = iter(ds)
    texts, total = [], 0
    while total < TOKENIZER_CHAR_BUDGET:
        t = next(it)["text"]
        texts.append(t)
        total += len(t)
    log(f"collected {len(texts)} docs, {total/1e6:.1f}M chars for tokenizer")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["<|endoftext|>"],
                                   show_progress=False)
    tok.train_from_iterator(texts, trainer=trainer)
    assert tok.token_to_id("<|endoftext|>") == 0
    log(f"tokenizer trained, vocab={tok.get_vocab_size()}")
    return tok, it  # return the live iterator too, so packing continues past this point


def pack_from_stream(it, tok, n_train_tokens, n_val_tokens):
    EOS = 0

    def fill(n):
        buf = np.empty(n, dtype=np.uint16)
        pos = 0
        while pos < n:
            ids = tok.encode(next(it)["text"]).ids
            ids.append(EOS)
            take = min(len(ids), n - pos)
            buf[pos:pos + take] = np.array(ids[:take], dtype=np.uint16)
            pos += take
        return buf

    log(f"packing {n_train_tokens/1e6:.1f}M train + {n_val_tokens/1e6:.1f}M val tokens...")
    train = fill(n_train_tokens)
    val = fill(n_val_tokens)
    log("packing done")
    return train, val


# ============================================================================================
# TRAIN / EVAL -- condensed from src/train.py, same logic.
# ============================================================================================

@dataclasses.dataclass
class TrainConfig:
    run_name: str = "center"
    seq_len: int = 256
    batch_size: int = 48
    lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 60
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    total_tokens: int = 6_000_000
    min_train_loops: int = 4
    max_train_loops: int = 32
    supervise_k: int = 5
    norm_penalty: float = 0.0
    fixed_train_loops: int | None = None  # if set, overrides randomization (axis 5 = off)
    eval_every_tokens: int = 1_500_000
    eval_batches: int = 8
    eval_batch_size: int = 4  # decoupled from train batch_size: eval_loop_sweep goes to 64 and
    # return_all_loops materializes every swept loop's logits at once. Was 8 -- crashed a real Kaggle
    # run (CUDA OOM, 14.55/14.56 GiB used) at its very first eval boundary (~step 976), the same
    # fragility already measured locally (LOG.md/report.md sec 4.3: batch=8 at max-loops=64 sometimes
    # tips a 14GB-class ceiling, batch=4 held reliably) -- recurring on different hardware, not a new
    # mystery. 4 is the already-proven-safe value, not a fresh guess.
    eval_loop_sweep: tuple = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(shard, batch_size, seq_len, device, rng):
    ix = rng.integers(0, len(shard) - seq_len - 1, size=batch_size)
    x = np.stack([shard[i:i + seq_len] for i in ix]).astype(np.int64)
    y = np.stack([shard[i + 1:i + seq_len + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def per_loop_ce(logits_per_loop, y, supervise_idx):
    # bounded-subset supervision -- dense all-loop supervision measured erratic 37-158s/step on MPS
    # at n_loops=16 (vs a stable ~6.5s/step for a 5-loop subset) from backward-graph fan-out; kept
    # here even for CUDA since it's strictly cheaper and the local finding transfers as a caution.
    losses = [F.cross_entropy(logits_per_loop[i].reshape(-1, logits_per_loop[i].size(-1)),
                               y.reshape(-1)) for i in supervise_idx]
    return torch.stack(losses).mean()


def sample_supervise_idx(n_loops, k, rng):
    last = n_loops - 1
    if k >= n_loops:
        return list(range(n_loops))
    rest = rng.choice(last, size=min(k - 1, last), replace=False).tolist() if last > 0 else []
    return sorted(set(rest + [last]))


def lr_at(step, total_steps, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    prog = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(prog, 1.0)))


def build_optimizer(model, cfg):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))


@torch.no_grad()
def evaluate(model, val, cfg, rng):
    model.eval()
    curves = {r: [] for r in cfg.eval_loop_sweep}
    max_r = max(cfg.eval_loop_sweep)
    for _ in range(cfg.eval_batches):
        x, y = get_batch(val, cfg.eval_batch_size, cfg.seq_len, cfg.device, rng)
        logits_per_loop, _ = model(x, n_loops=max_r, return_all_loops=True)
        for r in cfg.eval_loop_sweep:
            l = logits_per_loop[r - 1]
            curves[r].append(F.cross_entropy(l.reshape(-1, l.size(-1)), y.reshape(-1)).item())
    model.train()
    return {r: float(np.mean(v)) for r, v in curves.items()}


def run_arm(model_cfg: Config, train_cfg: TrainConfig, train_shard, val_shard, results):
    desc = ("UNTIED 33-layer stack (compute-matched non-looped control)" if model_cfg is None
            else f"state_renorm={model_cfg.state_renorm} inject_mode={model_cfg.inject_mode} "
                 f"depth_init={model_cfg.depth_init}")
    log(f"=== ARM {train_cfg.run_name} === model: {desc}  train: lr={train_cfg.lr} "
        f"grad_clip={train_cfg.grad_clip} warmup={train_cfg.warmup_steps} "
        f"tokens={train_cfg.total_tokens/1e6:.1f}M")
    torch.manual_seed(train_cfg.seed)
    rng = np.random.default_rng(train_cfg.seed)
    model = (UntiedStack(Config(), 33) if model_cfg is None
             else LoopedTransformer(model_cfg)).to(train_cfg.device)
    log(f"  params={model.num_parameters():,}")
    opt = build_optimizer(model, train_cfg)

    tokens_per_step = train_cfg.batch_size * train_cfg.seq_len
    total_steps = max(1, train_cfg.total_tokens // tokens_per_step)
    eval_every = max(1, train_cfg.eval_every_tokens // tokens_per_step)

    arm_history = []
    arm_t0 = time.time()
    for step in range(total_steps):
        if time.time() - T0 > MAX_SWEEP_SECONDS:
            log(f"  SWEEP TIME BUDGET HIT mid-arm at step {step}/{total_steps}, stopping this arm")
            break
        lr = lr_at(step, total_steps, train_cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_shard, train_cfg.batch_size, train_cfg.seq_len, train_cfg.device, rng)
        n_loops = (train_cfg.fixed_train_loops if train_cfg.fixed_train_loops is not None
                   else int(rng.integers(train_cfg.min_train_loops, train_cfg.max_train_loops + 1)))
        sup_idx = sample_supervise_idx(n_loops, train_cfg.supervise_k, rng)
        if train_cfg.norm_penalty > 0:
            logits_per_loop, state_norms, rms = model(x, n_loops=n_loops,
                                                       supervise_idx=set(sup_idx),
                                                       return_state_rms=True)
        else:
            logits_per_loop, state_norms = model(x, n_loops=n_loops, supervise_idx=set(sup_idx))
        loss = per_loop_ce(logits_per_loop, y, sup_idx)
        if train_cfg.norm_penalty > 0:
            # arXiv 2606.24898's norm-penalty intervention: lambda * K^-1 sum_k E||h_k||^2_rms.
            # Averaged over loops actually run so its scale does not depend on the sampled n_loops.
            loss = loss + train_cfg.norm_penalty * torch.stack([r.pow(2) for r in rms]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()

        if step % 50 == 0 or step == total_steps - 1:
            log(f"  step {step}/{total_steps} loss={loss.item():.4f} n_loops={n_loops} "
                f"gnorm={gnorm:.2f} state_norm=({state_norms[0]:.1f},{state_norms[-1]:.1f}) "
                f"{tokens_per_step*(step+1)/(time.time()-arm_t0):.0f} tok/s")

        if (step % eval_every == 0 and step > 0) or step == total_steps - 1:
            curve = evaluate(model, val_shard, train_cfg, rng)
            log(f"  EVAL step {step}: " + " ".join(f"r{r}={v:.4f}" for r, v in curve.items()))
            arm_history.append(dict(step=step, val_curve=curve))
            results[train_cfg.run_name] = dict(
                model_cfg=(dataclasses.asdict(model_cfg) if model_cfg is not None
                            else {'untied_stack': True, 'n_layers': 33}),
                train_cfg=dataclasses.asdict(train_cfg),
                history=arm_history, params=model.num_parameters(),
                elapsed_s=time.time() - arm_t0)
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)
            ckpt_path = os.path.join(OUT_DIR, f"{train_cfg.run_name}_last.pt")
            torch.save(dict(model=model.state_dict(),
                             model_cfg=(dataclasses.asdict(model_cfg) if model_cfg is not None
                                         else {'untied_stack': True, 'n_layers': 33}),
                             train_cfg=dataclasses.asdict(train_cfg), step=step,
                             tokens=tokens_per_step * (step + 1)), ckpt_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # defense in depth alongside the smaller eval_batch_size --
                # release cached-but-unused CUDA memory between eval calls rather than let it
                # accumulate across a run this long (same pattern as train.py's mps.empty_cache()).

    log(f"=== ARM {train_cfg.run_name} done in {time.time()-arm_t0:.0f}s ===")
    return results


def main():
    log(f"CUDA available: {torch.cuda.is_available()}, "
        f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    tok, stream_it = train_tokenizer()
    # Full-budget run of the screening winner (no_state_renorm), not the screening sweep this file
    # started as -- see the module docstring. 92M/6M matches the local runs' own train/val split for
    # comparable scale; actual tokens consumed will be governed by MAX_SWEEP_SECONDS, likely less.
    train_shard, val_shard = pack_from_stream(stream_it, tok, n_train_tokens=20_000_000,
                                               n_val_tokens=6_000_000)

    center_model = Config()
    results = {}

    # Single arm: the screening winner, at real scale. batch_size=8, matching the local full-budget
    # no_state_renorm run exactly (report.md sec 4.2) -- NOT raised for the T4. First attempt at this
    # (version 1 of this kernel) raised it to 96 on the unchecked assumption that "a T4 has more
    # memory" scales the safe batch size the same way; it OOM'd on the very first training step
    # (CUDA OOM at 14.4/14.56 GiB, full BPTT retaining up to 32 loops x 3 layers = 96 sequential
    # layers' activations at n_loops near the top of its randomized range). The actual constraint is
    # the model/batch/loop-count combination, not the hardware brand -- local batch=8 under a 14GB
    # self-imposed MPS cap already IS the verified-safe point for this exact computation, and a T4's
    # ~14.56GB ceiling is comparable, not more generous, so there was no real headroom to spend this
    # way. The T4's real advantage is FLOPS/sec (throughput), captured by eval_every_tokens instead.
    # The three LRs that NaN'd on MPS, run unchanged on CUDA. If they train here, §4.4's negative
    # is an MPS artifact and must be withdrawn; if they NaN here too, the claim gets much stronger.
    N_LAYERS = 33
    arms = [(f"baseline_cuda_lr{lr}", None,
             TrainConfig(run_name=f"baseline_cuda_lr{lr}", batch_size=8, total_tokens=6_000_000,
                          lr=lr, min_lr=lr/10, warmup_steps=300, grad_clip=0.5,
                          eval_every_tokens=1_000_000, min_train_loops=1, max_train_loops=1,
                          supervise_k=1, eval_loop_sweep=(1,)))
            for lr in (3e-3, 1e-3, 5e-4)]

    for name, mcfg, tcfg in arms:
        if time.time() - T0 > MAX_SWEEP_SECONDS:
            log(f"SWEEP TIME BUDGET HIT before arm {name}, stopping sweep here")
            break
        results = run_arm(mcfg, tcfg, train_shard, val_shard, results)

    log("\n=== SWEEP SUMMARY (best val CE per arm, and at which loop count) ===")
    for name, r in results.items():
        if not r["history"]:
            continue
        last = r["history"][-1]["val_curve"]
        best_r = min(last, key=last.get)
        print(f"  {name:<18} best: r={best_r:>3} CE={last[best_r]:.4f}  "
              f"r1={last.get(1, float('nan')):.4f}  full curve={last}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    log(f"DONE, total elapsed {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
