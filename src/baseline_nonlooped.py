"""Compute-matched non-looped baseline (report.md sec 8 item 1 -- flagged there as "the single
biggest gap" and "cheap to add"). Same building blocks as model.py (RMSNorm, RotaryEmbedding,
DecoderLayer -- imported directly, not re-derived), assembled as N *distinct* decoder layers applied
once each: no weight tying, no loop, no per-loop readout. Answers a different question than the
ablations in report.md sec 4: does depth help because it's a *loop* specifically, or would the same
FLOPs/token in an ordinary (non-tied) deep stack do just as well?

Loopie's framing (arXiv 2607.16051, report.md sec 2): compare at matched *compute*, not matched
parameters -- a looped model trades parameters for reused compute by construction, so this baseline
is *expected* to have far more parameters than the <=10M looped submission at the same FLOPs/token.
That is the point, not a flaw in the comparison.

MATCHED_LOOPS=11: no_state_renorm's actual best loop count (report.md sec 4.2, the number the
headline claim rests on) -- not an arbitrary or generous choice, and stated here so it's checkable.
N = 3 * MATCHED_LOOPS = 33 distinct decoder layers (layers_per_loop=3, matching model.py's Config).

Trains on the SAME local data (data/train.bin, data/val.bin) as every other local run in this
project, not a fresh stream -- this makes the comparison strictly more comparable than the Kaggle
run's necessarily-different data, at the cost of a smaller reachable token budget (local MPS
throughput, not a T4's).

Chunked/resumable subprocess execution, same reasoning and same pattern as train.py's run(): sustained
MPS load on this hardware silently corrupts output after ~700s in one process (LOG.md 2026-08-13
01:52), so this script runs one bounded chunk per invocation and is meant to be relaunched with
--resume by an external loop (matching run_full.py's own pattern), not run continuously in-process.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as LoopedConfig, RMSNorm, RotaryEmbedding, DecoderLayer  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MATCHED_LOOPS = 11
N_LAYERS = LoopedConfig().layers_per_loop * MATCHED_LOOPS  # 33

if torch.backends.mps.is_available():
    torch.mps.set_per_process_memory_fraction(14.0e9 / torch.mps.recommended_max_memory())


class NonLoopedTransformer(nn.Module):
    """N distinct DecoderLayers, applied once each, no weight tying across layers. Reuses model.py's
    primitives verbatim so every architectural choice except looping/tying stays identical."""

    def __init__(self, cfg: LoopedConfig, n_layers: int):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        self.embed = nn.Embedding(cfg.vocab_size, H)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(n_layers))
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings)
        self.final_norm = RMSNorm(H, cfg.rms_norm_eps)
        self.lm_head_weight = self.embed.weight  # tied, matching the looped model's own convention
        self.apply(self._init_weights)
        self._apply_depth_init()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = math.sqrt(2.0 / (5.0 * self.cfg.hidden_size))  # same width-aware std as model.py
            nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _apply_depth_init(self):
        """The looped model's depth_init axis (model.py's _apply_depth_init) scales residual-branch
        output projections by 1/sqrt(2*n_loop_eff) -- report.md sec 4.1 measured this as a real,
        moderate stabilizer (+0.140 val CE when turned off), not a cosmetic choice. A 33-DISTINCT-layer
        non-tied stack has the same deep-residual-stack instability this technique targets (standard
        GPT-2/nanoGPT practice: scale residual output projections by 1/sqrt(2*n_layer)) -- omitting it
        here was the actual bug (not a harness bug) behind a real NaN at step 51 of the first training
        attempt (LOG.md 2026-08-16). Uses N_LAYERS directly (33 real sequential layers here, vs the
        looped model's n_loop_eff=24 *effective* loops over 3 shared layers) since that's what's
        actually different in this model."""
        scale = 1.0 / math.sqrt(2.0 * len(self.layers))
        for layer in self.layers:
            layer.attn.o_proj.weight.data.mul_(scale)
            layer.mlp.down.weight.data.mul_(scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        h = self.embed(input_ids)
        cos, sin = self.rope(T, input_ids.device, h.dtype)
        for layer in self.layers:
            h = layer(h, cos, sin)
        return F.linear(self.final_norm(h), self.lm_head_weight)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@dataclasses.dataclass
class TrainConfig:
    run_name: str = "baseline_nonlooped"
    seq_len: int = 256
    batch_size: int = 8  # conservative starting point, matching the looped full-BPTT runs' proven
    # safe value, not a measured optimum for this specific (larger, but simpler-per-step) model.
    lr: float = 5e-4  # 6x lower than the looped model's 3e-3. Two escalating fixes before this held:
    # 3e-3 NaN'd at step 13 even with depth_init; 1e-3 got further (step 142) then still NaN'd, loss
    # ticking back up (6.54->6.62) just before -- a slow-buildup pattern, not an immediate blowup,
    # consistent with unbounded pre-norm residual-stream growth over 33 real sequential layers (no
    # renormalization anywhere in this stack, unlike the looped model's optional loop_norm) rather than
    # a single bad LR choice. Not independently tuned/swept -- flagged as a real limitation (single
    # seed, small manual LR search) if this run's numbers get reported.
    min_lr: float = 5e-5
    warmup_steps: int = 300
    weight_decay: float = 0.05
    grad_clip: float = 0.5  # tighter than the looped model's 1.0 -- see lr comment above
    total_tokens: int = 30_000_000
    eval_every_tokens: int = 100_000  # ~49 steps at batch=8/seq=256 -- must be comfortably inside one
    # 240s chunk (measured ~165 steps/chunk) or no checkpoint ever saves and every chunk restart loses
    # all progress. Same bug class as run_full.py's original eval_every_tokens miscalibration
    # (LOG.md 2026-08-13) -- should have checked for it here too, didn't, caught by reading the actual
    # step numbers in the log rather than trusting the driver loop's own "chunk done" summary.
    eval_batches: int = 15
    seed: int = 0
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt_dir: str = str(ROOT / "checkpoints")


def get_batch(shard, batch_size, seq_len, device, rng):
    ix = rng.integers(0, len(shard) - seq_len - 1, size=batch_size)
    x = np.stack([shard[i:i + seq_len] for i in ix]).astype(np.int64)
    y = np.stack([shard[i + 1:i + seq_len + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


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
def evaluate(model, val_shard, cfg, rng):
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = get_batch(val_shard, cfg.batch_size, cfg.seq_len, cfg.device, rng)
        logits = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    return float(np.mean(losses))


def run(train_cfg: TrainConfig, log_path: pathlib.Path, max_seconds: float, resume: bool):
    train_shard = np.memmap(DATA_DIR / "train.bin", dtype=np.uint16, mode="r")
    val_shard = np.memmap(DATA_DIR / "val.bin", dtype=np.uint16, mode="r")

    model_cfg = LoopedConfig()  # only field that matters here: hidden_size/heads/etc, not layers_per_loop
    model = NonLoopedTransformer(model_cfg, N_LAYERS).to(train_cfg.device)
    opt = build_optimizer(model, train_cfg)

    tokens_per_step = train_cfg.batch_size * train_cfg.seq_len
    total_steps = train_cfg.total_tokens // tokens_per_step
    eval_every_steps = max(1, train_cfg.eval_every_tokens // tokens_per_step)

    ckpt_dir = pathlib.Path(train_cfg.ckpt_dir) / train_cfg.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "last.pt"

    start_step = 0
    history = []
    if resume and ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location=train_cfg.device, weights_only=False)
        model.load_state_dict(saved["model"])
        # Optimizer state deliberately not resumed -- same fix as train.py's run(), same reason
        # (ZeroDivisionError in Adam's bias_correction1 from a device-transfer issue on reload).
        start_step = saved["step"] + 1
        rng = np.random.default_rng()
        rng.bit_generator.state = saved["rng_state"]
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        history = json.loads(log_path.read_text()) if log_path.exists() else []
        print(f"resumed from step {saved['step']} ({saved['tokens']/1e6:.2f}M tokens)", flush=True)
    else:
        torch.manual_seed(train_cfg.seed)
        rng = np.random.default_rng(train_cfg.seed)
        print(f"params={model.num_parameters():,} n_layers={N_LAYERS} "
              f"(matched to loop={MATCHED_LOOPS})", flush=True)

    t0 = time.time()
    tokens_seen = start_step * tokens_per_step

    for step in range(start_step, total_steps):
        if time.time() - t0 > max_seconds:
            print(f"max_seconds ({max_seconds:.0f}s) reached at step {step}/{total_steps}, stopping",
                  flush=True)
            break
        lr = lr_at(step, total_steps, train_cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_shard, train_cfg.batch_size, train_cfg.seq_len, train_cfg.device, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        lv = loss.item()
        if lv == 0.0 or not math.isfinite(lv):
            raise RuntimeError(f"degenerate training step at step {step}: loss={lv} -- almost "
                                f"certainly a broken forward pass, not a real result. Last good "
                                f"checkpoint (if any) is untouched at {ckpt_path}.")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()
        tokens_seen += tokens_per_step

        if step % 20 == 0 or step == total_steps - 1:
            print(f"step {step}/{total_steps} tok {tokens_seen/1e6:.1f}M loss={lv:.4f} lr={lr:.2e} "
                  f"gnorm={gnorm:.2f} t={time.time()-t0:.0f}s", flush=True)

        if (step % eval_every_steps == 0 and step > 0) or step == total_steps - 1:
            val_ce = evaluate(model, val_shard, train_cfg, rng)
            print(f"  EVAL step {step} tok {tokens_seen/1e6:.1f}M val_CE={val_ce:.4f} "
                  f"val_ppl={math.exp(min(val_ce,20)):.2f}", flush=True)
            history.append(dict(step=step, tokens=tokens_seen, val_ce=val_ce,
                                 train_loss=lv, grad_norm=gnorm.item()))
            log_path.write_text(json.dumps(history, indent=2))
            torch.save({"model": model.state_dict(), "rng_state": rng.bit_generator.state,
                        "torch_rng_state": torch.get_rng_state(), "n_layers": N_LAYERS,
                        "matched_loops": MATCHED_LOOPS,
                        "train_cfg": dataclasses.asdict(train_cfg),
                        "step": step, "tokens": tokens_seen}, ckpt_path)
            if train_cfg.device == "mps":
                torch.mps.empty_cache()

    return history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=240.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--total-tokens", type=int, default=30_000_000)
    args = ap.parse_args()

    tc = TrainConfig(total_tokens=args.total_tokens)
    log_path = ROOT / "checkpoints" / f"{tc.run_name}_history.json"
    hist = run(tc, log_path, max_seconds=args.max_seconds, resume=args.resume)
    print("CHUNK_DONE", "last_entry=", hist[-1] if hist else "none")
