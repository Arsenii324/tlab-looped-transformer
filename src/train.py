"""Train a LoopedTransformer on the packed FineWeb shards. One file, no config framework -- a
dataclass and a CLI override, matching the rest of this repo.

Dense per-loop supervision: the training loss is the MEAN cross-entropy across all n_loops returned
readouts, not just the final one. This is what makes "how much does each loop help" trainable at
all, and per the Readout Blind Spot paper (arXiv 2606.24898) it is also exactly the setup where
`state_renorm` matters -- dense supervision through a scale-invariant readout trains the exits fine
regardless of whether the carried state's raw scale is controlled, so state-scale drift has to be
watched separately (logged every step here, not just asserted away).

Loop count is randomized per step, uniform over [min_train_loops, max_train_loops] -- a
simplification of PLAN.md's "log-uniform" phrasing (noted here rather than silently matched): the
five ablation axes are tested on/off, not by sampling-distribution shape, so discrete-uniform is a
fine implementation of "randomization on" for that purpose.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig, LoopedTransformer  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# This machine runs several concurrent Claude Code sessions; a runaway allocation here should fail
# loudly (a clean OOM exception) rather than push the whole system into swap, which degrades
# everyone sharing it. First set at 8GB as a round-number guess -- too tight, and finding out why
# was itself informative: full BPTT at high loop counts must retain activations across every
# sequential layer application (32 loops x 3 layers/loop = 96) simultaneously for backward, which is
# a genuine ~8-10GB at batch=16, not a bug. 14GB leaves headroom for that while still bounding a
# real runaway; batch_size is halved in run_screening.py's full-BPTT arms rather than raising this
# further, to keep buying loop-count headroom instead of raw ceiling.
if torch.backends.mps.is_available():
    torch.mps.set_per_process_memory_fraction(14.0e9 / torch.mps.recommended_max_memory())


@dataclasses.dataclass
class TrainConfig:
    run_name: str = "center"
    seq_len: int = 256
    batch_size: int = 32
    lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 100
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    total_tokens: int = 90_000_000
    min_train_loops: int = 4
    max_train_loops: int = 32
    supervise_k: int = 5   # bounded subset of loops supervised per step; see per_loop_ce docstring
    # Supervision annealing (§4.17). Default None = constant density, i.e. every result before
    # 2026-08-23 is unaffected. When set, k switches ONCE at `supervise_switch_frac` of total steps.
    supervise_k_final: int | None = None
    supervise_switch_frac: float = 0.75   # fraction of TOTAL STEPS at which k switches
    norm_penalty: float = 0.0   # lambda for the auxiliary scale-visible loss (arXiv 2606.24898's
    # "norm penalty" intervention): lambda * K^-1 * sum_k E||H_k||^2_rms over the loops actually run.
    # 0.0 = off and the loss is bit-identical to before. Their reported lambda is 0.01.
    eval_every_tokens: int = 5_000_000
    eval_batches: int = 20
    eval_loop_sweep: tuple = (1, 2, 4, 8, 12, 16, 24, 32)
    seed: int = 0
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt_dir: str = str(ROOT / "checkpoints")


def get_batch(shard: np.memmap, batch_size: int, seq_len: int, device: str, rng: np.random.Generator):
    ix = rng.integers(0, len(shard) - seq_len - 1, size=batch_size)
    x = np.stack([shard[i:i + seq_len] for i in ix]).astype(np.int64)
    y = np.stack([shard[i + 1:i + seq_len + 1] for i in ix]).astype(np.int64)
    return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))


def per_loop_ce(logits_per_loop, y, supervise_idx=None):
    """Returns (mean_loss over supervised loops, list of (idx, scalar loss) for logging).

    `supervise_idx`: which loop indices to include. Supervising literally every loop (the original
    design) turns out to be computationally pathological on MPS -- confirmed by direct measurement,
    not assumed: at n_loops=16, all-loops supervision gave erratic 37-158s/step, while final-loop-only
    was a stable 5.6s/step and a bounded 5-loop subset was a stable ~6.5s/step (see LOG.md). Every
    intermediate loop's state has two outgoing graph edges under dense supervision -- one continuing
    the recurrence, one into that loop's own loss -- against one edge under final-only; MPS handles
    that fan-out far worse than the chain alone. A small fixed-size subset (final loop always
    included, plus a few random earlier ones) keeps the fan-out bounded regardless of how many loops
    actually run, while still teaching intermediate loops to be useful, which is the entire point."""
    if supervise_idx is None:
        supervise_idx = range(len(logits_per_loop))
    per_loop = []
    for i in supervise_idx:
        l = F.cross_entropy(logits_per_loop[i].reshape(-1, logits_per_loop[i].size(-1)), y.reshape(-1))
        per_loop.append((i, l))
    loss = torch.stack([l for _, l in per_loop]).mean()
    return loss, [(i, l.item()) for i, l in per_loop]


def sample_supervise_idx(n_loops: int, k: int, rng) -> list[int]:
    """Final loop always included (it's what eval ultimately reports); up to k-1 more sampled
    uniformly from the rest, without replacement."""
    last = n_loops - 1
    if k >= n_loops:
        return list(range(n_loops))
    rest = rng.choice(last, size=min(k - 1, last), replace=False).tolist() if last > 0 else []
    return sorted(set(rest + [last]))


def lr_at(step, total_steps, cfg: TrainConfig):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    prog = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(prog, 1.0)))


def build_optimizer(model, cfg: TrainConfig):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)  # RMSNorm weights, h0
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95))


@torch.no_grad()
def evaluate(model, val_shard, cfg: TrainConfig, rng):
    model.eval()
    curves = {r: [] for r in cfg.eval_loop_sweep}
    for _ in range(cfg.eval_batches):
        x, y = get_batch(val_shard, cfg.batch_size, cfg.seq_len, cfg.device, rng)
        max_r = max(cfg.eval_loop_sweep)
        logits_per_loop, state_norms = model(x, n_loops=max_r, return_all_loops=True)
        for r in cfg.eval_loop_sweep:
            loss = F.cross_entropy(logits_per_loop[r - 1].reshape(-1, logits_per_loop[r - 1].size(-1)),
                                    y.reshape(-1))
            curves[r].append(loss.item())
    model.train()
    return {r: float(np.mean(v)) for r, v in curves.items()}


def run(model_cfg: ModelConfig, train_cfg: TrainConfig, log_path: pathlib.Path | None = None,
        max_seconds: float | None = None, resume: bool = False):
    """`max_seconds`: hard wall-clock cutoff, checked every step -- independent of the token-budget
    arithmetic (which assumes a throughput estimate that could be wrong). Stops gracefully with
    whatever's already been checkpointed rather than running unbounded.

    `resume`: if True and a checkpoint already exists for this run_name, continues from it (model
    weights, RNG state, step count -- NOT optimizer state, see the load site below for why) instead
    of starting fresh. Exists because sustained MPS load
    measured directly to become unreliable after ~700s in one process on this hardware (LOG.md,
    2026-08-13 01:52: silent all-zero output after a GPU driver error flood, no exception raised) --
    the fix is running training as a sequence of short subprocess invocations, each resuming where
    the last left off, so no single process holds the GPU continuously long enough to hit it."""
    train_shard = np.memmap(DATA_DIR / "train.bin", dtype=np.uint16, mode="r")
    val_shard = np.memmap(DATA_DIR / "val.bin", dtype=np.uint16, mode="r")

    model = LoopedTransformer(model_cfg).to(train_cfg.device)
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
        # Deliberately NOT resuming optimizer.state_dict(): measured directly (LOG.md 2026-08-13
        # 02:15) that doing so raises ZeroDivisionError inside torch's Adam (bias_correction1==0)
        # on every resume, deterministically -- almost certainly the same class of device-transfer
        # issue as the RNG-state bug above (map_location moving Adam's internal step-count tensors
        # somewhere the bias-correction math doesn't expect). A fresh optimizer at each chunk costs
        # a few steps of reduced momentum after every resume; a fragile, silently-wrong resume of
        # 9M parameters' worth of state costs correctness. `opt` is freshly built above and used as-is.
        start_step = saved["step"] + 1
        rng = np.random.default_rng()
        rng.bit_generator.state = saved["rng_state"]
        # torch.set_rng_state (CPU generator) requires a CPU ByteTensor; map_location above moved
        # every tensor in the checkpoint (including this one) onto train_cfg.device.
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        history = json.loads(log_path.read_text()) if log_path and log_path.exists() else []
        print(f"resumed from step {saved['step']} ({saved['tokens']/1e6:.2f}M tokens)", flush=True)
    else:
        torch.manual_seed(train_cfg.seed)
        rng = np.random.default_rng(train_cfg.seed)

    t0 = time.time()
    tokens_seen = start_step * tokens_per_step

    for step in range(start_step, total_steps):
        if max_seconds is not None and time.time() - t0 > max_seconds:
            print(f"max_seconds ({max_seconds:.0f}s) reached at step {step}/{total_steps}, stopping",
                  flush=True)
            break
        lr = lr_at(step, total_steps, train_cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_shard, train_cfg.batch_size, train_cfg.seq_len, train_cfg.device, rng)
        n_loops = int(rng.integers(train_cfg.min_train_loops, train_cfg.max_train_loops + 1))

        _k_eff = (train_cfg.supervise_k if train_cfg.supervise_k_final is None
                  or step < train_cfg.supervise_switch_frac * total_steps
                  else train_cfg.supervise_k_final)
        sup_idx = sample_supervise_idx(n_loops, _k_eff, rng)
        if train_cfg.norm_penalty > 0:
            logits_per_loop, state_norms, rms = model(x, n_loops=n_loops, supervise_idx=set(sup_idx),
                                                       return_state_rms=True)
        else:
            logits_per_loop, state_norms = model(x, n_loops=n_loops, supervise_idx=set(sup_idx))
        loss, per_loop_losses = per_loop_ce(logits_per_loop, y, sup_idx)
        if train_cfg.norm_penalty > 0:
            # Averaged over the loops actually run this step, so its scale does not silently depend
            # on the sampled n_loops (which varies 4-32 here); that would make the penalty an
            # accidental function of the loop schedule rather than of the state norm.
            pen = torch.stack([r.pow(2) for r in rms]).mean()
            loss = loss + train_cfg.norm_penalty * pen

        # Fail loudly on degenerate output rather than silently continue. Not a hypothetical: a
        # sustained-MPS-load GPU driver failure (LOG.md 2026-08-13 01:52) made every forward pass
        # silently return exact zeros -- no exception, no NaN, loss==0.0 -- for hundreds of steps
        # across two arms before it was caught by reading raw log values instead of the "done"
        # summary line. A 4096-way cross-entropy landing on exactly 0.0 does not happen from real
        # training; only from a broken forward pass. Checked here, at the point of computation, not
        # discovered later by a human reading the log.
        lv = loss.item()
        if lv == 0.0 or not math.isfinite(lv) or state_norms[-1] == 0.0:
            raise RuntimeError(
                f"degenerate training step at step {step}: loss={lv} state_norm_last={state_norms[-1]} "
                f"-- almost certainly a broken forward pass (GPU driver failure or similar), not a "
                f"real result. Last good checkpoint (if any) is untouched at {ckpt_dir/'last.pt'}.")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()

        tokens_seen += tokens_per_step

        if step % 20 == 0 or step == total_steps - 1:
            entropy_proxy = per_loop_losses[-1][1]  # final-loop CE, cheap proxy logged every step
            print(f"step {step}/{total_steps} tok {tokens_seen/1e6:.1f}M loss(mean)={loss.item():.4f} "
                  f"loss(final)={entropy_proxy:.4f} lr={lr:.2e} gnorm={gnorm:.2f} "
                  f"state_norm[0,-1]=({state_norms[0]:.1f},{state_norms[-1]:.1f}) "
                  f"n_loops={n_loops} t={time.time()-t0:.0f}s", flush=True)

        if step % eval_every_steps == 0 and step > 0 or step == total_steps - 1:
            curve = evaluate(model, val_shard, train_cfg, rng)
            print(f"  EVAL step {step} tok {tokens_seen/1e6:.1f}M per-loop val CE: " +
                  " ".join(f"r{r}={v:.4f}" for r, v in curve.items()), flush=True)
            history.append(dict(step=step, tokens=tokens_seen, val_curve=curve,
                                 train_loss_mean=loss.item(), grad_norm=gnorm.item(),
                                 state_norm_first=state_norms[0], state_norm_last=state_norms[-1]))
            if log_path:
                log_path.write_text(json.dumps(history, indent=2))
            torch.save({"model": model.state_dict(),
                        # optimizer.state_dict() deliberately not saved -- see the resume-load
                        # comment above; it is never read back.
                        "rng_state": rng.bit_generator.state,
                        "torch_rng_state": torch.get_rng_state(),
                        "model_cfg": dataclasses.asdict(model_cfg),
                        "train_cfg": dataclasses.asdict(train_cfg),
                        "step": step, "tokens": tokens_seen},
                       ckpt_dir / "last.pt")
            if train_cfg.device == "mps":
                torch.mps.empty_cache()  # release cached-but-unused memory; other sessions share this machine

    return history


if __name__ == "__main__":
    mc = ModelConfig()
    tc = TrainConfig()
    hist = run(mc, tc, log_path=ROOT / "checkpoints" / f"{tc.run_name}_history.json")
    print("done:", hist[-1] if hist else "no eval logged")
