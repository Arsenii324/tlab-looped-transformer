import os
import re
import pathlib
SEED = int(os.environ.get('ARM_SEED', '0'))

SRC = pathlib.Path("/Users/a2mogus/build-projs/barannikov-work/tlab-loop-transformer")
model_code = (SRC / "src" / "model.py").read_text()

out = []
out.append(f'"""DataSphere driver: duo-causal attention + scale-invariant depth gate, seed {SEED}.\n')
out.append('In-job paired. RECURRENCE-side (duo-causal W=2,3) vs READOUT-side (gate) vs control.\n"""\n')
out.append('from __future__ import annotations\n')
out.append('import contextlib, dataclasses, json, math, os, subprocess, sys, time\n')
out.append('import numpy as np, torch, torch.nn as nn, torch.nn.functional as F\n\n')
out.append('T0 = time.time()\n')
out.append('MAX_SWEEP_SECONDS = 3.0 * 3600\n')
out.append('OUT_DIR = "."\n')
out.append('RESULTS_PATH = "results.json"\n\n')
out.append('def log(msg):\n    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)\n\n')

# Append model code (strip future/sys/dataclasses imports)
lines = model_code.splitlines()
m_lines = [l for l in lines if not (l.startswith("from __future__") or l.startswith("import ") or l.startswith("from "))]
out.append("\n# === MODEL DEFINITION ===\n")
out.append("\n".join(m_lines))

# Append training helper routines
out.append("\n\n# === TRAINING ROUTINES ===\n")
# Extract data packing and tokenizer
out.append('''
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
    M25 = 3_500_000

    def make_tcfg(name):
        return TrainConfig(run_name=name, batch_size=8, total_tokens=M25, eval_every_tokens=500_000,
                           supervise_k=5, min_train_loops=4, max_train_loops=32, seed={SEED}, eval_loop_sweep=SWEEP)

    arms = [
        ("dc_control_sSEED", Config(state_renorm=False), make_tcfg("dc_control_sSEED")),
        ("dc_w2_sSEED",      Config(state_renorm=False, kv_window=2), make_tcfg("dc_w2_sSEED")),
        ("dc_w3_sSEED",      Config(state_renorm=False, kv_window=3), make_tcfg("dc_w3_sSEED")),
        ("dg_norm_sSEED",    Config(state_renorm=False, depth_gate_mode="state_norm"), make_tcfg("dg_norm_sSEED")),
    ]

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
''')

out_file = pathlib.Path(os.environ.get("OUT_MAIN", "main.py"))
text = "".join(out)
text = text.replace("sSEED", f"s{SEED}").replace("seed={SEED}", f"seed={SEED}")
assert "SEED" not in text.replace("ARM_SEED",""), "unsubstituted SEED marker left in generated main.py"
out_file.write_text(text)
print(f"Wrote {out_file} ({len(''.join(out))} bytes)")
