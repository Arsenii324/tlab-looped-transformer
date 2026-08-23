---
tags: [looped-transformer, weight-tied, fineweb, from-scratch]
---
# tlab-looped-transformer

Weight-tied looped transformer: one 3-layer Qwen3-style block applied `r` times,
448-dim, **9,064,608 parameters**, trained from scratch on FineWeb next-token
prediction. T-Lab test task submission. Run `full_control90_kaggle`, 90.0M tokens, step 43944.

## Results

| metric | value |
|---|---|
| CE @ 1 loop | **3.9622** |
| best val CE | **3.6599** (at 10 loops) |
| val perplexity | **38.86** |
| bits/byte | **1.5829** (at 3.3358 bytes/token) |
| useful-depth plateau | **[6, 17]** on the dense 1..64 eval grid |
| loop gain (CE@1 − CE@best) | **0.3023** |

Perplexity is **tokenizer-dependent** and this model uses its own 4096-token BPE, so it
is not comparable across submissions; bits/byte is the figure that survives a change of tokenizer.

## These weights are the DENSE control, not the recipe the report recommends

**`METHOD.md` §2 recommends supervision annealing** — dense supervision for most of training, then
terminal-loop-only for the final ~10% of steps. **These weights do not implement it.** They are the
**dense control**: `supervise_k = 5` throughout, no `supervise_k_final`, no `supervise_switch_frac`.

That is deliberate. The annealing recommendation is about **where depth stays useful** — the useful
band widens at 6 of 6 seeds — and **not** about the loss: its CE half was withdrawn at n = 4, and the
six-point mean is −0.0247, inside the replicate floor. So the recipe that buys depth and the
checkpoint that gives the lowest loss are different arms, and the lowest-loss one ships. A 10M-token
checkpoint of the annealed recipe exists (`rec_sw90_s2`); no 90M one does. See `METHOD.md` §4.

## Why this checkpoint, when the report contains a better perplexity

**A reader who finds 37.52 in the report and 38.86 here should know this was a decision, not an
error.** A norm-penalty arm at the same 90M budget reaches **37.52** perplexity. It is not what ships,
for four reasons measured rather than asserted:

1. **88% of its apparent loop-gain advantage is loop-1 damage** (`ΔCE@1 = +0.2263`). It wins the
   loop-gain statistic by making one loop *worse*, not by making depth worth more.
2. **Its useful band narrows**, [6,17] → [6,14]. On the axis this task actually asks about — value
   *from many loops* — it is the worse model.
3. **It is the only arm whose map converges** (ρ = 0.9953 / 0.9915 at loops 32/64), which is the
   regime the report's §2 argues against on independent evidence.
4. **It carries a clipping confound the stored artifacts cannot resolve.**

The control has no confound on either axis. See `report.md` §4.6b and §6.0b/D3, and
`submission/METHOD.md` §4.

## Counting the parameters — the obvious way gives the wrong answer

`sum(v.numel() for v in state_dict.values())` returns **10,899,616**, which is *over* the task's 10M
cap. That is an artifact of **weight tying**, this architecture's central feature: `lm_head` and
`embed` are the same `nn.Parameter` under two names, so a `state_dict` sum counts the tied embedding
twice. The difference is exactly `vocab x hidden` = **1,835,008**.

```python
sum(p.numel() for p in m.parameters())   # 9,064,608  <- the real count, and what every number uses
```

## Files, and why the tokenizer is one of them

- `model.pt` — weights (`torch.load`, `weights_only=False`; contains `model`, `model_cfg`, `train_cfg`)
- `tokenizer.json` — **the vocabulary these weights were trained with.** Do not substitute another
  one and do not retrain it: a mismatch raises nothing and reports CE ≈ ln(4096) =
  8.3178, i.e. chance, which looks like a broken model rather than a broken setup.
- `model.py` — the architecture, so this checkpoint loads without cloning the GitHub repo.

## Verify the download before trusting a number

```bash
python src/check_tokenizer_identity.py <this checkpoint> --expect-ce1 3.9622
```

That gate judges vocabulary against *chance* and protocol drift against the sample's own SEM, so it
distinguishes "wrong vocabulary" from "slightly different eval batch". Expect |diff| well under 0.1.

```python
import torch
from model import Config, LoopedTransformer
ck = torch.load("model.pt", map_location="cpu", weights_only=False)
m = LoopedTransformer(Config(**ck["model_cfg"])); m.load_state_dict(ck["model"]); m.eval()
logits_per_loop, state_norms = m(input_ids, n_loops=10, return_all_loops=True)
```

## This checkpoint's own state norms

| loop | 1 | 8 | 16 | 64 |
|---|---|---|---|---|
| ‖h‖ | 466.6 | 2334.4 | 3977.3 | 12424.4 |

Stated because the released models differ in state scale by up to 380x. Any absolute clamp/threshold level in the report was measured on **one** checkpoint and does not transfer; derive levels from these numbers or re-run `src/radial_clamp.py` on this checkpoint, which does it for you.

## Config

`{"vocab_size": 4096, "hidden_size": 448, "n_heads": 4, "n_kv_heads": 2, "head_dim": 112, "intermediate_size": 1344, "layers_per_loop": 3, "n_prelude": 0, "n_coda": 0, "rms_norm_eps": 1e-06, "rope_theta": 10000.0, "max_position_embeddings": 512, "readout_mode": "norm", "convex_gate": false, "explore_noise": 0.0, "explore_anneal": true, "fixed_gate": null, "truncate_bptt": null, "state_renorm": false, "inject_mode": "additive", "depth_init": true, "residual_scale": null, "scale_clock": false, "gate_alpha_init": 0.874, "n_loop_eff": 24, "cond_mode": "none", "cond_lora_rank": 4, "cond_lora_branches": 4, "cond_fixed_branch": null, "kv_untie_buckets": 1, "depth_gate_mode": "none", "xsa": false, "kv_window": 1}`

See the GitHub repo's `report.md` for the full ablation set, the negative results, and the
failure log (§6.0).
