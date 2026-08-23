# DECISIONS.md — every choice, with its provenance

For an AI reader. No prose padding. Written 2026-08-23 in response to a reviewer's request for a
**blind-spot pass on the choices, not the results**.

**Provenance tags.**
- **MEASURED** — an ablation in this project decided it. Section named.
- **INHERITED** — copied from a reference implementation or an earlier config. Source named, and
  whether that config is still the one being run.
- **ASSUMED** — chosen on judgement, never tested.
- **UNEXAMINED** — nobody, including me, thought about it until this table was written.

Values are pulled from `src/model.py` / `src/train.py` programmatically, not recalled. Note the
`Config` defaults differ from the **headline** config in two places (`state_renorm`, `batch_size`);
the headline column is what actually ran.

---

## 1. Model

| choice | value (headline) | tag | provenance / what would change it |
|---|---|---|---|
| base block | Qwen3 decoder layer | INHERITED | task says "base on Qwen3". Verified against the real `Qwen3DecoderLayer` to **2.38e-07** before any compute (`test_model.py` [2]) |
| hidden_size | 448 | MEASURED | `param_budget.py` search over (H, heads, kv, d_head, I, V, layers) for max block share under 10M. 89% of params land in the reused block |
| n_heads / n_kv_heads | 4 / 2 | ASSUMED | GQA ratio 2:1 taken as conventional. Never varied. Attention is ~25% of a layer, so the sensitivity is real but unmeasured |
| head_dim | 112 | ASSUMED | = 448/4. Never varied independently |
| intermediate_size | 1344 | ASSUMED | MLP ratio **3.0**, not Qwen3's 8/3 ≈ 2.67 nor the common 4.0. Chosen by the budget search as a round multiple; **never screened as an axis.** ~64% of block params sit here, so this is the largest unscreened capacity decision in the model |
| activation | SwiGLU | INHERITED | Qwen3. Never questioned; standard across every looped paper read |
| layers_per_loop | 3 | ASSUMED | "one block looped r times" reading of the task. 1 and 2 were in the budget search but never *trained* |
| vocab_size | 4096 | ASSUMED | small so the embedding table does not eat the 10M budget (V·H = 1.8M at 4096). **Far below the field**: Schwethelm 32k, LoopMTP 50k, MixerLoop 32k. Makes token-level perplexity **incomparable to any published number** — which is why bits/byte is reported alongside |
| positional | RoPE, theta 1e4 | INHERITED | Qwen3 default. Untouched |
| max_position_embeddings | 512 | ASSUMED | 2× seq_len. Never exercised beyond 256 |
| norm | RMSNorm, eps 1e-6, **fp32 upcast** | INHERITED | copied from Qwen3's reference *including the upcast*. §6.0b: at the deep schedules' ‖h‖≈1e5 the per-element mean-of-squares is 2.3e7 vs fp16's 65,504 ceiling — a hand-rolled fp16 norm would overflow to `inf`, and the risk **grows with loop count**. Verified empirically |
| **state_renorm** | **False** | **MEASURED** | §4.1: −0.744 nats, the largest single effect in the project. §4.3: the `True` variant contracts and goes inert |
| inject_mode | additive | MEASURED | §4.1 null vs concat+adapter. §3: measured `‖e‖/‖h‖ = 7e-5`, which *explains* the null — injection is negligible against a state that grows |
| depth_init | True, `1/√(2·n_loop_eff)` | INHERITED | Huginn's `σ²_out = 1/(5hl)` idea, adapted. §5.0 tested the alternative (`ε=λ/(N√L)`) and found **no measurable benefit** |
| n_loop_eff | 24 | ASSUMED | **and known-wrong**: it should be μ_rec of the schedule actually run (18 for U[4,32], 40 for U[32,48]). Left at 24 deliberately so every arm shares one init; flagged in `run_residual_scale.py`'s docstring |
| learned h0 | yes | ASSUMED | a **third option** neither exemplar uses — Huginn samples `s₀ ~ TruncNormal(2/5)`, Done Right ablates random init and finds it mixed-to-negative. Never compared against either here |
| prelude / coda | 0 / 0 | MEASURED | §4.5, budget-matched: a prelude buys 0.355 nats **and makes the model depth-inert over [1,96]** |
| loop conditioning | none | ASSUMED | §3.4's function-vs-table rule rejects per-t tables. **But see the honest note below** — annealed supervision (§4.17) *is* loop conditioning via the loss |
| MoE | excluded | ASSUMED | economics invert under a parameter cap: MoE spends stored params to buy sparse FLOPs; here params are the binding constraint and FLOPs are free. **Untested, and it is the live threat to §3.3** |
| attention variant | GQA | ASSUMED | MLA never tried. Not a parameter play at this width (~neutral vs GQA-2); it is a *quality* play, and its cache benefit is worth zero under teacher-forced eval |

## 2. Data and tokenizer

| choice | value | tag | provenance |
|---|---|---|---|
| corpus | FineWeb `sample-10BT` | ASSUMED | task says FineWeb. **The field mostly uses FineWeb-*Edu*** (Schwethelm, Parcae, LoopMTP, LoopFormer). Untested difference |
| tokenizer | byte-level BPE, 4096 | ASSUMED | trained on the first ~19,319 docs; packing starts at doc 20,000 so train/val never see tokenizer-training text |
| **tokenizer shipping** | `configs/tokenizer.json` | **MEASURED** | §6.0 rows 20–21. Identity with the Kaggle-trained vocab is now **verified** (`check_tokenizer_identity.py`), and cold `data.py` from a fresh clone reproduces **sha256-identical** shards |
| digit handling | **80 multi-digit tokens** merged (`00`,`10`,`12`,`19`…) | UNEXAMINED | raised early, never resolved. Marginal for bpb; makes any arithmetic claim from this model unsafe. None is made |
| seq_len | 256 | ASSUMED | §8.0b argues depth demand scales with context, so this is likely a **first-order** choice made on cost grounds alone |
| tokens packed | 92.0M of a 100M cap | UNEXAMINED | the packing target bound, not the cap. 8% of the allowance never used |
| bytes/token | 3.3358 | MEASURED | re-verified to 4 dp over the full 6M-token val shard (20,014,585 / 6,000,000). An earlier draft used 3.45 *chars* from a 5-doc sample |

## 3. Optimization — **the weakest section of this table**

| choice | value | tag | provenance |
|---|---|---|---|
| optimizer | AdamW, β=(0.9, 0.95) | INHERITED | standard. **Correctly matches the lineage being replicated**: Sharma & Vu, Huginn, SCSE, STARS and Loopie(20B) are all AdamW. Muon is used by Parcae/LoopMTP/Schwethelm — a different (speedrun) culture, not a scale threshold |
| **learning rate** | **3e-3** | **INHERITED — and the config it came from no longer exists** | tuned under the **center** config (`state_renorm=True`). Turning that off moved ‖h‖ by 3 orders of magnitude and the gradient regime with it (‖G‖_F 31× smaller tied-vs-untied at init). **Never re-swept in the regime actually run**, and it is ~10× Sharma & Vu's 3e-4 at this parameter scale, 6× Huginn's 5e-4. **Being screened now** (`tlab-hyper-screen`) |
| min_lr / schedule | 3e-4, cosine | INHERITED | 10:1 ratio, conventional. Never varied |
| warmup | 40–100 steps | ASSUMED | varies by driver. SCSE uses 500. Never tested |
| **weight decay** | **0.05** | **UNEXAMINED** | not one of the five axes, never varied. Field: 0.1 (SCSE, Loopie), 0.01 (Sharma & Vu). **Being screened now** |
| grad_clip | 1.0 | INHERITED | conventional. Never varied |
| batch_size | 8 (32 in the default dataclass) | ASSUMED | 8 chosen for MPS memory and then held fixed **so deep schedules stay comparable** — deliberately not reduced to buy memory when μ_rec=44/56 OOM'd |
| tokens/step | 2,048 | derived | 8 × 256 |
| precision | **fp32 throughout** | MEASURED | verified: no `autocast`/`float16`/`bfloat16`/`half`/`GradScaler` anywhere in the training path |
| gradient checkpointing | **none** | UNEXAMINED | full activation retention across all loops. **This is what bounds the deep schedules** — μ_rec=56 and 44 both OOM'd at 14.75 GiB; Huginn checkpoints every recurrent step. Cost: the μ_rec=56 pair |
| optimizer state on resume | **not restored** | MEASURED | deliberate: restoring Adam state raised `ZeroDivisionError` on every resume. Fresh Adam per 240s chunk. §4.15: this makes momentum resets land at **load-dependent steps**, one of two sources of the measured run-to-run floor |
| chunking | 240s wall-clock | ASSUMED | sized against a measured ~700s MPS corruption window. **Wall-clock, not step-count** — the reason above |

## 4. The loop and the supervision — where the contribution is

| choice | value | tag | provenance |
|---|---|---|---|
| loop schedule | `U[4,32]` (headline) / `U[32,48]` (deep) | MEASURED | §4.11, §4.16b: useful depth is a fixed fraction of trained depth |
| BPTT | full, no truncation | MEASURED | §4.1 tested `truncate_bptt=8`; full won |
| supervision density | k=5 → k=1 for the last 10–25% | **MEASURED** | §4.14, §4.16 (threshold at k=1, not a dial), §4.17 (annealing, 2 seeds, in-job control) |
| readout | final loop, RMSNorm'd | MEASURED | §4.6 / `scale_control`: raw and final-only readouts tested |
| aggregation across loops | none (final state only) | ASSUMED | equals LoopMTP's "Only last" ablation, which their gate beats by ~200k params. **But it conflicts with annealing** — annealing works by moving the readout *to* the end; aggregation spreads it back across the trajectory. Cannot have both |
| eval protocol | chunked, non-overlapping | MEASURED | §4.2: sliding-window stride-64 gives 1.6436 bpb vs chunked 1.6938. Chunked used **everywhere** for comparability; both reported |

---

## Q1. MEASURED rows decided under a config that has since changed

1. **§4.1's entire screening sweep** was run at a **wall-clock** budget, not a token budget, and cost/step varies 4.7× across arms. Already retracted in-text; two of five axes flip. The `state_renorm` result survives because its effect (−0.744) is an order of magnitude above the confound.
2. **`inject_mode`, `depth_init`, `truncate_bptt`** were all decided in that same sweep, at ~1M tokens, with `state_renorm=True`. **None has been re-tested in the no-renorm regime**, and §4.15 puts the floor at that budget above several of the effects. These are the weakest MEASURED rows in the table.
3. **`n_loop_eff=24`** was set for a schedule (μ_rec=24) that no headline run uses.
4. **LR 3e-3** — the headline case, described above.

## Q2. What a reviewer of this literature asks first, ranked by how much it could move the headline

1. **LR 3e-3 at 9M params** — 10× the field. If 1e-3 wins by >0.05 nats, every number here is measured at a stated handicap. *Screening now.*
2. **MLP ratio 3.0, never screened** — 64% of block params. Larger capacity lever than anything in §4.
3. **Weight decay 0.05, never varied** — *screening now.*
4. **vocab 4096 vs the field's 32–50k** — makes token perplexity incomparable; bits/byte mitigates but does not remove it.
5. **seq_len 256** — §8.0b argues depth demand scales with context, so this may cap the very effect being studied.
6. **FineWeb vs FineWeb-Edu** — the field's default differs from the task's wording.
7. **No gradient checkpointing** — bounded the deep schedules and cost a pair of arms.
8. **MoE excluded** — Sparse Layers (2605.09165) argues *dense* looped models scale worse than sparse ones, which would place every conclusion here in the unfavourable regime.

## Q3. Unknown unknowns — what I did not think to ask

- **§3.4 now argues against itself.** It says this architecture "has no loop conditioning at all, so it passes the function-vs-table rule trivially and uninterestingly." Since §4.17, that is false in spirit: **annealed supervision is loop conditioning applied through the loss rather than the parameters**, and it satisfies the rule *perfectly* — there is no table because there are no parameters. §3.4 should use §4.17 as its example. Not yet fixed.
- **STARS's placement taxonomy has four cells; two were tested.** Pre-Norm (unbounded growth) and approximately Post-Norm (`state_renorm=True`, bounded, poor attractors) — both reproduced independently here at 9M. **Pre-Sandwich** (`h + Norm(f(Norm(h)))`, bounding ‖Δh‖ *without* bounding ‖h‖, which is what LoopMoE uses) was never tried. Prediction from this project's own data: still 1/t, because ‖h‖ stays linear.
- **The KV trajectory is already nearly depth-invariant** (§4.3: `‖norm1(h)‖` 25.13→21.36 while `‖v‖` falls 82.9→39.6), which means CART's "compute K,V once and reuse" is approximately where this model lands *unforced*. Never named as an axis.
- **Nothing here tests whether the annealing effect survives a vocabulary or sequence length the field would recognise.** Every supervision result is at V=4096, T=256.
- **No seed-level variance on the headline 90M runs.** One seed each, 9.4h apiece.

## Q4. Changing which of these would invalidate comparability

**Would invalidate** (do not change before the deadline): LR, weight decay, batch size, seq_len, vocab, MLP ratio, `n_loop_eff`, the eval protocol, `BYTES_PER_TOKEN`. Every cross-run number in a 3,300-line report is measured against these.

**Safe to change** (affects nothing already measured): gradient checkpointing (memory only), the chunk length, anything in the monitoring/harvest layer, and adding *new* arms at held-fixed values — which is why the LR and weight-decay screens are run as **new arms with an in-job reference**, not as a change to the headline config.
