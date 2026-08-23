# Full state dump for the reviewing agent — 2026-08-22 ~23:30 MSK, ~24h to deadline

Written for an AI reader, not a human. Maximal detail, including things I think are wrong or
suspicious. Nothing withheld. Where I have not measured something, it says so.

Reading order for the repo's docs: `OPS.md` (live status) → this file → `report.md` (deliverable)
→ `LOG.md` (chronological ledger) → `PLAN.md` (original rationale).

---

## 1. The model, exactly

One config, `src/model.py::Config`. Headline runs use `state_renorm=False`, everything else default.

| field | value | note |
|---|---|---|
| vocab_size | 4096 | custom byte-level BPE (GPT-2 style pre-tokenizer), trained on ~60MB FineWeb text |
| hidden_size | 448 | |
| n_heads / n_kv_heads | 4 / 2 | GQA, groups=2 |
| head_dim | 112 | note **n_heads·head_dim = 448 = H**, so no dimension expansion in attention |
| intermediate_size | 1344 | **exactly 3·H**, not 8/3·H and not 4·H — see §6.1, I think this is a real inefficiency |
| layers_per_loop | 3 | the reused block is 3 full Qwen3-style decoder layers |
| n_prelude / n_coda | 0 / 0 | headline. §4.5 of report.md tests 1/1 and 1/0 and 0/1 |
| rms_norm_eps | 1e-6 | |
| rope_theta | 10000.0 | |
| max_position_embeddings | 512 | but seq_len used is 256 |
| truncate_bptt | None | full BPTT through every loop |
| state_renorm | False | winning arm; True is the `center` control |
| inject_mode | "additive" | `h_{t} = block(h_{t-1} + e)`, e constant in t |
| depth_init | True | scales o_proj and down_proj by `1/sqrt(2·n_loop_eff)` |
| n_loop_eff | 24 | ← **suspicious**, see §6.4; mean of the train sampler is 18, not 24 |
| readout_mode | "norm" | RMSNorm before tied head. "raw"/"final_only" implemented, arms queued |

Block internals are a faithful Qwen3 decoder layer, **verified numerically against the installed
`transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer`** with weights copied across:
`max|diff| = 2.38e-07` (test [2]). Pre-norm RMSNorm, per-head QK-Norm on head_dim *before* RoPE,
GQA via `repeat_interleave`, SwiGLU, no biases anywhere, `F.scaled_dot_product_attention(is_causal=True)`.

Loop body, verbatim semantics:
```
e = embed(x); e = prelude(e)              # prelude is empty in headline runs
h = h0.expand(B,T,H) + e                  # h0 is a learned [H] parameter, init zeros
for t in range(n_loops):
    h_in = (h + e) if t > 0 else h        # additive injection; t=0 already has e
    h    = block(h_in, cos, sin)          # 3 decoder layers, weights shared across t
    if state_renorm: h = loop_norm(h)     # separate RMSNorm, NOT the readout norm
    logits_t = lm_head(final_norm(coda(h)))   # tied lm_head = embed.weight
```

**Parameter budget: 9,064,608 total** (state_renorm=False) / 9,065,056 (True, +448 for loop_norm).
Independently re-derived by `src/param_budget.py` and asserted equal in test [1] across 4 configs.
- reused block: 7,228,704 (79.7%) = 3 × 2,409,568
- tied embedding: 1,835,008 (20.2%) = 4096 × 448
- h0 + final_norm (+ loop_norm): 896–1,344
One decoder layer at this width = 2,409,568 (attn 602,336 incl. 224 QK-norm params; MLP 1,806,336;
2 norms 896). **MLP is 75% of the layer.**

Convention used: **total** params ≤10M, embeddings included. arXiv 2604.21106 reports
non-embedding; under that convention I'd have 7.23M and ~2.8M of headroom. This is an unresolved
decision I flagged for the user and did not act on unilaterally.

## 2. Training setup, exactly

`src/train.py`. AdamW (`torch.optim.AdamW`), betas default (0.9, 0.999), weight_decay 0.05 applied
only to `ndim>=2` params (RMSNorm weights and h0 excluded). Cosine LR to `min_lr`, linear warmup.
Headline: lr 3e-3 → min 3e-4, warmup 60–100 steps, grad_clip 1.0, batch_size 8, seq_len 256.

**Loop-count schedule:** `n_loops ~ Uniform{4..32}` per step, so μ_rec = 18.
**Supervision:** the final loop is always supervised, plus up to `k−1 = 4` more loop indices sampled
uniformly without replacement; loss = mean CE over that supervised subset. Measured density (by
simulating the actual sampler, not by assumption): d(1)=0.337, d(4)=0.335, d(8)=0.233, d(12)=0.174,
d(16)=0.132, d(24)=0.075, d(32)=0.035. **Monotonically decreasing — so the "optimum sits at the
supervision-density peak" hypothesis is already false as stated**, since the density peaks at loops
1–4 and the optimum is 8–11.

Data: FineWeb streamed, packed to uint16 shards. 92M train / 6M val tokens. Measured **3.3358
bytes/token** over the full val shard (this constant was wrong twice before: estimated from 5
documents, and counting chars not bytes; every earlier bits/byte was ~3.4% optimistic).

## 3. Every result, with the instrument that produced it

**Headline (Kaggle T4, 45,975,552 tokens, 5.29h):** best val CE **4.0071 at loop 8**, CE@1 4.2580,
loop gain 0.2509 nats, **bits/byte 1.7330**. Degrades smoothly to loop 128; **crosses its own loop-1
CE at loop 106** (3.3× the trained max of 32). No NaN, no cliff.

**Screening (7 arms, 18 min each, ~0.9–1.2M tokens):** `no_state_renorm` −0.744 nats vs center was
the only large effect. **The sweep was wall-clock-budgeted, which is a systematic confound**: the two
arms that got 34% more tokens are exactly `truncate8` and `fixed_loops16`. Token-corrected at this
project's own measured 0.398 nats/e-fold: `fixed_loops16` −0.128 → ~−0.01 (null), `truncate8` −0.016
→ ~+0.10 (**reversed**; full BPTT wins). Seed spread on the *same* comparison is 0.746 vs 0.496 =
0.25 nats, and four of five nominal effects are below that. **Only `state_renorm` is resolved.**

**Dynamics (`src/state_dynamics.py`, readout-space):** the winning config does **not** contract.
Relative perturbation ‖Δh‖/‖h‖ flat ≈1.2 and *larger* at loop 64 than 24. Consecutive increments
align at `cos(du_t,du_{t−1}) → 0.9999`; increment is **97–98% parallel to h itself**; ‖h‖ 1655 →
30097 roughly linearly. Unit step halves per doubling of t (0.0249/0.0105/0.0051/0.0026 at 8/16/32/64)
= exactly `1/t`. **Saturation is geometric dilution, not fixed-point convergence.** Instrument passes
its null: on `state_renorm=True` the same script reports textbook contraction (rel. pert 0.211 →
0.0000, ‖h‖ pinned at 29.6361 from loop 16). This is a **replication of Lemma 2 of arXiv 2606.24898**
(confirmed from its LaTeX), not a discovery; what's ours is the contraction refutation, the
*persistence* of direction (Lemma 2 bounds step size, not step agreement), and the depth range.

**Radial clamp (`src/radial_clamp.py`, zero training):** clamping each token's RMS to ‖h₁‖/‖h₈‖/‖h₁₆‖
before both decode and recurrence relocates the optimum to loop **5 / 15 / 24** while best CE stays
**4.0115 / 4.0114 / 4.0133 vs 4.0071 unclamped — a 0.006-nat spread**; loop gain flat 0.245–0.251.
Tight clamp degrades to **7.71 at loop 64** (vs 4.16 unclamped). Unclamped control reproduces the
published eval to **1.9e-07**. Reproduces 2606.24898's Table 4 result at K=4 (they report ΔCE
+0.0004…+0.0055; I get −0.012), and inverts past it — **their entire study stops at K=4**.
Implication: **scale is a rate parameter, not a ceiling.** Norm growth is partly *protecting* the
model from a drift that is monotonically harmful past loop 8.

**Sandwich (`src/run_sandwich.py`, 4 arms, all 9,064,608 params, all 1,187,840 tokens, all 12–98
layer-applications/step):** double dissociation. Prelude buys 0.34 nats and destroys 88% of loop gain
(0.0586 → 0.0071); coda buys ~nothing on CE and moves the optimum 11 → 20. Full sandwich saturates
at loop 4 *despite training on [12,96]*. A naive prelude+coda at H=448 would cost 13.88M vs a 10M cap
— at this budget the envelope must be paid for out of the reused block.

**Injection tension, resolved at zero compute:** ‖h‖ after loop 1 over training is 35.3 (step 24) →
**27,926 (peak, step 504)** → 3,696 (final). ‖e‖ is 0.424 at init, 1.879 final. So ‖e‖/‖h‖ ≈ 1.2e-2
early and ~1e-4 after the explosion. **Injection is a formative-phase mechanism**; `inject_none` is
the worst screening arm because it starves representation-building, not because injection does work
in the converged model. Also: **‖h‖ peaks at step 504 and falls 7.6× afterwards** — the norm growth
is not a monotone training-time pathology.

**Compute-matched non-looped baseline:** 33 distinct layers, 81.4M params, no tying. Could not be
trained stably (NaN at steps 13/51/142/411 across LR 3e-3→5e-4, grad_clip 1.0→0.5, +depth_init).
**Caveat I hold openly: this ran on MPS, the backend this project documents as producing fake NaNs
under load.** I am not deleting the result but it needs a CUDA rerun to be load-bearing.

## 4. Instruments (all pinned by tests)

`src/test_model.py` — 9 checks, all passing: [1] param count vs independent formula across 4 configs;
[2] vs real Qwen3DecoderLayer (2.38e-07); [3] full-BPTT vs truncated forward identity (0.0);
[4] no_grad windowing; [5] state_renorm bounds the norm; [6a] n_prelude=n_coda=0 bit-identical to the
pre-sandwich model (0.0) and [6b] non-vacuity; [7] `kaggle/main.py`'s inlined model copy vs
`src/model.py` across 3 topologies (0.0); [8] the cross-depth `kv_source` hook is inert when unused
(0.0); [9] readout modes differ in the right places + the norm penalty is differentiable.

`src/paired_eval.py` — **new, and the thing that makes the rest interpretable.** Eval noise between
two independent samples of the *same* checkpoint is ~0.065 nats, larger than most effects measured.
Frozen set of 2048 sequences × 256 tokens = 524,288 scored tokens = **8.74% of the val shard**
(previous evals used 0.3–0.7%), per-sequence CE, bootstrap over sequences on the paired Δ.

## 5. A real bug worth knowing about

`model.forward` had `ctx = torch.no_grad() if truncating else torch.enable_grad()`.
**`enable_grad()` overrides an outer `no_grad()`**, so with `truncate_bptt=None` (the default *and*
the winning config) every `@torch.no_grad()` eval retained a full autograd graph across all loops.
That is the actual cause of the Kaggle `eval_batch_size=4`, the 14GB MPS guard, and an eval-boundary
OOM — all previously charged to "64-loop forwards are expensive". Fixed to `nullcontext()`; forward
values identical at `max|diff| = 0.0e+00`. It also unblocked the 128-loop sweep, which was recorded
in the report as blocked on MPS driver fragility.

## 6. Things I think are inefficient or wrong — please attack these

**6.1 MLP ratio 3.0 and the 75% problem.** `intermediate_size = 1344 = 3·H`. SwiGLU has *three*
matrices (gate, up, down), so the MLP is 3·H·I = 1,806,336 params = **75% of every layer**, while
attention is 25%. The param_budget search considered ratios {2.0, 8/3, 3.0, 4.0} and picked by
"maximize block/total", which pushes the ratio *up* — that objective is a proxy I chose, not a
measured optimum. Standard SwiGLU practice is 8/3·H precisely to compensate for the third matrix; at
8/3 I'd free ~200k params/layer = 600k total. **I have never ablated the MLP ratio.** Given that
loops multiply the block, and the MLP is the majority of the block, this is plausibly the single
biggest misallocation in the design.

**6.2 No dimension expansion in attention.** `n_heads·head_dim = 4·112 = 448 = H`. q_proj is square.
With GQA at 2 KV heads, k/v are 448→224. Attention is 602k of 2.41M. At seq_len 256 attention FLOPs
are a rounding error, so this is a *parameter* allocation question, not a compute one. Unexplored:
whether fewer/wider heads or more heads at smaller head_dim matters for a block applied 64 times.

**6.3 Optimizer is plain AdamW. I have not tried Muon.** Muon on the 2D params (the block's four
projections + the MLP's three) with AdamW retained for embeddings/norms is the obvious
cheap-to-try upgrade, and the reported gains are largest exactly where I am (small model, short
horizon). **A concern specific to this architecture that I have not seen addressed anywhere:** the
looped block's weights receive gradient from up to 32 applications per step, so the gradient is a sum
over compositions of the *same* matrix. Whether Muon's orthogonalized update is well-behaved under
that regime is genuinely unclear to me, and it might be the more interesting question than the
speedup.

**6.4 `n_loop_eff = 24` does not match the sampler — and Huginn's own convention says 18.**
RESOLVED as a defect: Huginn initialises out-projections with `σ²_out = 1/(5·h·l)` where
`l = l_P + r̄·l_R + l_C` uses the **mean** recurrence r̄. Our sampler's mean is 18, so by that
convention the constant should key off 18, not the 24 actually used. Verified from their LaTeX.
Not changed mid-flight (it would invalidate comparability with every run already completed), but it
is now a known-wrong constant rather than an unexamined one.

**6.4a original note.** `depth_init` scales output projections by
`1/sqrt(2·n_loop_eff)` = `1/sqrt(48)` ≈ 0.144, but the mean loop count is **18**, not 24. So the
init is tuned for a schedule I am not running. `no_depth_init` was +0.142 nats (worse) in screening,
so the mechanism helps, but the constant is wrong and I never swept it. Related: the reviewer's
`ε = λ/(N√L)` residual-scaling prescription would *replace* this and predicts the
`fixed_loops16`-vs-random result should vanish. **Not implemented — this is item 5 and the only
shortlist item I have not started.**

**6.5 seq_len 256 is probably leaving free loss on the table.** At 3.3358 B/tok that's ~854 bytes of
context, and a scored token averages ~427 bytes of left context. The MLP dominates FLOPs at this
width (~3.6 MFLOP/tok/layer vs ~0.5 for attention scores at T=256), so 256→512 is nearly free in
compute and gives every scored token more context. It also bears on the premise: if per-token
required computation scales with context, 256 tokens caps what depth could ever be *for*. **I kept
256 for comparability with every existing run and did not change it under time pressure.** I think
that was the right call for the headline but it should be stated as a limitation.

**6.6 MLA is not worth it here and I want to say so explicitly.** At seq_len 256 with a 9M model the
KV cache is trivial and attention is a rounding error in FLOPs. The only legitimate reason would be
*parameter* efficiency (low-rank KV freeing budget for the reused block), and that competes directly
with simply fixing 6.1, which is bigger and simpler.

**6.7 The vocab/embedding tension.** 4096 vocab costs 1.84M = 20% of budget. ALBERT-style
factorization (V×k + k×H) with V=32768, k=64 would cost 2.13M and buy ~1.3× more bytes/token, at the
cost of a rank-64 bottleneck on a 32k softmax. Unmeasured. Under the **non-embedding** convention
this whole tension evaporates and I'd spend the freed 2.8M on width instead.

**6.8 Loop-count schedule shape.** Uniform{4..32} was chosen as a simplification of PLAN.md's
"log-uniform" and never revisited. STARS uses log-normal μ=2 σ=0.7 over [1,100]. Given that my
optimum (8) sits at 44% of μ_rec (18) — a deviation from the rule other work reports — the
distribution shape is a live suspect and is what `src/run_supervision.py` is testing right now.

**6.9 Dense per-loop supervision may be the thing capping depth.** Loss is the mean CE over
{final} ∪ {4 sampled}. Sharma & Vu's own finding is that dense per-loop supervision trains the
*exits*, not the recurrent state. I have never trained final-loop-only, and LoopMDM — the one paper
with monotone improvement past its training max — uses **final-loop loss only and no injection**.
That combination contradicts three of my five design axes simultaneously and I have not tested it.

## 7. Compute state, right now

- **Kaggle slot 1** `tlab-loop-fullrun`: 90M tokens, current config. Running since ~20:05, ETA ~06:30.
- **Kaggle slot 2** `tlab-loop-normpenalty`: 90M tokens + norm penalty λ=0.01. Running since ~23:05.
  Model verified bit-identical to `src/model.py` including the penalty term (0.0e+00).
  Kaggle weekly quota ~30 GPU-h; these two consume ~21h of it.
- **Local MPS** (serial): `run_supervision.py` (3 schedules × 2 seeds, 2.5M tok each, token-budgeted)
  → `run_scale_control.py` (raw / final_only / penalty / control × 2 seeds) → paired scoring.
- **DataSphere** just verified working (profile `default` = arsen4ikvar, project `bt12q57tmrs03pnt8drc`,
  `gt4.1`). First job submitting now: the per-token exit dump. Note for the record: the CLI's
  requirements parser rejects a bare `--index-url` line, so the cu121 pin has to go in `cmd:`.

## 8. What I do next, in order

1. **The exiter, and I agree it is the strongest remaining move.** `min_k E[CE] = 4.0071` is a
   statement about the average; `E[min_k CE]` is strictly smaller unless every token has the same
   argmin depth. The gap is the headroom, and measuring it costs one forward pass. `src/exit_dump.py`
   dumps per-token/per-loop CE + entropy + top1−top2 margin + ‖Δh‖/‖h‖ + successive-output KL.
   Then: oracle bound (flagged as label-using and optimistically biased), zero-parameter threshold
   rules, entropy-decile bucketed depth allocation, and a small Q-exit head on the frozen backbone —
   all fit on calibration, scored on test. Under teacher forcing this is pure **readout selection**,
   so none of the KV-cache-absence literature applies. Either outcome is a result.
2. Read the supervision-density and scale-control arms under paired eval when they land.
3. Pull both 90M runs, paired-compare them, pick the headline.
4. Item 5 (`ε = λ/(N√L)`) if a slot frees; it is the only shortlist item not started.
5. Rewrite `report.md` around: a model that provably does not contract still saturates at 8; the
   cause is angular dilution; scale control relocates the optimum without raising the ceiling; and
   here is whether per-token depth demand is heterogeneous enough to exploit.

## 9. What I want from you specifically

- **6.1 (MLP ratio) and 6.3 (Muon under weight-tied composition)** are the two where I think you can
  save me from a design error I cannot test in 24h.
- Is there any published measurement of **per-token argmin-depth dispersion** on AR text? If the
  distribution is known to be concentrated, item 1 is a null and I should say so up front.
- Does anyone report **final-loop-only loss vs dense per-loop loss** for a looped LM on AR text at
  matched tokens? That is 6.9 and it is the single largest untested axis in my design.
