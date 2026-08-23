# Reply — 2026-08-23 14:05 · wide-angle status

Your four yes/no questions are answered first because **all four are done**, and **three came out
against the prediction** — including one that reversed a claim of mine an hour after I made it.

---

## The four proposals: all RUN, three refuted the prediction

### (a) `B` on an untrained model — **DONE. Prediction refuted, backwards.**
You predicted `B_init ≈ 0`, near-orthogonal increments, rising with training, tying §4.16c to §4.12's
emergence curve.

| | B (loops 1–18) | B (all 32) | first step |
|---|---|---|---|
| trained, dense | 0.4384 | 0.5082 | 0.1053 |
| trained, terminal-only | 0.5283 | 0.6192 | 0.1143 |
| **untrained, same architecture** | **1.9929** | **2.2215** | **0.5778** |

**An untrained model travels 4.5× further and has zero capability.** Training *reduces* angular path
~4×. So `B` is path length, not "useful computation", and the sentence *"terminal-only buys more
useful angular computation"* was **withdrawn**. Your Huginn shape (untrained 110–115°/step → trained
42–59°) is directionally the same phenomenon: training makes steps *smaller and more coherent*. Here
the first untrained step is ~34°, the trained ~6°.

### (b) `B` over a fixed range — **DONE. The confound was real.**
Ratio at each arm's own `k*`: **1.384 / 1.417**. Over a fixed 1–18 range: **1.205 / 1.245**. Part of
the effect was "the optimum sits later", exactly as you said.

### (c) `B` across readout modes — **DONE. Confirmed, and far bigger than supervision.**

| arm | B (1–18) | vs control | loop gain |
|---|---|---|---|
| control (RMSNorm readout) | 0.4405 | — | 0.1056 |
| norm penalty | 1.7914 | 4.1× | 0.2522 |
| final-only norm | 5.0421 | 11.4× | 0.2170 |
| **raw (scale-visible)** | **5.7427** | **13.0×** | 0.2214 |

**Readout mode moves the geometry 13×; supervision moves it 1.2×.** But `B`'s ordering is *not* an
ordering of quality — `raw` has 13× the path and a worse CE than `final_only` — and the untrained
control forbids reading path length as merit. Stated as magnitude-of-influence only.

### (d) Argmin depth by position — **DONE. Refuted at this scale.**
Mean oracle depth by position: 21.60 (0–32) → **20.73** (224–256). **Position explains 0.06%** of the
variance (loop-1 entropy explains 0.71%). The drift is −0.88 loops and runs *opposite* to
"context is paid for in unrolls". This **strengthens** §4.7: depth demand is explained by neither the
trajectory nor position, so the obvious nuisance escape is closed.

### And your chord-vs-arc correction was right, and it reversed the sign
You said hooking all three layers was worth one pass — not for periodicity, but because `B` is a chord
whose error depends on within-loop curvature, and the arms might curve differently. **They do:**

| | dense s0/s1 | terminal s0/s1 | terminal ÷ dense |
|---|---|---|---|
| B chord (1/loop) | 0.4310 / 0.4223 | 0.5188 / 0.5245 | **1.203 / 1.242** |
| **B arc (3/loop)** | 1.4756 / 1.4701 | 1.1743 / 1.2262 | **0.796 / 0.834** |
| arc ÷ chord | **3.42 / 3.48** | **2.26 / 2.34** | |

**Terminal-only travels ~20% LESS at true resolution, not 20% more.** The real quantity is within-loop
curvature: dense's path is 3.4× its chord, terminal's 2.3×. **Terminal-only makes the within-loop
trajectory straighter and gets further while travelling less.** That is a better mechanism than the
one it replaces, and everything depending on `B`'s direction is withdrawn.

---

## 1. What the submitted model actually is

**Params: 9,064,608 total.**

| component | params | share |
|---|---|---|
| embedding = lm_head (**tied**), V=4096 × H=448 | 1,835,008 | 20.2% |
| block layer 0 / 1 / 2 — MLP (SwiGLU, I=1344) | 1,806,336 each | 19.9% each |
| block layer 0 / 1 / 2 — attention (GQA 4 q / 2 kv, d_head 112, QK-Norm) | 602,336 each | 6.6% each |
| 6 × RMSNorm inside the block (2 per layer) | 448 each | — |
| `final_norm` (RMSNorm) | 448 | — |
| **`h0`** — a learned vector, shape **(448,)** | 448 | — |
| `loop_norm` | **None** (`state_renorm=False`) | — |

**Forward pass, exactly:**
1. `e = embed(input_ids)` → `[B,T,448]`.
2. **`h = h0.expand(B,T,-1) + e`** — `h0` is a *single learned 448-vector* broadcast over batch and
   position, and **the first injection is additive unconditionally**, regardless of `inject_mode`
   (`model.py:413`). Worth knowing: "no injection" arms still get this one.
3. For `t` in `range(n_loops)`:
   - `h_in = self._inject(h, e) if t > 0 else h` — additive: **`h + e`**, i.e. the *same* embedding
     re-added every loop. (`concat` uses a 2H→H adapter; not used in the headline config.)
   - `h_new = block(h_in, cos, sin)` — **all three DecoderLayers, pre-norm, in sequence**.
   - optional gate (unused here), then `h = h_new`.
   - `loop_norm` is **None** in the headline config, so nothing normalises the carried state.
   - **`states.append(h)` — ONE sample per loop, after all three layers.** Your assumption was right,
     and it is exactly what made `B` a chord. `intraloop_states.py` hooks the three layers instead.
4. Readout at supervised loops: `readout_mode="norm"` → `final_norm(h)` then the **tied** head.
   Because RMSNorm is scale-invariant, **the readout sees direction only** — this is the fact §4.3,
   §4.6 and §4.16c all rest on.

**Supervision schedule, as code** (`src/train.py`, added today):
```python
_k_eff = (train_cfg.supervise_k if train_cfg.supervise_k_final is None
          or step < train_cfg.supervise_switch_frac * total_steps
          else train_cfg.supervise_k_final)
sup_idx = sample_supervise_idx(n_loops, _k_eff, rng)
# sample_supervise_idx: ALWAYS includes the last loop, plus k-1 uniformly sampled from the rest
loss = torch.stack([F.cross_entropy(logits[i], y) for i in sup_idx]).mean()
```
`n_loops` is redrawn per step from `U[min_train_loops, max_train_loops]`.

**Train vs eval:** the only conditional on `self.training` is exploration noise (`explore_noise=0` in
every reported arm), so **the eval forward is identical to training's** apart from `no_grad` and the
loop count swept. Eval uses chunked non-overlapping windows; §4.2 reports the sliding-window
alternative (1.6436 vs 1.6938 bpb) and does not use it anywhere else.

## 2. Which checkpoint ships — still open, and here is what decides it

| candidate | ppl (protocol-matched) | plateau | the case against |
|---|---|---|---|
| 90M control | **38.86** | [6,17] | not the annealed method §3.5 recommends |
| 90M + norm penalty | **37.52** | [6,14] | wins ppl, but **88% of its loop-gain is loop-1 damage** and it *reverses* character between 2.5M and 90M |
| `tlab-deep-full` (μ=40 annealed, ~32M) | lands ~17:18 | — | fewer tokens by construction; expected ~+0.5 nats |

Full config of the two 90M runs: `H=448, 4 q-heads / 2 kv, d_head 112, I=1344, 3 layers/loop, V=4096,
seq 256, batch 8, lr 3e-3 → 3e-4 cosine, wd 0.05, grad-clip 1.0, AdamW β=(0.9,0.95), fp32,
`state_renorm=False`, additive injection, depth_init, `n_loop_eff=24`, loops `U[4,32]`,
`supervise_k=5` (**no annealing** — both predate it), 43,944 steps × 2,048 tok = 90.0M.
**Decision is the user's (`QUEUE.md` U3).** Whichever ships, the card states the 12× shrinkage.

## 3. In flight / queued

| item | lands | what each outcome changes | cut first? |
|---|---|---|---|
| `tlab-deep-full` | ~17:18 | fills §4.17's deep half. If its plateau mid ≈ 22 (dense-like) the deep half is withdrawn | **no — last artifact** |
| `tlab-hyper-screen` wd arms | ~15:00 | closes the last unscreened axis; LR already settled (3e-3 optimal) | yes, first |
| local annealed run (unlaunched) | ~40 min | **unblocks R45** — exit rules on an annealed checkpoint, the missing cell of §4.7 | second |
| W6 anchor account, W8 token-keyed rule (writing) | any time | framing; no compute | — |

**If the machine died now:** the report stands. Everything load-bearing is measured, the headline is
protocol-matched, and only the deep artifact and §4.7's missing cell would be lost.

## 4. Where the report and the code disagree

You found two; here are those plus what I found checking:
1. **§3.2's five-axis description is written for `state_renorm=True`** — the config no headline run uses.
2. **§3.4 said the architecture has no loop conditioning** while §4.17 *is* loop conditioning through
   the loss. **Fixed today**, and §3.4 now uses annealing as its best example.
3. **`n_loop_eff=24`** is a constant for a schedule *no* headline arm runs (they use μ_rec 18 or 40).
   Documented in `DECISIONS.md` as known-wrong-and-deliberate, but §3 does not say so.
4. **"No injection" arms still receive one additive injection at t=0** (`model.py:413`). §4.1 describes
   `inject_none` as no injection; it is *no re-injection*.
5. **§4.3's "increments near-parallel, cos → 0.9999"** is a loop-boundary property; at layer resolution
   they are **anti-correlated (−0.368)**. Scoped today.

## 5. Another week vs comparability-doesn't-matter

**Another week** (keeps every number comparable): 3 seeds everywhere; screen the MLP ratio (64% of
block params, never screened); re-run §4.1's five axes in the no-renorm regime, since `inject_mode`,
`depth_init` and `truncate_bptt` were all decided under a config that was later abandoned; a
non-recurrent baseline at matched tokens; `B_L` across train-at-L with weights actually saved.

**Right now, if comparability didn't matter** (a different and shorter list): raise `seq_len` to 512
(§8.0b argues depth demand scales with context, so 256 may cap the effect being studied); raise vocab
to ~16k so perplexity is comparable to published work; **switch the readout to final-only norm**,
which at 2.5M beat the control by 0.098–0.517 and is 12% damage-driven; anneal supervision from the
start of the schedule rather than as a fraction. **The gap between the lists is the point:** the
week-list is about *believing* the numbers, the now-list is about *making them larger*, and I would
take the first.

## 6. The weakest point — the thing I would attack first

Not in §6, and not a caveat I have written anywhere:

> **The method's effect size is the same order as the corrections I have applied to it today.**

`sw90` beats its in-job control by **0.061–0.081 nats**. Today alone: a re-zeroed-vs-raw noise
comparison was withdrawn; argmin was retired after 63/82 curves failed; a cross-job reference flipped
the sign of the μ=40 result; the angular budget was corrected three times and its **sign reversed**;
two of my pre-registered predictions were falsified. Every one of those was 0.02–0.2 nats, or a sign.

The method survived each — it is the one claim with an in-job control at two seeds *and* a 4× budget
replication. But a grader is entitled to ask why they should believe a 0.07-nat effect from a project
that has demonstrated, repeatedly and in writing, that its own instruments were wrong by that much
until checked. **My honest answer is that the checking is the evidence** — the effect survived
instruments that killed several neighbours — but that is an argument about process, not about the
number, and it is the argument I would least like to defend.

The second-weakest, and it is close: **`sw90` is one arm at one schedule.** `sw75` fails at seed 1,
the μ=40 arms are damage-driven, and "both endpoints improve" is specific to μ_rec=18. The
recommendation rests on a narrower base than §3.5's confidence implies.
