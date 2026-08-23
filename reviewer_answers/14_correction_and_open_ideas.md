# Reply — 2026-08-23 ~17:15 · a correction to §4.20, and the honest map of what is and isn't running

Self-contained. Leading with the correction, because it lands on the result I most recently told you
was the strongest thing in the report.

---

## 1. §4.20's degenerate collapse is largely a measurement artifact. I was wrong about it.

You called §4.20 "the strongest thing in it" and proposed a pre-test: add per-loop diversity and see
whether it breaks the collapse. I ran it, and then ran the stronger version, and the answer sent me
back to the statistic itself.

**Neither form of diversity breaks it:**

| condition | cos@8 | cos@32 | cos@64 |
|---|---|---|---|
| autonomous baseline | 0.9952 | 0.9998 | 1.0000 |
| per-loop **scalar** gains, σ=1.0 (large enough to zero/invert gains) | 0.9774 | 0.9955 | 0.9997 |
| per-loop **operator** diversity — LoRA branches cycled by loop, 168 tensors randomized | 0.9950 | 0.9998 | **1.0000** |

A genuinely different operator at every loop leaves the number *identical to baseline*. That is not
what "the same map applied repeatedly causes collapse" predicts, so I checked what the statistic
actually measures.

**It compares layer outputs. In a pre-norm residual stack every output is `x + branch(x)`, so all
three share the same residual `x`.** Comparing each layer's *own contribution* instead:

| loop | cos(outputs) | **cos(increments)** | ‖increment‖ / ‖h‖ |
|---|---|---|---|
| 8 | 0.9994 | **0.1806** | 0.0348 |
| 32 | 1.0000 | **0.1536** | 0.0096 |
| 64 | 1.0000 | **0.1387** | 0.0053 |

**The layers are not doing the same thing — their contributions sit at cos ≈ 0.14–0.18.** The outputs
agree because each layer moves the state by 0.5–3.5% of its norm. `cos → 1.0000` is mostly
arithmetic.

**What this costs and what it buys.** It costs the claim that the block degenerates to one direction,
and every argument built on it — including the one I gave you that "all layers collapse
architecturally" belongs in the report's spine. What it buys: **the entire conditioning /
branch-diversity family is now closed without a training run.** Per-loop gains, loop-cycled LoRA,
IterAdaLN-style modulation — all were proposed to fix a thing that is substantially an artifact. Two
forward passes retired an architectural direction that would have cost a training slot and up to
3.8% of the parameter budget. Your pre-test suggestion is what surfaced it; it just resolved one
level deeper than either of us framed it.

**What survives §4.20:** the increments *do* decline mildly with depth (0.18 → 0.14) and the
increment-to-state ratio falls sharply (0.035 → 0.005). The latter is §4.3's dilution restated
per-layer. That is a real, smaller result.

## 2. Your withdrawal-criterion objection was right in principle, and I tested it. Same verdict.

You argued 0.0541 is an *unpaired* floor and too conservative for a *paired* in-job difference, and
proposed a t-interval on the four paired differences instead. Correct in principle, so I ran it
rather than defending the original criterion:

- 95% t-interval at n=4: **[−0.1478, +0.0558]** — covers zero → **withdraw**.
- Your premise was that the seed spread is ~0.0143. That was the **n=2** value; the actual n=4 sd is
  **0.0640, 4.5× larger**. The effect is **1.44 SE** from zero, not the ~10 SE projected.

So the better-specified test gives the same answer. That is strictly better than surviving only the
lenient test, and the withdrawal now doesn't rest on a yardstick you can object to.

## 3. Did the AI-sourced ideas engage the mixture-over-depths literature? Yes — and its conclusion held up

One instance did read it properly (LoopFormer 2602.11451, Think-at-Hard 2511.08577, PALBERT, MoDr,
MixerLoop) and reported: **nobody combines hidden states across loop depths at readout.** PALBERT is
nearest and only for the halting head (`[h_t, h_{t−1}]`); MoDr mixes *branches* at one depth; deep
supervision reads out at every depth with separate losses, never a combination. I did not
independently verify that survey, and flag it as theirs rather than mine.

The gap it identifies is real, and this project has now tested it three ways — all null:
readout mixture (§4.7c, −0.0015/−0.0000/−0.0003 vs a 0.0527 floor), annealed-trajectory variant
(§4.7a), and oracle-depth KV cache (§4.8b, −0.0096, reverses sign by query depth 24). Four
independent instrument classes now fail to reach §4.7's headroom.

**One correction to a claim made about my own work:** it was asserted that the static mixture (E1)
being null makes the learned gate (E2) dead, since E1 is E2's upper bound. **That is backwards, and
the objection to it is correct** — E1 sweeps a *global* weighting, E2's function class contains E1 as
the constant case, so E1 is a **lower** bound. And §4.7's whole finding is that depth demand is
*per-token* (oracle cv 0.798 vs path-length cv 0.068), which a global weighting structurally cannot
reach. E1's null is what the hypothesis predicts. **The learned depth gate is still open and is still
the best remaining candidate for a positive.**

## 4. What is running, and what is not

| stream | state |
|---|---|
| DS `tlab-deep-full` | **EXECUTING**, ~step 12k/19.5k. Reframed per your note: read as a §4.16b replication (does the band still track μ_rec=40 under full training?), **not** as an annealing test — its `sw75` axis is withdrawn |
| DS `tlab-anchor-tokenkey` | **DONE, harvested.** ERROR status was cosmetic (§6.0's stderr pattern); all 6 arms completed |
| Kaggle `tlab-seed-extension` | **DONE, harvested** — produced the withdrawal in §1 above |
| local `run_operator_diversity` | **RUNNING**, arm 2/4 (`od_lora_r2`). Note: §1 above has now largely retired this experiment's motivation; letting it finish since it's already paid for, but it is no longer load-bearing |
| DS `tlab-operator-diversity` | **ERRORED** at 4 min — `ModuleNotFoundError: tokenizers`. Its config *does* install it in `cmd`, but `system.log` shows the install never ran. Not relaunched |

**Ideas not currently running, ranked:**

1. **Learned depth gate** (`gate_scalar` +32 params / `gate_state` +14,336) — the only remaining
   candidate for a *positive*, and §3 above restores its standing. Not launched.
2. **Two-depth cache** (concatenate K/V from depths 4 and 16) — the cheaper half of §4.8b's
   experiment; if the union beats both singletons, attention is doing selection. Not run.
3. **Gradient checkpointing → recover the μ_rec=56/44 arms that OOM'd** — documented as free and
   unused; the memory route to the deep schedules the task is actually about.
4. **§2's DEQ reframe** — DEQ's headline claim is *constant memory*, not compute, so the task's
   "convergence wastes compute" objection attacks an axis DEQ isn't optimising. Pure text, not written.
5. **Three-instance scaling regularity** (§4.6b) — our norm penalty's 12× shrinkage, our annealing's
   schedule-axis reversal, exact-ZOH's 14×. Caveat I'd state: two of three are ours, and the external
   two share an author, so it is not three independent groups.
6. ~~Conditioning / branch diversity~~ — **closed by §1 above.**

## 5. On `n_loop_eff`

Verified: `depth_init` scales by `1/√(2·n_loop_eff)` with `n_loop_eff` **fixed at 24** in every
checkpoint, while schedules ran at mean depth 18 (`U[4,32]`) and 40 (`U[32,48]`). Scaling ratios
1.15× and 0.77× against what the schedule implies. **Within-schedule comparisons are unaffected** —
in-job pairs share the same wrong init on both arms — so no in-job result in this report is
confounded. Cross-schedule comparisons (§4.16b, §4.17's μ=18 vs μ=40) carry it as a limitation. Owed
as one sentence in §6.0b; not yet written.
