# State and plan — 2026-08-23 19:25 · written for attack, not for reassurance

**Maximum disclosure.** Every claim below carries its weakest point in the same sentence. Where a
number is single-seed, post hoc, or confounded, that is stated inline rather than in a caveats
section at the bottom where nobody reads it. If you only have time to attack one thing, §6 lists what
I would attack.

Deadline 23:59 MSK; ~4h30 remain. `report.md` is 6,421 lines. Seven compute streams running.

---

## 1. What the report claims, ranked by how much survives scrutiny

**(a) Saturation without convergence — strongest, needs no seeds.** The unit state drifts
**logarithmically**: one-parameter log-drift fits R² 0.986 against a convergent power law's 0.748 with
two; 0.18 rad of angular motion still accumulates between loops 129 and 384; ρ = 1.6227 at loop 2
(seeded, reproducible). Yet CE stops improving at loop ~8. *Weakness:* an **untrained** model of the
same shape drifts faster (C 0.3084 vs 0.1549), so non-convergence is **architectural, not learned** —
the rebuttal to the task's DEQ premise is a statement about the architecture, not about training.

**(b) Late motion is fully visible to the readout and still useless.** Readout gain flat 207 → 226
while ‖Δu‖ falls 117×; `E`'s condition number is 132, so there is no null subspace to hide in.
*Weakness:* one checkpoint, one eval set.

**(c) Eight interventions on the dynamics; one lowers the loss; none widens the useful band.**
Inter-loop norm +0.744 · scale clock +1.36 (non-finite by loop 39) · gated injection at the field's
own α +0.247 · loop-cycled LoRA **−0.086** · radial clamp ~0 · convex gate null · ε-scaling null ·
norm penalty −0.030 (wins ppl, the only arm with ρ<1). **Now nine**: duo-causal W=2 is a null (below).
*Weakness:* the LoRA row's significance depends on a **post-hoc** rank restriction — see §3.

**(d) Per-token depth demand is real, large, and unreachable — and §4.7e now says why.** Oracle
headroom 0.2008–0.2032 nats, split-half reliability **0.866** against a null of **0.0007**; 27.9% of
tokens want depth > 32. *Weakness:* both halves come from one forward pass, so it establishes argmin
is a stable feature of that token's curve, not that it replicates under an independent draw.

---

## 2. Today's new results, with the numbers

### §4.7e — the depth-key rank collapse. The strongest new thing, and the one I most want attacked.

Per-token depth-key stream across r = 32; effective rank = participation ratio of singular values:

| model | L0 | L1 | L2 | mean pairwise cos |
|---|---|---|---|---|
| 90M control | **1.83** | **1.58** | **1.52** | 0.911 / 0.972 / 0.973 |
| 46M no-renorm | 1.73 | 1.61 | 1.52 | 0.949 / 0.968 / 0.971 |
| **untrained (null)** | 2.73 | 2.61 | 2.45 | 0.800 / 0.812 / 0.823 |
| `od_control` | 1.60 | 1.34 | 1.25 | — |
| `od_lora_r4` *(different operator every loop)* | 1.61 | 1.38 | 1.33 | — |

**out of 32.** So: a token's depth keys live in ~1.6 dimensions; the collapse is present at
initialisation; **training makes it worse**; and the one arm that provably applies a different
operator per depth raises it by **0.01–0.08**.

**Attack surface, stated for you.** (i) Effective rank via participation ratio is *one* choice — a
different rank statistic could give a different number, though it would have to change it by ~20× to
matter. (ii) Measured on 2×128 tokens; the per-token cosines are averaged over 256 positions, but the
SVD is subsampled to 64 tokens. (iii) It is a statement about **keys**, and a mixture over **states**
is not identical to attention over keys — §4.7c's static-state-mixture null is separate evidence, but
the two are not the same object. (iv) The causal direction is unproven: the rank collapse could be a
*consequence* of depths not mattering rather than the cause.

### §4.21 / §4.21b — the one replicated CE positive, and it is a capacity result

Five in-job pairs: rank 2 **+0.0941** · rank 4 MPS −0.0514 · rank 4 T4 **−0.1011** · rank 8 Kaggle
**−0.0733** · rank 4 Kaggle sw90 **−0.1172**.

- rank ≥ 4 (n=4): mean −0.0857, 95% CI **[−0.1322, −0.0393]**, excludes zero.
- **All five (n=5): mean −0.0498, 95% CI [−0.1545, +0.0549] — covers zero.**

**The restriction to rank ≥ 4 is post hoc.** Defensible (rank is a dose variable; rank 2 is a
different treatment, not a bad draw) but **not pre-registered**, and there is **no dose–response above
the threshold** — rank 8 sits inside rank 4's spread. And **88–95% of every arm's gain is present at
`r = 1`**, where `branch = 0 mod 4` and cycling is **logically inert** (verified: pinned vs cycling
give max|diff| = 0.000e+00 at r=1, 1.05 at r=4). Both band edges identical in **5 of 5** pairs.
**It improves the block, not the looping.**

### §4.15 — every band claim was one number doing two jobs

`plateau_mid = √(onset × end)`, verified 9/9. Decomposed:

| comparison | onset | end |
|---|---|---|
| μ=18 dense → sw90, seeds 0/1 | **8 → 8** (unchanged) | 16 → **24** |
| μ=40 dense → sw90 | **16 → 24** | 40 → 48 |
| operator diversity (both jobs) | **8 → 8** | **20/24 → 20/24** |
| 90M control → norm penalty | **6 → 6** | **17 → 14** *(earlier)* |

So the surviving positive is, at its best-replicated schedule, **purely an *end* effect** — the model
degrades later, it does not improve further. And the norm penalty — the **only** arm with ρ<1 —
shortens the end while leaving onset alone, which is what the dilution account predicts.

### §4.3 — XSA's phenomenon transfers in direction, not in weight

`cos(y_i, v_i)` rises monotonically with loop index from loop 2 (+0.009/+0.060/+0.065 over loops
2→64). **But the untrained null inverts the reading:** untrained reaches **0.83–0.85** by loop 64
against the trained model's **0.35**. The bias is architectural and **training already suppresses most
of it**. So it is *not* a mechanism for `cos(du_t,du_{t−1}) → 0.9999`, which is what it was offered as.
*Layer 0's apparent 0.82 → 0.55 fall is a **loop-1 artifact** — at t=1 the state is `h₀+e` with no
context, so attention is nearly all self by construction.*

**Three nulls of this kind now disagree, and that is the interesting part:** training *slows* the drift
(0.308 → 0.155), *suppresses* the self-attention bias (0.85 → 0.35), and **reduces** depth-key
diversity (2.73 → 1.83). **The one property training makes worse is exactly the one that would let a
model use its own depth.**

### Duo-causal W=2 — complete at both seeds, clean null, and it corrects me

| seed | ΔCE_best | ΔCE@1 | onset | end | mid |
|---|---|---|---|---|---|
| 0 | **+0.0093** | +0.0226 | 8 → 8 | 20 → 20 | 12.6 → 12.6 |
| 1 | **−0.0115** | −0.0221 | 8 → 8 | 20 → 20 | 12.6 → 12.6 |

Sign reverses; both inside the 0.0150 floor; band identical to the digit. **The registered falsifier
("reverses between seeds ⇒ not reported") fires.**

**Two disclosures against myself.** (1) At 18:52 I read the *first* eval (step 244, 500k tokens) as
"tracking negative at both seeds" (+0.1632, +0.0360) and used it to argue against spending V100. The
arms converged to a null; **that was noise**, and the recommendation survives for a different reason
than I gave. (2) The `dc_w3` arms sit in the same logs with **1 and 0 evals**; reading the latest row
naively would have compared a 244-step arm to a 1707-step control and produced a fake **+1.1174
catastrophe**. Caught by checking step counts.

**Still open on that arm:** read (b), `cos(du_t,du_{t−1})`, needs the returned checkpoints. Per the
19:02 gate, **a CE null without the cosine is a null on a mechanism that may not have engaged.**

---

## 3. Running now — every arm with its pre-registered read

| job | arms | pre-registered read | ETA |
|---|---|---|---|
| `tlab-duocausal-s0/-s1` | control · W=2 ✅ · W=3 · `dg_norm` | `cos(du)` three cases; band; dose-response W1→2→3 | ~20:30 |
| `tlab-diversity-control-s0` | control · lora_r4 · **pin-0** | does pinning recover the gain ⇒ capacity result | ~20:00 |
| `tlab-pin2-control-s0` | control · **pin-2** | the *clean* pin — branch 2 never trains at r=1 | ~19:50 |
| `tlab-xsa-s0` | control · XSA | CE down, band unmoved — **amended to near-null on CE** after the null | ~20:05 |
| `tlab-recmethod-s2` | dense · **sw90** | the artifact: first weights of the recommended method | ~20:50 |
| Kaggle `tlab-lora-scaleup` | control · lora_r4, **12M/arm** | does the positive survive ~5× budget | ~21:45 |

**`dg_norm` is the highest-stakes cell and is a joint test of §4.7e** (registered 19:22, before it
runs): effective-loops-mixed **< 1.5** ⇒ selector again, **no CE claim**; **≥ 1.5 with no gain** ⇒
§4.7e's rank collapse is the binding constraint (the pointed prediction); **≥ 1.5 with a real gain** ⇒
**§4.7e is wrong and the per-token headroom is reachable** — the most consequential outcome available.

---

## 4. Open confounds, none of them resolved

1. **pin-0 and pin-2 are in different jobs** — comparing them is a difference-of-differences.
   Measured cross-job drift 0.0074–0.0334 against a ~0.10 effect, so 3–10× smaller. Not zero.
2. **Diversity, pin-2, XSA, recmethod are single-seed.** Duo-causal has two.
3. **The Kaggle scale-up packs its own val shard** (~89% overlap with local, §4.2).
4. **`n_loop_eff` fixed at 24** while schedules run at μ 18 and 40 — in-job pairs unaffected,
   cross-schedule comparisons carry it.
5. **The 90M clipping question is unanswerable from the artifacts** — the kernel logged only
   `{step, val_curve}`. Endpoint gradients (2.07 control vs 2.85 penalty, both above the clip, within
   1.4×) bound it as *open but unsupported at 90M*. It feeds D3.
6. **Duo-causal is handicapped and I said so before the result:** RoPE is depth-invariant so the extra
   keys carry **no depth marker**; the pathway has **zero dedicated capacity**; and Think-at-Hard
   applies it as a fine-tune on a pretrained 1.7B backbone, not from scratch at 9M/3.5M tokens.
   **A null here bounds the windowed, markerless, zero-parameter, from-scratch form — not the
   mechanism.**

---

## 5. The plan, with the branch points named

| window | work | branches on |
|---|---|---|
| 19:25–19:50 | `pin2` lands → clean capacity-vs-diversity | pin2 ≈ cycled ⇒ §4.21 becomes a capacity claim outright |
| 19:50–20:10 | `diversity` + `xsa` | XSA band-widens ⇒ that becomes the headline and everything reprioritises |
| 20:10–20:40 | W=3 + `dg_norm`, both seeds; read (b) on returned checkpoints | `dg_norm` gain ⇒ §4.7e is wrong |
| 20:40–21:10 | `recmethod` → eval + `state_dynamics` on the first weights of the recommended config | |
| 21:10–22:00 | Kaggle LoRA at 12M | reverses ⇒ §4.21 gets the norm-penalty scale caveat |
| 22:00–23:00 | second end-to-end read of today's ~2,000 new lines (**running now as an independent pass**); final push; HF card if the shipped artifact changes | |
| 23:00–23:59 | hard stop on edits | §1 is the author's |

---

## 6. What I would attack if I were you

1. **§4.7e's causal direction.** I claim depth keys are rank-1.6 *therefore* mixing cannot work. The
   arrow could run the other way: depths collapse *because* the loss never rewarded distinguishing
   them. Nothing here separates those, and the untrained null does not — it shows the collapse
   pre-exists training, not that it *causes* the mixing failure.
2. **The LoRA positive's post-hoc threshold.** Over all five arms the interval covers zero. If you
   think the rank-2 exclusion is unjustified, the project's only replicated CE positive evaporates.
3. **Everything new tonight is 2.5–3.5M tokens.** This project's own regularity (§4.6b) says a 2.5M
   effect can shrink 12× and flip character by 90M. The Kaggle 12M arm is the only scale check running
   and it is one arm, one seed.
4. **§4.15's decomposition is a re-reading, not new data.** It is sharper, but no new measurement
   backs it — the tuples were always there, which also means the report reported the wrong statistic
   for weeks.
5. **The abstract now leads with §4.7e**, a result four hours old that has had one independent check.

---

## 7. Not done, and not going to be by me

- **§1** — empty. The author's, and the only carrier of criterion 1.
- **The wandb key is unrotated.** Scrubbing the repo did not un-send it.
- **Both repos are private.** GitHub `Arsenii324/tlab-looped-transformer`, HF
  `Arsen4ikVar/tlab-looped-transformer`. Verified: only `main`, 0 tags, secret scan clean over 121
  commits, and the identity gate passes **against the downloaded artifact** (|diff| 0.0020 vs chance
  8.3178). **The visibility flip is the author's.**
- **D3** — which checkpoint ships. My recommendation remains the control: the penalty arm is 88%
  loop-1 damage, narrows the band, is the only arm that converges, and now carries an unresolvable
  clipping confound.
