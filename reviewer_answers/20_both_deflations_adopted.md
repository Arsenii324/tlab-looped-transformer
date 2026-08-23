# Reply — 2026-08-23 18:55 · both deflations adopted, the reframing is better than the claim it replaces, and one arm is already tracking negative

All three of your corrections are in. **Two of them make the report's newest positive weaker and the
third makes it sharper, and the third is the one that matters** — following your own argument through
turns an unexplained positive into an explained one, which is worth more.

---

## 1. The rank-2 exclusion is post hoc, it decides the claim, and it now says so

Reproduced your arithmetic exactly:

| set | n | mean | sd | 95% t-interval | |
|---|---|---|---|---|---|
| rank ≥ 4 | 4 | −0.0857 | 0.0292 | **[−0.1322, −0.0393]** | excludes zero |
| **all five** | 5 | −0.0498 | 0.0843 | **[−0.1545, +0.0549]** | **covers zero** |

§4.21 now states the restriction as a **post-hoc judgement, in the same breath as the interval**, and
names the parallel you drew: *this is structurally the annealing withdrawal six hours earlier — n
arms, one reverses, the interval straddles — except there the reversal was kept and here it is
excluded.* The defence I'd offer is the one you already anticipated, and it is a real distinction
rather than a rescue: **rank is a dose variable, so rank 2 is a different treatment, not a bad draw
of the same one.** That justifies a threshold; it does not make the threshold pre-registered.

**And you are right that the dose–response is absent, which is the strongest available defence and
the data does not supply it.** Rank 8 (−0.0733) sits *inside* rank 4's spread (−0.0514, −0.1011,
−0.1172). More rank buys nothing. That is now in the text, because a monotone rank→effect curve would
have made the threshold nearly unarguable and its absence is informative in the other direction.

## 2. Your point (2) is the important one, and following it through changes what the section claims

You caught me writing, of the LoRA arms, that *"a loop-cycled adapter does act at r = 1, so unlike the
gate there is no inertness argument."* **That is backwards on the point that matters.**

At `r = 1` the branch index is `0 mod 4 = 0` — **one branch, applied once.** The adapter is active, so
the block really does have more capacity; but **cycling is a claim about different operators at
different depths, and there is only one depth.** So the mechanism is inert even though the parameters
are not.

**Verified rather than argued.** I implemented the control you asked for and used it as an
identity check first: with LoRA `B` randomised identically in both models, branch-pinned and cycling
give **max|diff| = 0.000e+00 at r = 1** and diverge by **1.05 at r = 4**.

So with **88–95%** of every LoRA arm's ΔCE_best already present at `r = 1`, ~90% of the effect is
**added capacity, not operator diversity.** It composes with two facts already in the section:
Δgain sits inside every measured floor, and the band is identical to its own control in **5 of 5**
pairs. Three independent statements, one conclusion.

**The headline sentence is now yours, near-verbatim, because it is better than mine:**

> Eight interventions. One lowers the loss, replicated across three platforms — and ~90% of that gain
> is present at loop 1, where the cycling is inert. **It improves the block, not the looping.** None
> widens the useful band.

## 3. The control that settles it is running, not deferred

`tlab-diversity-control-s0`, launched 18:53 on a T4. Three in-job arms, 2.5M tokens each, seed 0:
control · cycled LoRA r=4 · **LoRA r=4 with the branch index pinned to 0** — parameter count
**identical to the digit (9,473,184 both, verified)**, diversity zero.

`cond_fixed_branch` is two lines in `src/model.py` and was gated before launch as above.
**Prediction registered before the number exists, and it is yours:** on the r=1 decomposition, the
pinned arm should recover most of the −0.10, making §4.21 a capacity result. ETA ~20:00.

## 4. The parameter-scaling defence — I can't use it, and I'd rather say so than borrow it

Your Schwethelm arithmetic (A=41, α=0.235 → 0.0129 predicted vs 0.0857 measured, 6.6×) is the right
objection to pre-empt. **But that paper is not in `papers/sources/`** — only bibliography entries in
two other tarballs — and §6.0 row 22 is the row where this project quoted a summariser as a primary
source and was wrong about it in three separate documents. **So no exponent of theirs is quoted.**

What replaces it is better anyway: **the pinned-branch control is the same test run internally**, on
this model, at this budget, needing no external constant. If it recovers the gain, "you just added
capacity" is *confirmed* rather than rebutted — and I'd rather have that answer from my own control
than an order-of-magnitude argument from a fit I haven't read.

## 5. Your point (4) is now a registered hard gate, written before the data

Adopted verbatim as policy, because it is this project's own §5 rule applied to the two instruments I
just caught failing. In `RUNS.md` at **18:51**, before either arm produced a number:

- **`dg_norm` gate:** effective-loops-mixed **≥ 1.5** or **no CE claim is made from that arm at all**.
  Below that it is a hard selector again (the broken one measured 1.01–1.05 of r), scale was not the
  binding constraint, and it reports as a second instrument failure rather than a depth result.
- **duo-causal gate:** `cos(du_t, du_{t−1})` must move from **0.9999**. If it does not, a CE null is
  *a null on a mechanism that did not engage* — a different finding, and the only one of the two worth
  anything.

Both gates can only **disqualify** an arm's CE, never rescue it. That asymmetry is why writing them
early costs nothing and buys the thing §6.0 rows 3, 5 and 16 are all about.

## 6. Interim, flagged as interim: duo-causal is tracking NEGATIVE at both seeds

First eval, step 244 (500k tokens), each against its own in-job control:

| seed | control best | W=2 best | Δ |
|---|---|---|---|
| 0 | 6.3481 @r8 | 6.5113 @r4 | **+0.1632** |
| 1 | 6.3291 @r8 | 6.3651 @r8 | **+0.0360** |

Worse at both, band not widened. **This is one eval at 500k tokens and my registered read is at arm
end**, so it is not a verdict — but the sign agrees across seeds and it is the direction of the
"regresses ⇒ kill" falsifier. One implementation sanity check does hold: `dc_w2` runs at 2,262–2,345
tok/s against the control's ~2,600, **≈11% slower against a predicted +9%**, so the extra attention is
real work and not a silent no-op.

**Practical consequence:** this is what I'd have spent a V100 on, and on current evidence I won't.

## 7. Running now

| job | what | ETA |
|---|---|---|
| `tlab-duocausal-s0` / `-s1` | control · W=2 · W=3 · scale-invariant gate, 2 seeds | ~20:30 |
| `tlab-diversity-control-s0` | **capacity vs diversity**, 3 in-job arms | ~20:00 |
| `tlab-recmethod-s2` | weights for the method §3.5 recommends, which had none | ~20:50 |
| Kaggle `tlab-lora-scaleup` | the LoRA positive at 12M/arm, ~5× its screening budget | ~21:45 |

Four T4s and one Kaggle GPU; verified against `job list` that exactly these are EXECUTING.

**Submission is live and verified** (private, both): GitHub `Arsenii324/tlab-looped-transformer` —
only `main`, **0 tags**, `submission` never pushed, secret scan clean across 121 commits. HF
`Arsen4ikVar/tlab-looped-transformer` — the identity gate passes **against the downloaded artifact**,
|diff| **0.0020** vs chance 8.3178.

**§1 is still empty and still the author's.** You are right that it is the only remaining item
carrying criterion 1 and the only one that cannot be parallelised. The dated reversals are assembled
in `needs_user/SECTION1_RAW_MATERIAL.md` as record, not draft.
