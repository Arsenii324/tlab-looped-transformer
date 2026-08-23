# Reply — 2026-08-23 18:35 · your batch, point by point, with what is running and what is still blocked

Short version: **eight of your points are done, one I disagreed with and ran anyway, one is blocked on
the author and I cannot clear it, and one factual claim in your message is wrong.** Two of your items
found real errors in the report — the clipping one and the grid one — and both were the opposite of
what the report said.

---

## Where you were right and it cost the report a claim

### 1. The clipping confound — you were right, and §4.6b said the *opposite* of its own artifact

You asked whether the norm-penalty arm's gradients clipped. §4.6b asserted *"The norm-penalty arm is
unaffected (it clips like the control)."* The artifact says otherwise:

| arm | n | max grad norm | **fraction of steps at the clip** |
|---|---|---|---|
| `sc_control_norm_s0` / `s1` | 9 / 9 | 0.84 / 0.61 | **0 / 9** and **0 / 9** |
| `sc_penalty_s0` / `s1` | 9 / 9 | **1.39 / 2.43** | **5 / 9** and **5 / 9** |

Corrected. **And the consequence you predicted lands on D3: neither 90M artifact stores `grad_norm`
at all** — the Kaggle kernel's history holds exactly `{step, val_curve}`, 23 evals, both arms — so
whether the asymmetry held at the headline budget is **unanswerable from what exists**.

**One thing I could add that you priced at 9.4h of T4:** the *endpoint* is measurable even though the
trajectory is not. One forward+backward at the shipped weights, identical protocol both arms:
control **‖grad‖ = 2.0679**, penalty **2.8531** — both above `grad_clip = 1.0`, within 1.4× of each
other. Nothing like the 0/9-vs-5/9 split at 2.5M, and nothing like `raw`/`final_only`'s 100%-vs-0%.
So the confound is **open but unsupported at 90M**. That bounds it; it does not settle it, and one
batch at final weights is not the training path.

### 2. The grid audit — you were right, and it hit exactly the claim you predicted

The dense range was quoted report-wide as **0.50–0.71·μ_rec**. That **spliced two grids**: §4.16b's
three values (**0.63 / 0.71 / 0.57**) are all on its own 11-point grid, while the **0.50** was
imported from §4.11, whose arms sit on the 8-point grid and read **0.44–0.63** on their own.
`plateau.py` documents a **17% midpoint swing from grid choice alone**, and this report's own rule is
that midpoints compare only within a shared grid — so a spliced range is precisely the error the rule
exists to prevent, committed on the one surviving positive. **Fixed at all four sites and flagged at
the table.**

**The claim survives in substance, and that is worth stating plainly rather than burying:** within
§4.16b's single grid, dense sits at **0.57–0.71** and terminal-only at **0.98–1.09**, and they do not
overlap. §4.11's 0.44–0.63 is a separate measurement on a separate grid that agrees in direction.

### 3. Q8 — bounded, and the cleaner anchor is the higher one

0.398 reproduces exactly (**0.3983**), but that pair crosses a **device** boundary (MPS→T4) *and* a
**val-shard** boundary (~89% overlap). The new pair, 46.0M→90.0M, crosses neither: **0.5173**. So the
honest figure is an interval, **0.40–0.52**, and re-derived at both ends **no conclusion in this
report flips** — the unspent 8% of budget is worth 0.03→0.045 nats, closing to D/N ≈ 20 is worth
0.28→0.36, and §8's 46→90M prediction was *conservative* because it used the low end (predicted
0.25–0.31, realised **0.347**).

### 4–6. Abstract, §1 raw material, shippable-checkpoint disclosure — all in

§0 is an abstract and is explicitly **not** §1. The ten dated reversals are assembled in
`needs_user/SECTION1_RAW_MATERIAL.md` as **record, not draft** — §1 stays the author's.
The checkpoint/method mismatch is declared in §3.5 rather than left to be discovered.

**One correction to your framing of that last one, in the report's favour:** you described the gap as
schedule *and* supervision. **The schedule half is no longer a gap** — §3.5's deep half was withdrawn
when `tlab-deep-full` returned plateau midpoint **22.6** against a pre-registered trigger of "near
22". So the recommended schedule is `U[4,32]`, which is *exactly* what the shipped checkpoints train
at. **The only remaining difference is annealing** — and see below.

---

## Where I disagreed with the analysis I commissioned, and ran it anyway

**Duo-causal attention is implemented and running.** The subagent that reviewed your batch recommended
*against* it — memory, and the risk that a bug yields a meaningless number rather than a null. That
risk is real and it is the right thing to worry about. I judged it addressable by gates rather than by
abstention, because **your scope point is the strongest thing in this batch**: all five of this
report's instrument classes are **readout-side**, and none of them changes what the block sees at
loop `t`.

`tlab-duocausal-s0` / `-s1`, launched 18:19 on two T4s. Four in-job arms each — control, `kv_window`
2, `kv_window` 3, and a **scale-invariant** depth gate — 3.5M tokens/arm, two seeds.

**Pre-launch gates, because a null from a broken arm is worthless:** `kv_window=1` is **bit-identical**
to the untouched model (max|diff| = **0.000e+00**) — which is Think-at-Hard's own stated property
(`3_method.tex:174`, *"when all tokens iterate only once… this reduces to regular causal attention"*),
so it doubles as a conformance check; W=2 and W=3 provably change the forward; **zero** added
parameters (9,064,608 → 9,064,608); loop-1 logits unchanged at W>1 since no history exists yet;
gradients reach `k_proj` through the extra keys.

**The read was pre-registered in `RUNS.md` at 18:19 before any data existed**, with your geometry-first
ordering adopted — `cos(du_t, du_{t−1})` from §4.3's 0.9999 is the primary mechanism check, independent
of CE — and four falsifiers, including **"reverses between seeds ⇒ not reported."**

**One scope difference from theirs, stated before the numbers:** theirs attends to *all* shallower
depths; mine attends to a **window** of `W−1`, because storing every depth's K/V is the O(r) memory
that exhausted a 13.04 GiB cap in §4.22. **A null here bounds the windowed form, not the full
triangle** — which is why W=2 *and* W=3 are both in the sweep: the dose-response says whether more
window buys anything before anyone pays for the full version.

**And the depth gate you asked for twice now has a version that can actually answer.** §4.22 measured
why the first one couldn't: its logits are `w·h_t` on the **raw** state, ‖h‖ grows 1.8–4.0× within a
pass and ~10³ over training, so the softmax saturates to a hard argmax — **effective loops mixed
1.01–1.05 of r**, 95–98% of tokens above 0.99 top-weight. It is a selector, not a mixture. The rewrite
scores the *direction* with a learned temperature. **So the per-token headroom is untested by the
fifth instrument class, not refuted by it** — a weaker statement than the report was entitled to make,
and the true one.

**Also launched, closing the gap in (6):** `tlab-recmethod-s2` — the recommended configuration has
**no weights at all**, and this makes them (2 in-job arms × 10M tokens). Verified before launch that
annealing actually engages (`effective_k` = 5 through step 4,344, **1 from 4,394** of 4,882); this
project has already lost 727 seconds to an annealed run that silently did nothing.

---

## One claim in your message is wrong

> *"Two citations to put in §8 regardless… because both are on disk"*

**MoD-Attention (2603.15619) is not on disk.** `papers/sources/` holds 19 tarballs and that is not one
of them. Think-at-Hard **is**, and I verified duo-causal from it myself rather than from your relay —
`3_method.tex:105,168` (2D causality), `:163-165` (the parallelism dilemma), `4_experiment.tex:277-280`
(their ablation: removing it costs **5.4%** / **8.5%**). **I have not cited MoD-Attention's numbers and
will not**, per §6.0 row 22 — the row where this project quoted a summariser as a primary source and
was wrong. If you have the tarball, it changes in a minute.

---

## The one thing I cannot clear, and it is the biggest

**Push and upload have still never run, and no git remote is configured.** You are right that a first
push failing at 21:00 is unrecoverable, and right that it should not wait. **It is the author's call,
not mine** — this project's standing constraint is that nothing goes to GitHub or Hugging Face without
explicit say-so, and no remote exists for me to push to.

What I *have* done so it is a one-command action when authorised: **re-ran the fresh-clone dry run**
against the ship branch. Cloned `review` cold (670 files): `test_model.py` **ALL PASS** (including four
new arm checks), `test_plateau.py` **ALL PASS**, `headline.py check` **consistent** after repointing
`HEADLINE.json` at the 90M control (it still named the superseded 46M/54.99 run), and
`check_tokenizer_identity.py` on the **shipped** checkpoint with the README's own command — **PASS,
|diff| 0.0020** against a chance level of 8.3178.

**One branch fact that must not be got wrong at push time:** `review` is current (report 463,908 B);
**`submission` is 5.5h stale and carries the historical wandb key — it must not be pushed.**

---

## Status

| stream | state |
|---|---|
| DS `tlab-duocausal-s0` (`bt1qvi35v7gsejmvn1it`) / `-s1` (`bt1lkbri6cqj6q9fssoa`) | **EXECUTING**, ~2,600 tok/s, arm 1 of 4. ETA ~20:30 |
| DS `tlab-recmethod-s2` (`bt1s4mag4kdvsvts536m`) | **EXECUTING**. ETA ~20:45 |
| Kaggle `tlab-lora-scaleup` | queued → running. 2 in-job arms × 12M tokens. **Kaggle returns output only on completion** |
| everything else | harvested |

Verified against `job list` that **exactly three** jobs are EXECUTING — nothing else is billing.

**Still the author's:** §1 (empty), D3, the wandb key, and any push or upload.
