# Reply — 2026-08-23 13:05 · your five unknowns, each measured; plus run state

You closed with *"five things I don't know that would change what I say."* All five are now measured
rather than recalled. Two of them change the report, and one of those you were right to suspect.

First, on your retraction: **accepted, and it cost nothing, because nothing was built on it.** The
only change the arc measurement produced in the report is §4.3's scope — `cos → 0.9999` labelled a
loop-boundary property, −0.3681 labelled layer-resolution, the two described as different objects
rather than a tension. That edit was already in before your retraction arrived. The mechanism, the
efficiency ratio, the ρ derivation and the `3/√3` null are not in the report and will not be. Your
four objections are right, and the second one — attention–MLP anti-correlation being a static
property of pre-norm blocks — is the one that would have embarrassed us, because it means the
instrument cannot separate "what looping did" from "what a pre-norm block is."

---

## Q1 — which grid produced the headline plateau `[6,17]`?

**Dense, every integer 1..64.** `checkpoints/full_control90_kaggle/eval_full_control90_kaggle.json`,
n=64 points, written by `src/eval.py` (which sweeps `range(1, max_loops+1)`). Recomputed just now:

| checkpoint | dense 1..64 | sparse {1,2,4,8,12,16,24,32} |
|---|---|---|
| 90M control | plateau **(6,17)** mid 10.1 onset 6 · best CE 3.6599 @10 | (8,16) mid 11.3 onset 8 |
| 90M norm-penalty | plateau **(6,14)** mid 9.2 onset 6 · best CE 3.6250 @8 | (8,12) mid 9.8 onset 8 |
| 46M no-state-renorm | (5,14) mid 8.4 onset 5 | (8,12) mid 9.8 onset 8 |

**The good news:** the two headline arms `[6,17]` vs `[6,14]` are *both* dense, so that comparison —
the one §3.5 rests on — is grid-matched and valid.

**Your worry is still correct elsewhere:** the same two models read `[8,16]` and `[8,12]` on the
sparse grid, which is a different qualitative story (identical onset, indistinguishable bands). Any
section quoting a sparse-grid plateau is not comparable to the headline's dense one. The 46M row is
also the one `plateau.py`'s own docstring uses as its cautionary example, so the 17% swing is
documented — but documented in the module, not enforced at the call sites. **Owed: a grid column in
every plateau table.** Logged.

## Q2 — do §4.3's norms transfer to the 90M artifact? **No. You were right to ask.**

Measured on a fixed 4×256 validation batch, all three checkpoints, CPU, just now:

| checkpoint | ‖h‖@1 | ‖h‖@8 | ‖h‖@16 | ‖h‖@32 | ‖h‖@64 | @64/@1 |
|---|---|---|---|---|---|---|
| 46M no-state-renorm *(§4.3's source)* | 1659.5 | **6639.7** | 10674.1 | 17605.9 | **30270.8** | 18.2× |
| **90M control** *(the artifact)* | 466.6 | **2334.4** | 3977.3 | 6920.6 | **12424.4** | 26.6× |
| 90M norm-penalty | 4.4 | **17.5** | 28.8 | 49.6 | **89.4** | 20.3× |

§4.3's quoted 6630 @8 and 30097 @64 reproduce on the 46M model (6639.7 / 30270.8 — different eval
batch, same numbers). They are **2.4–2.8× wrong for the 90M control**, and ~380× wrong for the
norm-penalty arm, which the penalty drives down by more than two orders of magnitude.

**What survives and what doesn't.** The *relative* dilution account survives cleanly: growth from
loop 1 to 64 is 18.2× / 26.6× / 20.3× — the same shape in all three. What does not transfer is every
**absolute** norm statement and, more importantly, **the radial-clamp levels**, which were chosen as
`{|h1|, |h8|, |h16|}` on the 46M model. Applying those levels to the shipped checkpoint would clamp
it to roughly 2.5× its own natural scale. §4.6's clamp result is a statement about the 46M model and
must say so.

## Q3 — was the local annealed run launched?

**Yes — and it had silently produced nothing for 727 seconds until an hour ago.** Worth stating
plainly because it is the second instance of the task statement's own prediction.

`train.py` saves a checkpoint at exactly one site: inside the eval block. `run_anneal_local.py` was
hand-written with `eval_every_tokens = 1_250_000`, against the reference checkpoint's `312_500`. At
`batch_size=8` that puts the only save 610 steps away, while a 240 s chunk (the MPS-corruption
workaround) reaches ~250. So no checkpoint was ever written, `resume` stayed `False`, and each chunk
retrained from step 0 — three chunks, `steps_logged=0 last_step=-1` each time, an empty output dir.

Two fixes, both in: `train.py` now checkpoints on the `max_seconds` break, so the chunk length —
a hardware workaround — can no longer invalidate a run; and `run_anneal_local.py` now **derives its
config from the reference checkpoint** instead of re-typing it, so "matched pair" is true by
construction rather than by transcription. Relaunched 12:42, `PRE-FLIGHT` confirms it differs from
`sd_dense_k5_s0` in `run_name` and the two annealing fields only. ~83 min cap. This is R45's missing
cell.

## Q4 — does §3.5 still overclaim at μ_rec=40?

**It does not say "better on both axes"** — it carries a PROVISIONAL banner and an explicit
tension table that already concedes *"At μ_rec = 40 it costs 0.030 nats to move the useful band from
~23 loops to 32–64."* So the trade is stated.

**But your sharper point stands: it states the cost and not the decomposition.** Recomputed:

| arm | CE@1 | CE_best | plateau | ΔCE_best vs dense | ΔCE@1 vs dense | verdict |
|---|---|---|---|---|---|---|
| `da_mu40_dense` | 5.6513 | 5.4658 @24 | (16,40) mid 25.3 | — | — | — |
| `da_mu40_sw90` | 5.7262 | 5.4394 @32 | (24,48) mid 33.9 | −0.0264 | **+0.0749** | damage-driven |
| `da_mu40_sw75` | 5.8263 | 5.4466 @48 | (32,64) mid 45.3 | −0.0192 | **+0.1749** | damage-driven |

Your +0.0749 / +0.1749 are exact. And there is a detail that makes it worse than "a trade": these
are CUDA runs, where this project's measured replicate floor is 0.0150 (dense) and **0.0541
(terminal)**. Both ΔCE_best values fall **inside** the terminal floor; both ΔCE@1 values clear it.
So at the schedule the task's premise is actually about, **the ceiling improvement is not resolvable
and the loop-1 damage is** — the deeper band is bought, not free. §3.5 owes that sentence, in §3.5,
not §6.

## Q5 — what is in §1?

The reserved placeholder, verbatim: *"Reserved for the author's own account of how the approach was
arrived at — the task explicitly grades this separately from implementation, and explicitly asks
that it not come from an LLM. Not filled in here."* Three stimulus drafts sit in
`needs_user/section1_drafts_STIMULUS.md`, marked not-for-the-report. It is the user's slot and it is
still empty.

---

## Run state, since you asked

| stream | what | state |
|---|---|---|
| **DataSphere** | `tlab-deep-full` — deep artifact, μ_rec=40 annealed, every step ≥32 loops | **EXECUTING**, started 07:29, ~5.3 h in. Returns **curves only** (the `outputs:` defect). Fills §4.17's deep half. DS cannot be peeked — cancelling *is* the harvest |
| | `tlab-hyper-screen` | **SUCCESS, harvested** — see below |
| | `tlab-node-probe`, `tlab-term-seed1`, `tlab-deep-anneal2` | terminal, collected |
| **Kaggle** | `tlab-loop-fullrun`, `tlab-loop-normpenalty` | both finished 2026-08-22 evening; nothing queued. Both 90M checkpoints are local |
| **Local (MPS)** | `local_anneal_sw75_s0` | relaunched 12:42 after the Q3 bug; ~83 min |

**New since reply 09 — the weight-decay screen landed.** Six arms, ~875 s each, in-job:

| arm | lr | wd | CE@1 | CE_best | onset | ΔCE_best vs ref | ΔCE@1 | Δgain |
|---|---|---|---|---|---|---|---|---|
| `hp_wd0.01` | 3e-3 | 0.01 | 5.4532 | **5.3502** | 8 | **−0.0190** | −0.0179 | +0.0011 |
| `hp_wd0.1` | 3e-3 | 0.1 | 5.4652 | 5.3686 | 8 | −0.0005 | −0.0059 | −0.0053 |
| `hp_ref` | 3e-3 | 0.05 | 5.4711 | 5.3692 | 8 | — | — | — |
| `hp_wd0` | 3e-3 | 0.0 | 5.4821 | 5.3935 | 8 | +0.0243 | +0.0110 | −0.0132 |
| `hp_lr6e-3` | 6e-3 | 0.05 | 5.5308 | 5.4424 | 12 | +0.0732 | +0.0597 | −0.0135 |
| `hp_lr1e-3` | 1e-3 | 0.05 | 5.5545 | 5.4725 | 4 | +0.1033 | +0.0834 | −0.0199 |

wd=0.01 beats the inherited 0.05 by 0.0190, just clear of the 0.0150 CUDA-dense floor; wd=0 is
clearly worse; 3e-3 is confirmed optimal. **The column that matters is Δgain: every arm is within
±0.02 of zero, and onset is 8 for all five well-trained arms.** Hyperparameters buy absolute loss and
do not touch depth exploitation — which is a clean negative for §6.0b, and mildly reassuring for the
whole report, since it means the inherited LR was not quietly setting the depth results.

---

## One thing I want on the record

The task statement says: *"Современные ллмки умеют решать сложные открытые математические задачи, но
при этом запросто возьмут **неправильный токенизатор** или **забудут сохранить чекпойнт**."*

Both named failure modes happened in this project. The tokenizer one: the README instructed a grader
to run `train_tokenizer.py` first, which would have overwritten the shipped vocab and made every
released checkpoint evaluate at chance (~8.32 nats) — caught only because you asked D1. The
checkpoint one, twice: every DataSphere job discarded its weights, and the bug in Q3 above is the
same failure in a different costume, found today. Criterion 2 is explicitly *"не теряться в коде"*,
and the honest form of that is a failure table with these in it (§6.0, 23 rows) rather than a claim
that we didn't lose track — we did, three times, and each was caught by an artifact colliding with
an action rather than by review.

## Owed, from this reply

1. Grid column in every plateau table (Q1).
2. §4.3 and §4.6 scoped to the 46M model; clamp levels re-derived if they are to be quoted against
   the shipped checkpoint (Q2).
3. §3.5 gains the μ_rec=40 decomposition sentence, with the noise floor (Q4).
4. §6.0b gains the wd screen and its Δgain null.
