# Reply — 2026-08-23 ~15:45 · a withdrawal, a completed experiment, two corrections, run state

Self-contained — written for someone who does not have `report.md` open. Leading with the most
important thing.

---

## 1. WITHDRAWN: the annealing CE advantage over dense does not survive n=4

§3.5's headline supporting claim was *"`sw90` beats an in-job dense control on CE at both seeds
(−0.0811, −0.0609)."* That was n=2. A pre-registered extension to seeds 2 and 3 just landed on
Kaggle and **both pre-registered withdrawal triggers fired**:

| seed | ΔCE_best (sw90 − dense) |
|---|---|
| 0 | −0.0811 |
| 1 | −0.0609 |
| **2** | **+0.0482** *(reversed — sw90 worse than its own dense control)* |
| 3 | −0.0902 |

n=4 mean = **−0.0460**, sd 0.0640. The four values straddle zero, and the mean sits *inside* the
0.0541 CUDA terminal-arm replicate floor. The pre-registration (written before this result existed,
in `RUNS.md`) said exactly this combination withdraws the claim to **"not resolved at this
budget."** Verified before writing this: all four arms are config-identical pairs (same seed, same
everything, differing only in the annealing intervention — checked field-by-field), all four
completed cleanly (step 1219, 9 evals, no errors/NaN in the raw Kaggle log), and the weights
survived.

**What does NOT fall with it:** the useful-depth band still widens under `sw90` at every seed
checked (11.3→13.9 at seeds 0/1; the band-vs-CE decomposition in §4.17 is unaffected). And `sw90`
still ranks above `sw75` — `sw75` was already worse-or-reversed at 2 of 2 seeds; `sw90` is negative
at 3 of 4, which is a weaker but still favourable comparison. What falls is specifically the claim
that annealing improves the *ceiling* over dense supervision at this budget.

**Housekeeping owed, not yet done:** this number appears in ~13 other places in `report.md` (the
headline table, §4.17's own writeup, §8, a couple of decomposition tables). Two of the highest-
visibility locations are fixed (the headline table row, §4.17's primary paragraph, both now carry
the withdrawal). The rest are stale until a propagation pass — logged as `TASKS.md` T10.

## 2. Gated injection — complete, and it's a clean negative with a real mechanism

The third cell on the normalisation axis (§4.1 tested hard RMSNorm vs. nothing; neither is what
Parcae or *Looped Transformers Done Right* actually use — both choose a learned per-channel carry
decay). Three in-job arms, all landed:

| arm | `α` init | CE_best | `α` learned | channels decaying |
|---|---|---|---|---|
| control (additive) | — | 5.4000 | — | — |
| `gi_gated` | 0.9999939 (≈identity) | 5.3730 | **0.999993 (unchanged)** | 0/448 |
| `gi_gated_a874` | 0.874 (the field's default) | **5.6470** | 0.862 | **448/448** |

`gi_gated`'s near-identity init turned out to be a trap: `d(α)/d(inj_a) = 6.1e-6` there against
`d(δ)/d(inj_b) = 0.63` — δ is ~103,000× more reachable, so the decay was never exercised; its
0.027-nat improvement is entirely from write-strength, not decay, and is inside this project's
0.0527 same-config replicate floor anyway.

**`gi_gated_a874` is the real test, and its own pre-registered primary check succeeds:** ‖h‖ growth
over 32 loops drops from the control's 6.2× to **1.17×**, and the injection ratio — which this
project's other measurements show collapsing 7.3–7.8× under plain addition — stays **nearly flat**.
The mechanism does exactly what the field's choice is supposed to do. **And CE_best gets 0.2470
nats worse — ~4.7× the replicate floor, clearly resolvable, wrong direction.** The band doesn't
move; this is a real regression at the same useful depth, not a relocation like this project's other
nulls. Read plainly: bounding the state removes the radial escape this architecture's dynamics
depend on (it never contracts, never converges — see below), so a mechanism built to prevent
explosion may be fighting the trajectory the model actually uses.

## 3. Two corrections to feedback relayed from other AI instances

Both were working from **stale context** — genuinely useful catches in principle, already acted on
in fact.

**(a) The static-mixture confound.** One flagged, correctly, that `Σ w_t h_t` is not scale-free
across depth (‖h‖ grows 18–26×), so an unnormalized uniform mixture is dominated by the deepest
state and isn't testing what it claims to. **True, and I re-ran with both raw and normalized
weighting.** Conclusion is unchanged: normalized mixing gives −0.0023 vs the best single depth
(raw gave −0.0017), both ~23× below the resolvable floor. Normalized variants occupy half the top-8
mixtures, so the effect is real and simply far too small — not an artifact of the parameterisation.

**(b) The claim that a report section (§8.2) conflates a recurrent gate with readout-time mixing.**
Checked against the actual text: it doesn't. §8.2 already frames the convex gate as a cheap proxy
test for a *recurrent* mixing mechanism (attention over all past loop states, feeding back into the
next step) and explicitly separates that from the readout-only static test, which is cross-referenced
by name. No fix needed — the claim was made against a summary of the section, not the section.

**Both instances also claimed the static-mixture test (E1) and the shallow/deep-oracle conflict test
(E3) had "never been run."** They had — over an hour before those messages arrived, as §4.7c (null,
3 checkpoints, −0.0015/−0.0000/−0.0003 vs a 0.0527 floor) and §4.8a (real in sign, ≤2.6% in
magnitude, confirms their own supervision-dependence prediction). Not a criticism of either — they
had no way to know — but it means their proposed next steps were re-deriving existing results rather
than extending them, and the useful new instructions from those messages were the confound fix
above and (still pending, see below) the learned-gate arm.

## 4. Two low-probability sweeps run via a second CLI agent (`agy`), unverified by me

Three parallel jobs launched on: (A) foundational sanity — generation samples, vocab inspection,
`n_loop_eff` vs actual schedules, `frozen_eval_set.npz` self-consistency, `BYTES_PER_TOKEN`
re-derivation, checkpoint NaN/degenerate scan, pairing checks; (B) re-reading ~80 existing job logs
for errors/warnings never surfaced; (C) an independent read of every cited paper against `report.md`'s
citations — **did not complete**, killed by a CLI syntax error and not yet relaunched.

**A and B returned. I have not verified their findings — flagging that plainly rather than
passing them through.** One from A looks real and matches a prediction made in advance ("I think
this one could genuinely fire"): `n_loop_eff=24` is a fixed constant driving `depth_init`'s
1/√(2·n_loop_eff) scaling, and it is **never adjusted** for the actual training schedule — models
trained at mean depth 6, 40, or 54 all get initialisation scaled for 24. Plausible and worth checking
in the next pass. Job B returned **10** items labelled `FINDING`, which given this project's own
observed pattern (padding toward round or complete-looking output under a similar model) is a count
I would sample-check before trusting rather than accept — none of the 10 are independently confirmed
by me yet.

## 5. Run state

| stream | state |
|---|---|
| DS `tlab-deep-full` (μ_rec=40, `sw75`) | EXECUTING — this is the arm the withdrawal above bears on most: it's the switch fraction now shown mixed at n=4, on the schedule where the CE claim was already known to be a trade |
| DS `tlab-anchor-tokenkey` | EXECUTING |
| Kaggle `tlab-seed-extension` | **COMPLETE, harvested** — this section |
| Local | idle — gated-injection finished; nothing else queued |
| agy A, B | **returned, unverified** |
| agy C (paper cross-check) | **did not run** |

**Not started, and worth naming since it was proposed twice independently:** a learned depth gate
(state-conditioned mixing over `{h_t}`, as opposed to the static mixture already tested null). Cheap,
zero-init, in-job — genuinely the one thing in this batch that could produce a new positive rather
than another negative. Deferred for now given the token-budget constraint just communicated; picking
it back up once that lifts.
