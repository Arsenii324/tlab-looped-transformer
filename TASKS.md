# TASKS — live stack of things I have taken on and NOT finished

Scope, deliberately narrow: **commitments I made in response to the user's messages, that are still
open.** Not a plan, not a backlog, not everything that could be done. `QUEUE.md` is the exhaustive
ledger; this file is the short list I can un-stack. **Delete a line when it is done — this file only
ever shrinks unless the user adds to it.**

Last updated 2026-08-23 18:22.

---

## Open — mine

| # | task | from | state |
|---|---|---|---|
| ~~T17~~ | **DONE 2026-08-23 17:50.** First end-to-end read of `report.md` (5,448 lines). **12 defects found, 3 serious, none findable by grep**: sec3.5 restated both withdrawn claims under their own withdrawal blocks; sec8 carried the cross-job number sec4.17 replaced with one of opposite sign; sec4.2 still said loop gain was "flat" after its own paired correction. All fixed; recorded as sec6.0 row 33 + unknown-known #26 | unknown-known #7 | **CLOSED** |
| ~~T14/T11/T12~~ | **CLOSED 18:19.** The learned depth gate ran and §4.22 measured *why* it fails: its logits are `w·h_t` on the RAW state, so the softmax saturates to a hard argmax (effective loops mixed **1.0 of r**). It cannot express a mixture. The **scale-invariant** rewrite (`state_norm`) is now launched — see T18 | reviewers x2 | **CLOSED, superseded by T18** |
| **T18** | **IN FLIGHT — `tlab-duocausal-s0` / `-s1`, launched 18:19 on two T4s.** 4 in-job arms each (control, duo-causal W=2, W=3, scale-invariant gate), 3.5M tok/arm, seeds 0+1. **Read pre-registered in `RUNS.md` 18:19 with four falsifiers, before any data existed.** ETA ~20:30. Harvest: (a) plateau band vs in-job control, (b) `cos(du_t,du_{t−1})` post-hoc on the returned checkpoints — outputs named explicitly, not globbed | the readout-side/recurrence-side gap | **running** |
| **T19** | **Full-budget replication of the LoRA positive.** §4.21 states plainly that no full-budget LoRA arm exists and that this is the check that decides it — the norm penalty shrank 12× and flipped character between 2.5M and 90M. Kaggle has ~7h quota. **Kaggle is NOT early-stop-safe**, so it must fit end to end | §4.21's own scope caveat | **next action** |
| T1 | **Read the remaining project `.md`s in full.** Still unread end-to-end: `LOG.md`, `PLAN.md`, `BRIEFING.md`, `RUNS.md`, `STATE_FOR_REVIEWER.md`, `REVIEW_NOTES.md`, `DECISIONS.md`, `METHODS.md`, `INTERVENTIONS.md`, `README.md`, `HANDOFF.md`, 3 paper summaries | "Read all .md's… in full" | **not started.** Read so far: `QUEUE.md`, `INDEX.md`, `OPS.md`, `TASKS.md`, `DATASPHERE_NOTES.md`, **`report.md` end to end (T17)**, reviewer replies 14–18, the task statement |
| T7 | **§8 writing items still owed**: W1 (MLA × LLA), W2 (LoopMTP aggregation conflict), W4 (STARS Pre-Sandwich), W5 (decomposition tables for remaining §4 sections) | reviewer, logged in QUEUE | not started |
| ~~T20~~ | **Fresh-clone dry run — DONE 18:15.** Cloned the ship branch `review` cold (670 files): `test_model.py` ALL PASS (incl. 4 new arm checks), `test_plateau.py` ALL PASS, `headline.py check` consistent after repointing at the 90M control, and `check_tokenizer_identity.py` on the SHIPPED checkpoint **PASS, \|diff\| 0.0020**. **`submission` is 5.5h stale and must not be pushed** | submission gate | **CLOSED** |
~~T8~~ done, §4.20.

## Blocked on the user

| # | task |
|---|---|
| U1 | **§1** — the idea narrative. Reserved, empty. Stimulus drafts in `needs_user/section1_drafts_STIMULUS.md` |
| U2 | **Rotate the wandb key.** Scrubbing the repo did not un-send it. `needs_user/ROTATE_WANDB_KEY.md` |
| U3 | **D3 — which checkpoint ships.** Control (38.86 ppl, the config §3.5 describes) vs norm-penalty (37.52, but 88% loop-1 damage and a narrower band) |
| U4 | Any GitHub / HF push. **Note: `git push --tags` or `--mirror` WILL be rejected** — a tag carries 1.83 GB of >100 MB blobs; branches are clean (`OPS.md`) |

## Closed today (kept one cycle, then deleted)

- **T2 harvested 17:23.** `tlab-deep-full` SUCCESS, 30.0M tokens. **Pre-registered falsifier fired**: plateau mid 22.6 against a trigger of "near 22"; mid/μ_rec 0.57 is dense-range. Deep half of §3.5 withdrawn. Survives: band [16,32], the deepest useful band in the project.

- **T16 closed clean.** Seven distinct eval grids exist; the load-bearing comparisons (§3.5's n=4 annealing extension, §4.17's original seeds, §4.18's falsifier) all share one 11-point grid. μ_rec=40 arms use a different grid but are only compared to each other. No cross-grid midpoint comparison found. Recorded in §4.15.

- **T13** §6.0b now states the `n_loop_eff=24` limitation (in-job pairs unaffected; cross-schedule carry it).
- **T15** §4.20's retraction propagated to the two sections that cited it as a live architectural finding.
- **T14** is effectively running as `od_depth_gate` (arm 4/4) — read pre-registered in `RUNS.md` 17:40 with two caveats (uniform-mixture start ≠ control; its plateau is over mixture-window size, not depth).

- **T3 harvested (16:51).** `tlab-anchor-tokenkey` reported ERROR but all 6 arms completed (§6.0's known wandb/pip-stderr-marks-ERROR pattern); results.json recovered. **§4.18 falsifier: partial fail** — onset invariant at 12 across k=5/3/2 as predicted, but band mid 17.0/17.0/19.6, so §4.18 downgraded to shallow-edge-only. **Token-keyed vs fraction-keyed RESOLVED**: token rule wins by −0.2208, ~4× floor, largest supervision effect measured; §3.5 now token-keyed on evidence.
- **Gemini's two probes reviewed, bug-fixed, re-run, and written up** as §4.8b (oracle cache null) and a §4.8a extension (within-token displacement). Alignment bug logged as §6.0 row 32.

- **T10: annealing-withdrawal propagation, complete.** Checked all 8 remaining mentions of '0.0811' after the first pass -- all are either raw per-seed data (seed 0 genuinely was -0.0811; still true) or already inside a withdrawal/context block. Fixed: headline table, sec3.5 primary claim, sec8 dissociation-reopened paragraph, sec8 'main result' circular defense, gain_decomp table footnote.

- Process every point across **all** reviewer messages, not just the latest → cumulative ledger R1–R60 in `QUEUE.md`
- Reviewer-facing answers as separate files in a subfolder → `reviewer_answers/00`–`12`
- The reviewer's five unknowns → all measured, `reviewer_answers/10`
- R45 (exit rules on an annealed checkpoint) → §4.7a, matched pair
- Deep repo review S1–S11 → 11/11 closed
- Correction-propagation audit → 11/11 applied
- Claims→instrument traceability → applied (ρ reseeded, `gain_decomp` share relabelled, `cumulative_exit` motivation)
- Layer Duplication → **decided against**, with the reason recorded (its own reconciliation means it bears on no claim of ours)
