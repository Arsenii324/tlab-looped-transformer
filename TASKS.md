# TASKS — live stack of things I have taken on and NOT finished

Scope, deliberately narrow: **commitments I made in response to the user's messages, that are still
open.** Not a plan, not a backlog, not everything that could be done. `QUEUE.md` is the exhaustive
ledger; this file is the short list I can un-stack. **Delete a line when it is done — this file only
ever shrinks unless the user adds to it.**

Last updated 2026-08-23 ~15:05.

---

## Open — mine

| # | task | from | state |
|---|---|---|---|
| ~~T17~~ | **DONE 2026-08-23 18:20.** First end-to-end read of `report.md` (5,448 lines). **12 defects found, 3 serious, none findable by grep**: sec3.5 restated both withdrawn claims under their own withdrawal blocks; sec8 carried the cross-job number sec4.17 replaced with one of opposite sign; sec4.2 still said loop gain was "flat" after its own paired correction. All fixed; recorded as sec6.0 row 33 + unknown-known #26 | unknown-known #7 | **CLOSED** |
| T14 | **Learned depth gate** (`gate_scalar` +32 / `gate_state` +14,336) — restored as the best remaining candidate for a positive after correcting the 'E1 is E2's upper bound' error (E1 is a LOWER bound; a global weighting cannot reach a per-token signal) | reviewers x2 | **not launched**, local GPU busy with od arms |
| ~~T11~~ | **Gemini's `tlab-operator-diversity` DS job errored on `ModuleNotFoundError: tokenizers`** — its `cmd` installs it but `system.log` shows the install never ran. Local run produced only `od_control`. **Decide: relaunch or drop.** Note `lora_cycle` costs +408,576 params (4.51%) and is a fixed per-branch table — if it wins, §3.5's zero-param scale argument does not cover it | Gemini session, 16:23 | **not relaunched** |
| ~~T12~~ | **`od_lora_r2`/`od_lora_r4`/`od_depth_gate` arms never completed** — the learned depth gate (+448 params) is the one the reviewers twice called the only remaining chance at a positive | Gemini + reviewers | local GPU is free |
| T1 | **Read the remaining project `.md`s in full.** Still unread end-to-end: `LOG.md` (2098), `PLAN.md` (246), `BRIEFING.md` (366), `RUNS.md` (299+), `STATE_FOR_REVIEWER.md` (259), `REVIEW_NOTES.md` (188), `DECISIONS.md`, `METHODS.md`, `INTERVENTIONS.md`, `README.md`, `HANDOFF.md`, and the 3 paper-summary `.md`s | "Read all .md's that you expect to have in context, in full" | **not started.** I read `QUEUE.md`, `DATASPHERE_NOTES.md`, the task statement, `anthropic-prompting.md`, `compute-yandex-datasphere.md` + the `.py` files I touched |
~~T4~~ Kaggle `tlab-seed-extension` harvested 15:37 -> n=4 result WITHDRAWS the headline (see T10).
| T7 | **§8 writing items still owed**: W1 (MLA × LLA), W2 (LoopMTP aggregation conflict), W4 (STARS Pre-Sandwich), W5 (decomposition tables for remaining §4 sections) | reviewer, logged in QUEUE | not started |
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
