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
| T1 | **Read the remaining project `.md`s in full.** Still unread end-to-end: `LOG.md` (2098), `PLAN.md` (246), `BRIEFING.md` (366), `RUNS.md` (299+), `STATE_FOR_REVIEWER.md` (259), `REVIEW_NOTES.md` (188), `DECISIONS.md`, `METHODS.md`, `INTERVENTIONS.md`, `README.md`, `HANDOFF.md`, and the 3 paper-summary `.md`s | "Read all .md's that you expect to have in context, in full" | **not started.** I read `QUEUE.md`, `DATASPHERE_NOTES.md`, the task statement, `anthropic-prompting.md`, `compute-yandex-datasphere.md` + the `.py` files I touched |
| T2 | **Harvest `tlab-deep-full`** and fill §4.17's deep half | standing | EXECUTING, step 11800/19531 (60%) as of 15:45; plan is to let it finish (~16:45-17:00) rather than early-harvest, see reviewer_answers/13 sec6. Read against the pre-registration in `RUNS.md` 13:30 — it is **sw75**, which §3.5 narrowed away from |
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
