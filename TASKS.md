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
| T2 | **Harvest `tlab-deep-full`** and fill §4.17's deep half | standing | EXECUTING, step 8100/19531. Read against the pre-registration in `RUNS.md` 13:30 — it is **sw75**, which §3.5 narrowed away from |
| T3 | **Harvest `tlab-anchor-tokenkey`** — §4.18's falsifier (sw90 at k=5/3/2) + token-vs-fraction at 10M | "we should still not be dropping testing ideas" | EXECUTING |
| T4 | **Harvest Kaggle `tlab-seed-extension`** — (sw90 − dense) at seeds 2,3 → n=4 paired estimates | "Do you not want to start any local and/or Kaggle runs?" | pushed |
| T5 | **Report scale-clock arm 3** (`sc_clock_sw90`) and write the whole scale-clock negative into the report | reviewer proposal, user asked to test ideas | arm 3 running; arms 1–2 measured (**+1.36 nats, ‖w‖=1.34**) but **not yet in `report.md`** |
| T6 | **Land the gated-injection result** (fork running) — the third cell on the normalisation axis | my ranked #1, user launched it | fork running |
| T7 | **§8 writing items still owed**: W1 (MLA × LLA), W2 (LoopMTP aggregation conflict), W4 (STARS Pre-Sandwich), W5 (decomposition tables for remaining §4 sections) | reviewer, logged in QUEUE | not started |
| T9 | **8 load-bearing scripts still persist nothing** — `cumulative_exit` (§4.7b), `angular_budget` (§4.16c), `grad_spectrum` (§5.1), `oracle_null`, `intraloop_states`, `normpen_compare`, `rate_vs_path`, `qexit`. Their published numbers are reproducible but not *traceable*: verifying one means re-running, which only works while its inputs survive | traceability audit finding #1 | **3 of 11 fixed** (`jacobian_spec`, `gain_decomp`, `exit_rules`); 8 remain |
| T8 | **Write the degenerate-fixed-point result into the report** (min cross-layer cos → 1.0000 by loop 32) | measured today, reviewer-flagged | measured, **not yet in `report.md`** |

## Blocked on the user

| # | task |
|---|---|
| U1 | **§1** — the idea narrative. Reserved, empty. Stimulus drafts in `needs_user/section1_drafts_STIMULUS.md` |
| U2 | **Rotate the wandb key.** Scrubbing the repo did not un-send it. `needs_user/ROTATE_WANDB_KEY.md` |
| U3 | **D3 — which checkpoint ships.** Control (38.86 ppl, the config §3.5 describes) vs norm-penalty (37.52, but 88% loop-1 damage and a narrower band) |
| U4 | Any GitHub / HF push. **Note: `git push --tags` or `--mirror` WILL be rejected** — a tag carries 1.83 GB of >100 MB blobs; branches are clean (`OPS.md`) |

## Closed today (kept one cycle, then deleted)

- Process every point across **all** reviewer messages, not just the latest → cumulative ledger R1–R60 in `QUEUE.md`
- Reviewer-facing answers as separate files in a subfolder → `reviewer_answers/00`–`12`
- The reviewer's five unknowns → all measured, `reviewer_answers/10`
- R45 (exit rules on an annealed checkpoint) → §4.7a, matched pair
- Deep repo review S1–S11 → 11/11 closed
- Correction-propagation audit → 11/11 applied
- Claims→instrument traceability → applied (ρ reseeded, `gain_decomp` share relabelled, `cumulative_exit` motivation)
- Layer Duplication → **decided against**, with the reason recorded (its own reconciliation means it bears on no claim of ours)
