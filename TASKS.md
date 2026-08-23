# TASKS — live stack of things I have taken on and NOT finished

Scope, deliberately narrow: **commitments I made in response to the user's messages, that are still
open.** Not a plan, not a backlog, not everything that could be done. `QUEUE.md` is the exhaustive
ledger; this file is the short list I can un-stack. **Delete a line when it is done — this file only
ever shrinks unless the user adds to it.**

Last updated 2026-08-23 18:22.

---

## Open — mine

**Nothing is running and nothing is queued.** All seven jobs landed and are written up; all six gates
are green; everything is pushed to `origin/main`.

| # | task | state |
|---|---|---|
| — | final proportion read of `submission/README.md` and `RESULTS.md` §1b | **done 23:05** |
| — | end-to-end read of the whole `submission/` folder | **done 23:00 — 14 defects, all fixed** |
| T7 | §8 W5 decomposition tables | open, and **not** worth doing in the time left |

**Deliberately not started, with the reason:** LoRA × annealing at 12M (~3.1 h), a budget ladder
(~2.2 h), a 90M replicate (~9 h), a second width (~1 h). All are costed in `submission/LIMITATIONS.md`
§7. A run that cannot land before 23:59 cannot be written up honestly, and §6.0 has two rows about
launching under time pressure — one from tonight.

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
