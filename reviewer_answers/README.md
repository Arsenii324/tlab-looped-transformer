# reviewer_answers/

**START WITH `16_WHOLE_STATE.md`.** It is self-contained, supersedes every earlier reply where they
disagree, and ends with a self-check. Files 00–15 are the incremental record and several of their
claims were withdrawn by later measurements in this same project — they are kept unedited because
this project retracts visibly rather than deleting.

Superseded-in-part, and by what:
- `12`, `13` report the annealing CE advantage as holding at n=2 → **withdrawn at n=4** (§3(i) of 16)
- `13` presents the degenerate cross-layer collapse as a headline → **largely a shared-residual
  artifact** (§3(iii) of 16)
- anything quoting `rho = 1.7019` → **false precision from an unseeded estimator**; the seeded value
  is 1.6227 (§3(iv) of 16)

---

# reviewer_answers/

One file per reply to the external reviewer, newest last.

**CONVENTION (added 11:25, learned the hard way):** files here are sent at creation time and are
**not re-sent after editing**. So a correction to an already-sent file is invisible to the reader.
**Corrections must be new numbered files.** This was discovered when a retraction was written as an
edit to three sent files and would never have reached the reviewer — see `05_RETRACTION_73_percent.md`. Split out of the single
`ANSWERS_FOR_REVIEWER.md` on 2026-08-23 at the user's request.

| file | covers |
|---|---|
| `00_earlier_answers.md` | everything up to and including 2026-08-23 10:45: the deep full-budget artifact launch (and why its config departs from the proposal), the t/L spine correction, the rejection of "freeze the instruments", and the prior-art verification (2311.12424 confirmed; Think-at-Hard's "73%" — my "correction" to 85% was itself WRONG and is retracted; 73% is verbatim in the v3 source) |
| `01_full_task_statement.md` | the full-task-statement message: the missing final-architecture section (§3.5, written and marked provisional), the process-failures section (§6.0) as the criterion-2 deliverable, the slot recommendation (accepted — launched the missing constant-terminal seed-1 arm), the non-monotonicity reframing, and σ_max aimed at the DEQ premise |
| `02_deliverable_completeness.md` | D1–D3/S1–S3/T1: the tokenizer trap (live, now verified + a README bug that would have made every released checkpoint evaluate at chance), the fresh-clone dry run (the push would have been rejected — 564MB tracked files), checkpoint choice, and the exit-head spec |
| `03_decisions_and_blindspots.md` | the blind-spot pass: `DECISIONS.md` written to their format, what the forcing function caught (§3.4 arguing against itself, three untrustworthy MEASURED rows, the unscreened MLP ratio), both proposed screens launched, D2 closed with byte-identical shards, and the fp16 check |
| `04_report_map.md` | one line per claim across all 42 sections with status (HOLDS/NARROWED/RETRACTED/PROVISIONAL), marking what is NEW since the reviewer's 22 Aug copy — fixes the asymmetry where half the report is invisible to them |
| `05_RETRACTION_73_percent.md` | **retraction**: I told the reviewer their relayed "73%" was wrong and the paper said 85%. The v3 LaTeX says 73%. Their figure was right; mine came from a summarising HTML fetch I treated as a primary source |
| `06_angular_budget_and_regime_change.md` | the angular budget ran (B rises 1.38-1.42x -> terminal-only is a BUDGET intervention, §4.16c); the penalty's reversal adopted as framing and pre-registered before the 10M control landed; the same reversal found on the schedule axis, qualifying my own method; bpb calibration; and why four relayed citations are logged SECOND-HAND rather than cited |
| `07_since_06.md` | headline swapped on protocol-matched evals (ppl 54.99 -> 38.86 / 37.52) with the D3 disclosure adopted; LR screen complete and the inherited 3e-3 is optimal; and three items recovered from the reviewer's EARLIER messages — including unrecorded prior art on §4.17's own ingredient |
| `08_status_of_every_point.md` | every reviewer point in three tiers (in full / not in full / deferred-not-skipped), backed by the cumulative R1-R47 ledger in QUEUE.md, plus seven unknown-knowns that surfaced only when artifacts collided with actions |
| `09_wide_angle.md` | unscoped status: the full forward pass with param counts and exact hook placement, the shipping-checkpoint config, in-flight table, five report-vs-code disagreements, the another-week vs comparability-doesn't-matter split, and the weakest point (the method's effect size is the same order as today's corrections to it). All four proposals RUN; three refuted the prediction; the chord-vs-arc check REVERSED B's sign |
