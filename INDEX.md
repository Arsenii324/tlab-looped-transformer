# INDEX — every doc, classified by KIND

> **`SUBMISSION_STATE.md`** (root, written 2026-08-23 21:20) is the agent's honest read of where the
> submission stands: component-by-component confidence, what it would and would not bet on, ranked
> concerns, the five unknown-unknown checks run that evening, and a costed ablation queue. **For the
> author, not for the jury** — it is deliberately not in `submission/`.

Three kinds, because they age differently and you should trust them differently.

- **WORKING** — changes constantly, describes *now*. Overwritten, not appended. Trust the newest.
- **STATIC** — describes decisions, architecture, and method. Changes only when a decision changes.
- **LOG** — append-only history. Never rewritten. The audit trail behind every number.
- **DELIVERABLE** — the thing being graded.

If resuming cold: read **`OPS.md`** (it is a full state dump written against context loss), then
`QUEUE.md` (open points + the reviewer ledger), then `report.md`. Stop there.

**If you have just been compacted: read `OPS.md` §0-PRE first**, then
`reviewer_answers/16_WHOLE_STATE.md`. `COMPACTION_PROMPT.md` holds a custom compaction
instruction for next time.

**`OPS.md` §7b is the UNKNOWN-KNOWNS list** — 24 things that were true, visible in the artifacts, and
unwritten until something collided with them. Read it before trusting any instrument here.

**Three facts that are expensive to rediscover:** (1) HEAD is on branch `review`, a single squashed
commit — run `./rebuild_review.sh` before any review, and never push `submission` (it carries a
scrubbed-but-historical API key). (2) Every DataSphere job discarded its trained weights, so ~20
checkpoints are unrecoverable and two analyses are blocked on it. (3) argmin is retired as a
statistic; use `src/plateau.py`, and compare midpoints only across a shared eval grid.

## WORKING (read first, trust as current)
| file | what |
|---|---|
| `OPS.md` | Live status: what is running on which of the four compute streams, deadline arithmetic, ranked shortlist with done/running/not-started. **STATUS block overwritten every iteration.** |
| `RUNS.md` | Every job with its ID and **what to do when it lands**. Fetch commands at the top. Written because job IDs would otherwise be lost to a compaction. |
| `needs_user/` | **NOT IN THE REPOSITORY** — untracked 2026-08-23, kept on the author's disk only. |

## ADDED 2026-08-23 — read these too
| file | what | why it exists |
|---|---|---|
| `DECISIONS.md` | every choice with a provenance tag (MEASURED / INHERITED / ASSUMED / UNEXAMINED) + 4 uncomfortable questions | a blind-spot pass; found that the MLP ratio was never screened and that 3 architecture axes rest on a retracted sweep |
| `QUEUE.md` | **open points** (blocked / in-flight / deferred / user-owned) **and the R1–R47 reviewer-points ledger** | I twice audited only the newest reviewer message; the ledger is cumulative so that cannot recur |
| `reviewer_answers/00–27` | one file per reply to the external reviewer (**27 files as of 21:58**) | **files are sent at creation time and NOT re-sent after editing — corrections must be NEW numbered files** (learned via `05_RETRACTION_73_percent.md`) |
| `papers/sources/` | LaTeX of 16 cited papers + `TASK_STATEMENT_ru.txt` | so citations can be checked, not trusted. Exists because I once "corrected" a relayed number from a summarising web fetch and was wrong |
| `rebuild_review.sh` | regenerates branch `review` as ONE squashed commit | `/code-review` reviews the most recent commit, not the branch range. **Run before any review** |
| `needs_user/` | **NOT IN THE REPOSITORY.** Author-facing correspondence — rotation reminders, push checklists, the §1 stimulus drafts. Untracked and gitignored on 2026-08-23; the files remain on the author's disk. Nothing in `submission/` ever depended on them |
| `src/plateau.py` + `test_plateau.py` | the useful-depth band; **replaces argmin everywhere** | 63/82 stored curves have argmin margins below the noise floor |
| `src/gain_decomp.py` | Δgain = ΔCE@1 − ΔCE_best | separates "depth improved" from "loop 1 got worse"; the 90M norm penalty is 88% the latter |
| `src/angular_budget.py` | angular path length, fixed-range + **untrained control** | the untrained control refuted my own interpretation of it |
| `src/check_tokenizer_identity.py` | vocab-identity gate against *chance*, not a fixed tolerance | the Kaggle kernel never saved its tokenizer; identity had been inferred, not checked |
| `src/argmin_audit.py`, `cumulative_exit.py`, `intraloop_states.py`, `tl_seed_check.py`, `normpen_compare.py`, `ds_harvest.py` | audit + analysis instruments | see §7.1 of `report.md` |

## STATIC (decisions and method; change only when a decision changes)
| file | what |
|---|---|
| `STATE_FOR_REVIEWER.md` | Full technical dump for the reviewing agent: exact model config, exact training setup, every result with the instrument that produced it, and §6 — inefficiencies I believe exist in my own design (MLP ratio, Muon under weight-tying, `n_loop_eff` mismatch, seq_len, vocab, schedule shape, dense supervision). |
| `PLAN.md` | Original design rationale, rejected alternatives, what would flip each choice. |
| `REVIEW_NOTES.md` | Every substantive external-reviewer claim, whether I verified it from source, what I did, and **where I disagreed and why**. Written so the reasoning survives a compaction, not just the queue items. |
| `DATASPHERE_NOTES.md` | Working DataSphere invocation for this project + two traps the ccm-intro guide lacks. |
| `README.md` | Repo layout and quickstart. |
| `CLAUDE.md` | Project rules. Governs everything here. |

## LOG (append-only; never rewrite)
| file | what |
|---|---|
| `LOG.md` | Chronological ledger: every run, bug, fix, self-correction, with timestamps. |
| `queue_run.log` | Output of `run_queue.sh` (gitignored). Step-level START/OK/FAIL. |

## DELIVERABLE
| file | what |
|---|---|
| `report.md` | **§1 is an account reconstructed from the dated record** (`LOG.md`, §6.0, `RUNS.md`), not a recollection; its own banner says so. §2–§8 are the measured report. |

## STALE (superseded; kept for history, do not trust on conflicts)
`BRIEFING.md` (predates the Lemma-2 attribution and the clamp result), `HANDOFF.md` (replaced by `OPS.md`).

## Where numbers live — never quote one that is not traceable here
| artifact | contains | authority |
|---|---|---|
| `checkpoints/*/eval_*.json` | post-hoc dense per-loop sweeps | **authoritative for claims** |
| `checkpoints/*_history.json` | in-training 6-batch coarse-grid evals | **trajectories only** (disagree with post-hoc by up to 0.061 nats) |
| `checkpoints/*/dynamics_*.json` | per-loop norms, perturbation, step, increment cosine, ‖v‖, ‖norm1(h)‖ |
| `checkpoints/*/clamp_*.json` | radial-clamp curves, 3 levels + control |
| `checkpoints/*/crossdepth_*.json` | cross-depth KV grid (cache depth k × compute depth t) |
| `checkpoints/*/paired_*.npz` | per-sequence per-loop CE on the frozen set | for paired comparisons |
| `checkpoints/*/exitdump_*.npz` | per-token per-loop CE + entropy + margin + ‖Δh‖/‖h‖ + KL |
| `data/frozen_eval_set.npz` | the ONE frozen eval set all paired comparisons use |

## INSTRUMENTS added 2026-08-23 (the statistics half the report now rests on)
| file | what it decides | null / validation |
|---|---|---|
| `src/plateau.py` | **Replaces argmin everywhere.** The band of depths within `tol` of a curve's minimum, plus midpoint/onset. Depth claims are stated as plateaus. | `src/test_plateau.py` — 8 checks incl. flat, non-contiguous, degenerate-raises, geometric-midpoint identity, tolerance monotonicity, and exact reproduction of §4.9's published `trainL16` numbers |
| `src/argmin_audit.py` | Flags every stored curve whose argmin is decided below the noise floor. **63 of 82 are.** | enumerates JSON shapes explicitly after it silently skipped `sandwich_eval.json` once |
| `src/gain_vs_ce.py` | Tests the report's most-repeated claim (loop gain trades against CE) as a correlation over all 43 arms, stratified by device × budget. **Pooled ρ = −0.081; strata disagree in sign.** | ranks with tie-averaging; strata centred before pooling |
| `src/tl_seed_check.py` | Whether §4.9's t/L collapse survives a second seed. Computes seed noise on **both** raw and re-zeroed curves — the distinction that invalidated the original comparison. | reproduces §4.9's published spread row to **1.7e-06** before touching new data |
| `src/normpen_compare.py` | Resolves §4.6's pre-registered prediction. Shared-grid enforced; **floor is a parameter**, because the verdict flips between the measured and conservative values. | control-vs-itself returns "inert on both axes" |
| `src/ds_harvest.py` | Reads finished arms out of a **live** DataSphere attach log, so a multi-hour job can be analysed midway. Handles logs split by an attach death. | verified against the authoritative `results.json` of two completed jobs — exact match |

## MONITORS (all shell, all version-controlled, all restartable)
| file | what it prevents |
|---|---|
| `monitor_kaggle.sh` | A wrong slug making a live run look dead. Reads `KAGGLE_SLUGS.txt`; auto-downloads output on any terminal state. |
| `ds_watchdog.sh` | ~~A dead local attach silently losing a job's completion download.~~ **KILLED 18:40** for re-attaching an already-finished job; by 20:08 three attach logs had frozen with nothing covering it (`OPS.md` §7b #29). Harvest via `download-files --id`, never an attach log's step count. |
| `monitor_failures.sh` | Alert fatigue: reports each distinct failure signature **once**, replacing a monitor that re-fired stale terminal states every cycle. |
| `run_eval90.sh` | Swapping the headline on a cross-protocol comparison. Waits for GPU idle, then re-scores both 90M checkpoints under the protocol that produced the current headline. |

## Standing rules that outlive any session
- `report.md` §1 is reconstructed from the dated record and carries a banner saying so.
- Nothing pushed to GitHub/HF without explicit say-so.
- Never overwrite a published `eval_*.json` — back up, restore, verify md5.
- Budget arms by **tokens**, not wall clock.
- Verify every number against raw JSON before it enters `report.md`.
- DataSphere: `project job list` **before every submit** — submission is not idempotent.
- **argmin over a loop curve is retired.** 63/82 stored curves have argmin margins under 0.005 nats
  against a floor of 0.015-0.068. Use `src/plateau.py`; compare midpoints ONLY across a shared eval
  grid (grid choice alone moves a midpoint 17%).
- **Fixed seed is not a replicate** (§4.15). Measured run-to-run floors: MPS dense 0.031/0.068,
  CUDA dense 0.0150, CUDA terminal-only 0.0541. The floor is config-dependent, not just device-dependent.
- **Compare against the POOLED reference, never the single run that flatters the new arm** — and
  re-check when more replicates land. Both errors happened today and both were caught late.
- **Two different loop curves exist and must never be conflated: EVAL-AT-T** (train once, sweep
  inference depth — everything in this repo until 2026-08-22 23:40) **vs TRAIN-AT-L** (train separate
  models at fixed L, evaluate each at its own L). The task's "more loops is better" reads as
  train-at-L. See `OPS.md` for the sweep now running.
