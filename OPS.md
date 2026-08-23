# OPS — autonomous session plan, 2026-08-22 22:34 MSK → deadline ~2026-08-23 23:30 MSK

Operational state file for a long unattended run. Not a thinking record (that's LOG.md), not the
deliverable (report.md) — this exists so a context compaction or a cold restart can resume without
re-deriving the plan. Overwrite the STATUS block; append nothing else.

## Deadline arithmetic
- Deadline ~2026-08-23 23:30 MSK. `date` before trusting any elapsed-time assumption — this
  project's single most repeated lesson (a message once assumed 3 days left when ~8h remained).
- Kaggle 90M-token run launched 2026-08-22 ~20:05 MSK, cap 10.8h → expect completion ~06:30 MSK.
  Kaggle hard ceiling is 12h; the run stops itself gracefully before that.

## Ranked shortlist (source: reviewer message, 2026-08-22)
1. **[RUNNING]** 100M-token Kaggle run. ~0.25-0.31 nats by this project's own 0.398 nats/e-fold.
   No thinking required; the only thing that ever blocked it was a wall-clock cap.
2. **Radial clamp**, post-hoc on the Kaggle checkpoint, levels {|h1|,|h8|,|h16|}. Zero training.
   Decides whether dilution is the binding constraint or whether norm growth is *protecting* the
   model from its own harmful drift. Reviewer predicts clamping makes things WORSE past loop 8.
3. **Norm penalty + raw readout** — two of Sharma & Vu's four interventions never tried here, both
   zero-parameter. Token-budgeted, multi-seed.
4. **Supervision density** — mu_rec ~18 but optimum is 8; loop 8 gets a direct CE gradient ~3x as
   often as loop 25. Concentrate the schedule at 24-32; if the optimum follows, saturation is
   demand-side, not dynamics.
5. **eps = lambda/(N*sqrt(L))** residual scaling replacing depth_init's 1/sqrt(2*n_loop_eff).
6. **Cross-depth KV matrix** — the user's own early-exit idea. Build attention cache at loop t from
   loop-k states, sweep (k,t). Teacher-forced, no generation. The unclaimed half of that idea.

## Standing constraints (do not violate unattended)
- No subagents, no Workflow tool. Direct tool use only.
- Nothing pushed to GitHub/HF without explicit say-so. Kaggle kernel pushes are fine (already in use).
- Shared machine: check `ps aux -m` / swap before launching. "Better OOM than endless swap."
- Never overwrite a published eval_*.json — back it up first, restore, verify md5.
- Verify every number against raw JSON before it enters report.md.
- Budget screening arms by TOKENS, not wall clock (that confound already flipped two results).
- report.md §1 is the user's; do not fill it.

## DEFINITIVE PLAN (written 2026-08-23 00:10, deadline 23:30 = 23.3h)

### The fact that reshapes it
**DataSphere jobs are early-stop-safe** (tested: mid-compute cancel recovered BOTH stdout and the
declared `outputs:` file; cancel during *setup* recovers nothing). Our kernels write `results.json`
+ a checkpoint at **every eval boundary**, and run arms cheapest-first. So a DataSphere job may be
launched without fitting the cutoff and simply harvested at 18:00.
**Kaggle is NOT early-stop-safe** (output only on completion) and has ~9h weekly quota left, so any
Kaggle run must fit end-to-end. => **All remaining long compute goes to DataSphere.**

### The strategic point
The deliverable is *lowest val perplexity obtained BY exploiting many loops*. Today's headline
(46.0M tok, CE 4.0071, bpb 1.7330) and the two 90M runs all use a RANDOMIZED 4-32 schedule, i.e.
they are eval-at-T evidence. If the train-at-L sweep (ETA 04:30) is monotone in L, then a
**full-budget run at a fixed LARGE L** is the config that answers the task directly: best perplexity
*and* many loops, in one number. That is the highest-value remaining action and it depends on a
result arriving at 04:30.

### Timeline
| when | action |
|---|---|
| now-02:30 | analyze convex-gate A/B; write report §5/§6/§8 prose that needs no new numbers |
| ~03:30 | gate-sweep lands -> read the ORDERING across g (self-controlling, g=1.0 is the ungated model) |
| **04:30** | **train-at-L lands. DECISION POINT: pick L\* = argmin CE, check monotonicity.** |
| **~05:00** | **launch full-budget run at fixed L\* on DataSphere.** Early-stop-safe, so size it to run past the cutoff and harvest at 18:00. At mean-32 that is ~65M tokens by 18:00; at mean-16 the full 90M finishes ~14:20. |
| 06:30 | Kaggle 90M control lands -> pull ckpt, paired-eval it |
| 09:30 | Kaggle 90M norm-penalty lands -> **paired** comparison vs control (same seed+data order, verified) -> resolves the pre-registered prediction in §4.6 |
| 09:30-13:30 | local MPS queue finishes (crossdepth, paired, residual-scale, per-arm evals, argmin_anatomy, clamp-on-center, sandwich dynamics, 14.6M exit dump) |
| 13:30-15:30 | local GPU free -> any cheap post-hoc gap; verify every number against raw JSON |
| **15:30** | last launch for anything needing analysis before writing |
| **18:00** | **harvest the DataSphere long run (cancel if still going). HARD STOP on new compute.** |
| 18:00-19:30 | consolidate: fill §4.8/§4.9/§4.10 placeholders, re-verify all numbers |
| 19:30-23:30 | write/finalize `report.md`. No new compute. |

### Rules for the rest of the run
- No dependent chains beyond the single 04:30 decision point.
- Everything launched after 05:00 is either cheap post-hoc or early-stop-safe on DataSphere.
- Analyze on arrival; never let a result sit unread while compute continues.
- Kaggle: do not launch again unless something fails and quota clearly permits.

## STATUS — full state as of 2026-08-23 **17:30** (rewritten against a second compaction)

### 0-PRE. POST-COMPACTION GUIDE — written 2026-08-23 17:30, read this FIRST

**You are mid-session on the T-Lab looped-transformer submission. Deadline ~23:30 MSK today.**
Context was just compacted; the sections below are the durable state. Read in this order:

1. **`reviewer_answers/16_WHOLE_STATE.md`** — the single best summary of where the project stands.
   Self-contained, supersedes replies 00–15 where they disagree, ends with a 9-item self-check.
2. **This file, §0–§7b** — operational state, retractions, unknown knowns.
3. **`TASKS.md`** — the short list of open commitments (only ever shrinks).
4. **`QUEUE.md`** R1–R60 + S1–S11 — the cumulative reviewer ledger. Nothing is dropped from it.

**Three things that will save you from a mistake in the first ten minutes:**
- **Three claims were withdrawn today.** Annealing's CE advantage (n=2 → withdrawn at n=4),
  §4.20's degenerate collapse (a shared-residual artifact), and §3.5's deep half (`tlab-deep-full`'s
  falsifier fired at mid 22.6). **Do not re-assert any of them** — see §4 below. Earlier
  `reviewer_answers/` files still contain them and are kept as the visible retraction record.
- **Nobody has read `report.md` end to end**, including me, across three retractions. That is the
  largest outstanding risk on the graded artifact (`TASKS.md` T17). A *mechanical* pass was done
  (duplicate values, summaries overclaiming, cross-grid comparisons, unflagged citations) and found
  four real defects, all now fixed — but that is not the same as reading it.
- **Neither the git push nor the HF upload has ever run, and no remote is configured.** Both are the
  user's call, not yours. `upload_checkpoint.py` was fixed today and dry-run verified but never
  executed. The fresh-clone dry run passed at 11:00 and the tree has changed a lot since — **re-run
  it before shipping.**

**Behavioural notes that were earned the hard way today:**
- Prefix every DataSphere call with `GRPC_DNS_RESOLVER=native`, or it fails with a DNS error that
  looks like a network outage and is not.
- A DataSphere job reporting **ERROR** may have completed fine — stderr from pip/wandb marks the job
  ERROR. **Read the raw stdout before believing the status.** This happened twice today.
- Local Python is **anaconda base**, not `barannikov-work/.venv` (§7).
- Before writing any fork's or agent's number into the report, **re-measure it**. Two probes handed
  to me today paired oracle depths with the wrong tokens; both conclusions happened to survive, by
  luck of statistic rather than design.

### 0. If you read one thing
The deliverable is `report.md` (~3,900 lines, 0 placeholders, §1 reserved for the user). The method is
**§3.5**. Today's arc was a methodology audit that changed many claims; **every retraction is in
§6.0's table and in LOG.md with a timestamp.** Open items live in `QUEUE.md`, which also carries the
**R1–R47 reviewer-points ledger**. Replies to the external reviewer are `reviewer_answers/00..08`.

### 1. Git / submission state — READ BEFORE ANY PUSH OR REVIEW
- **HEAD is on branch `review`** = **ONE squashed commit** containing the whole project (482 files,
  138k insertions). `/code-review` with a clean tree reviews **the most recent commit**, not the
  branch range — which is why a review at 13:00 scoped itself to a doc-only diff and was wasted.
- **`./rebuild_review.sh` regenerates `review` as one commit. RUN IT BEFORE ANY REVIEW.**
- `submission` = full working history. `main` = the repo's first commit. Tag `main-backup-20260823`
  = the original main. **Commit day-to-day work, then rebuild `review` before reviewing.**
- **PUSH HAZARD: `git push --tags` / `--mirror` WILL BE REJECTED.** `refs/tags/main-backup-20260823`
  carries 4 blobs over GitHub's hard 100MB limit (563.9 / 563.8 / 560.7 / 181.9 MB — the
  `exitdump_*.npz` files and `ds_exit/ckpt.pt`), 1.83 GB total. **Every branch is clean** — verified
  per-ref — so `git push <remote> <branch>` succeeds. `.gitignore` cannot help; it does not untrack
  history. The tag is left intact (nothing is deleted here); just never push it. Write new dumps to
  the scratchpad, never into the repo.
- **The wandb key was leaked** into 18 configs + 18 commits. Scrubbed; `review` is verified clean
  (0 commits, 0 diff occurrences). **`submission` still contains it — do not push `submission`.**
  **The key still needs ROTATING** → `needs_user/ROTATE_WANDB_KEY.md`.
- Nothing has ever been pushed to any remote.

### 2. Headline numbers (protocol-matched, `src/eval.py`, same protocol as the 46M figure)
| run | tokens | CE | **ppl** | bpb | plateau |
|---|---|---|---|---|---|
| old headline | 46.0M | 4.0071 | 54.99 | 1.7330 | [5,14] |
| **90M control** (the §3.5 config) | 90.0M | **3.6599** | **38.86** | 1.5829 | [6,17] |
| **90M + norm penalty** (best ppl) | 90.0M | **3.6250** | **37.52** | 1.5678 | [6,14] |
Finishing the token budget bought **0.39–0.42 nats**; every architectural intervention is 0.002–0.19.

### 3. The method (§3.5), and what today established about it
Qwen3 3-layer block, weight-tied, **no prelude/coda**, **no inter-loop norm**, additive injection,
deep loop schedule, and **supervision annealing** — dense, then terminal-only (k=1) for the last
10–25% of steps. **Zero added parameters.**
- **§4.17 confirmed at 2 seeds in-job** (ΔCE_best −0.0811 / −0.0609) **and at 4× budget**
  (**outcome A**, 10M: ΔCE_best −0.0764, ΔCE@1 −0.0092, both-improve). Caveat: the loop-1 margin
  eroded −0.035 → −0.0092, trending toward outcome B without reaching it.
- **§4.16 threshold at k=1** (gain drops 0.162 from k1→k2, then varies 0.018 across k=2,3,5,8).
- **`an_rev50` control**: the same k=1 exposure placed FIRST gives no depth effect and the worst CE →
  the **final phase** sets the band.
- **§4.16b**: terminal-only's useful depth tracks μ_rec (1.09/1.00/0.98 at μ=18/32/40).

### 4. RETRACTIONS / WITHDRAWALS — do not re-assert any of these

**Three landed today after 13:45, all by pre-registered criteria that were written before the data
existed. Each is recorded in `report.md` with the superseded claim visible, not deleted.**

1. **Annealing's CE advantage over dense — WITHDRAWN at n=4.** Was −0.0811/−0.0609 at seeds 0/1.
   Seeds 2/3 gave **+0.0482** and −0.0902; n=4 mean −0.0460, sd 0.0640, t-interval
   [−0.1478, +0.0558] covers zero. **What SURVIVES: the band widens at 4/4 seeds** (+2.5/+2.5/+2.5/+7.2),
   including at seed 2 which reverses the CE claim. The claim is now "relocates the band, does not
   move the ceiling."
2. **§4.20's degenerate cross-layer collapse — largely a SHARED-RESIDUAL ARTIFACT.** cos(outputs)
   → 1.0000 is real but cos(**increments**) is 0.14–0.18; each layer moves the state by 0.5–3.5% of
   its norm so all outputs share a dominant residual. Confirmed by per-loop scalar diversity (σ≤1.0)
   and per-loop LoRA operator diversity (168 tensors randomized) both leaving cos@64 unchanged.
   **This closed the whole conditioning/branch-diversity family without a training run.**
3. **§3.5's deep half — WITHDRAWN.** `tlab-deep-full` (30M tokens, the only ≥32-loop artifact)
   returned plateau mid **22.6** against a pre-registered trigger of "near 22"; mid/μ_rec = 0.57 is
   in §4.16b's *dense* range, not terminal-only. The 2.5M screen had said mid 45.3.
   **Survives: band [16,32], CE improving to loop 24, graceful to 128 — the deepest useful band here.**

**Earlier retractions still standing** (see `report.md` §6.0, 32 rows): §4.4's untrainability claim;
`B`-as-useful-computation; "terminal-only buys more budget"; the strong t/L collapse; the 1.4×
angular budget; `sigma_max`→ρ mislabel; ρ=1.7019 false precision (seeded value 1.6227).

**One thing NOT withdrawn but downgraded:** token-keyed annealing beats fraction-keyed by −0.2208 —
**this is n=1** and §3.5 states it as a **lead, not a recommendation**. Measured seed spread on this
class of paired difference is sd=0.0640.

### 5. Instruments built today (all in `src/`, all with the null that validates them)
`plateau.py` (+`test_plateau.py`, 8 checks) · `argmin_audit.py` (63/82 curves have unresolvable
argmins) · `gain_decomp.py` (Δgain = ΔCE@1 − ΔCE_best; 49 in-job pairs) · `angular_budget.py`
(+fixed-range +untrained control) · `intraloop_states.py` · `cumulative_exit.py` ·
`check_tokenizer_identity.py` · `tl_seed_check.py` · `normpen_compare.py` · `ds_harvest.py`

### 6. Live compute — as of 2026-08-23 17:30

| stream | state |
|---|---|
| DS `tlab-deep-full` | **DONE 17:23, harvested.** Falsifier fired (mid 22.6). Artifact at `checkpoints/deep_full_results.json`. Curves only — predates the `outputs:` fix |
| DS `tlab-anchor-tokenkey` | **DONE, harvested.** ERROR status was cosmetic; all 6 arms completed |
| Kaggle `tlab-seed-extension` | **DONE, harvested** — produced the n=4 withdrawal |
| local `run_operator_diversity` | arm 4/4 (`od_depth_gate`) running. **Read pre-registered in `RUNS.md` 17:40** with two caveats: it zero-inits to a *uniform mixture* not the control, and its eval curve mixes over loops 1..r so **its plateau is over mixture-window size, not depth** — do not put it in the band tables |
| DS `tlab-operator-diversity` | ERRORED (`tokenizers` install never ran). Not relaunched; motivation retired by retraction #2 |
| agy jobs A/B | returned, **unverified except 3 sampled findings**; job C (citation cross-check) never ran — substitute audit done by hand, in `VERIFICATION.md` |

**Nothing else is queued. Local GPU frees up when arm 4/4 finishes.**

### 7. BLOCKED, with causes (see QUEUE.md B1–B3)
- **Every DataSphere job discarded its weights** — configs listed only `results.json` under
  `outputs:`. ~20 checkpoints unrecoverable. Fixed in 23 configs for future jobs. This blocks:
  `B_L` across the five train-at-L arms, and **the exit-rule test on an annealed checkpoint (R45)**.
  *A LOCAL annealed run would unblock R45 — `src/train.py` would need `supervise_k_final` added.*
- Four relayed papers unobtainable → SECOND-HAND in `VERIFICATION.md`, no claim rests on them.

### 7. INTERPRETER — every local number came from anaconda base, NOT the workspace venv
    /Users/a2mogus/anaconda3/bin/python3    3.11.0, torch 2.8.0, numpy 2.4.6, MPS: yes
`barannikov-work/.venv` (which the sibling CLAUDE.md names as the workspace default) has torch 2.13.0
and **no** tokenizers/datasets/huggingface_hub/transformers — it cannot run `src/data.py`,
`src/train_tokenizer.py`, or `test_model.py` check [2]. Bare `python3` resolves to anaconda on this
machine, which is why everything worked without anyone choosing it. Check the interpreter first if a
local number fails to reproduce.

### 7a. HOW TO RUN A REVIEW (read before trying)
`./rebuild_review.sh` builds two sized targets; a reviewer's limits are 500 files AND 8,000 lines,
and this repo is 486 files / 139k lines, so an unscoped review is refused outright.

    ./rebuild_review.sh
    /ultrareview review-code-base     # on branch 'review'      -> src/*.py + kaggle/main.py, 6,721 lines
    /ultrareview review-docs-base     # on branch 'review-docs'  -> report.md, 4,000 lines

Code and report want different reviews, which is why they are separate targets rather than one.
`configs/tokenizer.json` (19,728 lines) and `checkpoints/*.json` (50,932) are data and are excluded
from both diffs — they remain in the base commit, so a reviewer can still open them for context.
`git tag pre-squash-history` (963fab4) holds the pre-squash linear history; all of its content also
survives in `review`, verified file-by-file, but the commit-by-commit record lives only on that tag.

### 7b. UNKNOWN KNOWNS — things that were true, visible in my own artifacts, and unwritten
*Kept here rather than in `reviewer_answers/` because by definition these are what a fresh context
will not think to look for. Every one surfaced when an artifact collided with an action — none came
from asking myself questions.*

| # | what was true | what surfaced it | cost |
|---|---|---|---|
| 1 | **Every DataSphere job discarded its trained weights** (`outputs:` listed only `results.json`) | §4.16c needing the train-at-L checkpoints | ~20 checkpoints unrecoverable; 2 analyses still blocked |
| 2 | **The wandb API key sat in 18 tracked configs and 18 commits** | debugging an unrelated probe-job failure | one command short of a cloud review shipping it; **key still needs rotating** |
| 3 | **The README told a grader to run `train_tokenizer.py` first**, overwriting the shipped vocab | the reviewer asking D1 | every released checkpoint would have evaluated at chance (~8.32), looking like a broken model |
| 4 | **I was auditing only the newest reviewer message** | the user asking twice | earlier points silently dropped; fixed by the cumulative R1–R47 ledger |
| 5 | **§3 contradicted §4.7 in the same document** (3.45 chars/token vs 3.3358 bytes/token) | auditing older reviewer messages | I had verified the divisor and never fixed the sentence asserting the wrong one |
| 6 | **Kernel and local evals differ by a consistent ~0.04 nats** | `run_eval90.sh` producing both | exactly the size of several claimed effects; it is why the headline swap waited |
| 7 | **~44 sections and nobody has read the report end to end**, including me | writing the claim-level map | the map is a partial fix, not a real one |
| 8 | **`/code-review` reviews the most recent COMMIT, not the branch-vs-main range** | a review scoping itself to a doc-only diff | one ultra run wasted; the branch restructure had been built on the wrong assumption |
| 9 | **Committing to `review` destroys the single-commit property it exists for** | 28 commits had accumulated on it | fixed by `rebuild_review.sh`; **run it before any review** |
| 10 | **`B` was a chord, not a path** — a path integral sampled once per loop | a reviewer asking whether the hooks were per-loop | reversed §4.16c's sign (1.20 → 0.80) |
| 11 | **`inject_none` is "no RE-injection", not "no injection"** — `h = h0 + e` is unconditional | a fork reading `model.py:413` | the code carried a comment saying so; I had never read it |
| 12 | **A fork with my context can commit to `report.md` without the main session verifying** | finding commit `1f55028` I did not make | the claim happened to be right; rule now: re-measure before any fork-authored number stands |
| 13 | **The method's effect (0.061–0.081 nats) is the same order as the instrument corrections made today** | a fork's wide-angle pass | now written into §4.17 as the objection I would raise against myself |
| 14 | **I was sleep-polling at 115s for arms that take 14 minutes** | the user asking whether the sleeps were justified | burned turns; do work between checks instead |
| 15 | **`main` is a single initial commit**, so the whole project reads as one 486-file / 139k-line addition | `/ultrareview` refusing it against its 500-file / 8,000-line limits | the squash that fixed one review-scoping bug created its mirror image |
| 16 | **A review base must be an ANCESTOR of the head, not a sibling off main** — three-dot diff uses the merge base | building a sibling base and getting the same 139k diff back | silent: the branch looked right and scoped nothing | | the user asking whether the sleeps were justified | burned turns; do work between checks instead |

| 17 | **I repeated the exact small-n error I had withdrawn two hours earlier.** The token-keyed result (n=1) was written into §3.5 as "the recommendation should be read as token-keyed" — the same failure as the annealing CE claim, on a larger effect, in the same document | an external reviewer asking "what is n?" | **knowing the lesson did not prevent repeating it.** The only thing that caught it was someone outside the work asking a one-line question |
| 18 | **Seven distinct eval grids exist across the artifacts**, not the two I assumed; `plateau_mid` is grid-conditional with a 17% swing | auditing after surfacing it as a "known unknown" in a reviewer answer | audited clean — every load-bearing comparison shares a grid — but the risk was real and unwritten for the whole project |
| 19 | **§4.6b's `raw` and `final_only` readout arms train with gradients pinned at the clip 100% of steps** (raw norms 26.1, 85.1) while the `norm` control never clips (0.84) | sampling 3 of 10 unverified findings from a delegated log sweep | the intervention *causes* the clipping, so it cannot be separated by re-running. Those conclusions are about "that readout **under saturating clipping**" |
| 20 | **`n_loop_eff` is a constant fixed at 24 in every checkpoint** while `depth_init` scales by `1/√(2·n_loop_eff)` and schedules ran at mean depth 18 and 40 | a reviewer suggesting it as a 15-minute check | in-job pairs unaffected (both arms share the wrong constant); cross-schedule comparisons carry it. Now a stated §6.0b limitation |
| 21 | **Two delegated probes paired oracle-depth labels with the wrong tokens** — depths came from `frozen_eval_set.npz`, tokens were read as sequential `val.bin` slices (frozen starts are 219, 494, 2630…) | checking the method before trusting the output | fixed and re-run; **both conclusions survived — but by luck of statistic** (one was a within-token comparison immune to shuffling), not by design |
| 22 | **Neither the git push nor the HF upload has ever run, and no git remote is configured** | being asked point-blank about submission status at 18:00 | both are the literal submission targets. `upload_checkpoint.py` was fixed today (it previously shipped weights *without* the tokenizer) and dry-run verified, never executed |
| 23 | **Nobody had ever generated text from the shipped checkpoint** | a reviewer asking whether anyone had looked | it passes — recognisably English, grammatical, prompt-anchored. The cheapest end-to-end defect check in the project, unrun for the entire session, and the exact failure the task statement warns about |
| 24 | **§4.20's headline statistic measured layer *outputs*, which share a residual, rather than layer *contributions*** | trying to break the collapse with operator diversity and failing, then asking why | the finding was substantially an artifact. Two forward passes retired an architectural direction that would have cost a training slot and up to 3.8% of the parameter budget |

**Second meta-pattern, added 2026-08-23 17:30 and worth more than the first:** #17, #23 and #24 all
surfaced because *someone outside the work asked a one-line question* — "what is n?", "has anyone
looked at what it writes?", "does diversity break it?". None came from introspection, and #17 is the
sharpest case: I had written the lesson down, withdrawn a claim for violating it, and violated it
again two hours later on a larger claim. **Writing a lesson down does not install it.** The cheap
defence is not more self-review; it is keeping a channel open to someone who will ask the obvious
question.

**The first meta-pattern:** in #10, #11 and §4.3's increment finding,
**the sampling or scoping rate WAS the result.** A quantity measured at the wrong resolution did not
look wrong — it looked like a finding. That is the failure this project is most prone to, and the
check is always cheap: *ask what the instrument samples before believing what it reports.*

### 8. Doc map
`report.md` deliverable · `OPS.md` this file · `LOG.md` full chronology · `RUNS.md` jobs +
pre-registrations · `QUEUE.md` open points + R1–R47 ledger · `DECISIONS.md` provenance of every
choice · `VERIFICATION.md` external claims + SECOND-HAND · `REVIEW_NOTES.md` reviewer claims since
the start · `INDEX.md` doc classification · `reviewer_answers/` 00–08 · `needs_user/` blocked on user
· `papers/sources/` 16 verified papers + task statement

## Hazard: never pkill a DataSphere attach by its shared command line
Every `datasphere project job execute -p <project> -c config.yaml` invocation has an **identical**
command line regardless of which job it is attached to. On 2026-08-23 06:09, cancelling the
oversized deep-terminal job with
    pkill -f "ds_deepterm|job execute -p bt12q57tmrs03pnt8drc -c config.yaml"
also killed the attach for the *unrelated* `tlab-trainL-s1` job. The remote job was unharmed (jobs
run server-side; it was still EXECUTING 10 min later and was re-attached with
`datasphere project job attach --id <id>`, losing nothing) — but the local log stream stopped, and
with it the **automatic output download that fires when the job finishes**. A silent loss of results,
not of compute.
**Rule:** to stop watching one job, kill by PID, or redirect that job's attach to a uniquely-named
log and match on the log path. Never match on the shared `job execute` string. Before any pkill
touching datasphere, run `pgrep -fl datasphere` and confirm the match set is what you intend.
**Detection:** the DS monitor tracks jobs by ID server-side, so it would still have reported the
terminal state — but the log going quiet for 10 minutes is what actually surfaced this. Treat a
stalled attach log as an alert, not as a quiet run.

## Watchdog for dead DataSphere attaches (added 2026-08-23 06:47)
`./ds_watchdog.sh` — re-attaches any job whose local attach log has been silent for 4+ minutes while
`job get` still says EXECUTING. Cause: the CLI poller dies on `assert current_iam_token` (auth
refresh) while the remote job runs on. Happened twice tonight (trainL-s1 via my own pkill, cuda-null
via the auth assertion). The remote job is never harmed — but the attach carries the **completion
output download**, so a dead attach silently loses the results of a multi-hour run at the very end.
Triggers on **staleness, not on the error string**, so it also catches deaths that log nothing.
Skips `*_reattach.log` files so it cannot chase itself. Logs to `ds_watchdog.log`.
