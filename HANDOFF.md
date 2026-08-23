# HANDOFF — read this first if resuming this project cold

Written 2026-08-16 12:12 MSK, right before a context compaction, so a fresh session (or a
post-compaction continuation) has everything needed without re-deriving it. This is a snapshot, not
a ledger — for the full chronological history read `LOG.md`; for the deliverable itself read
`report.md`; for original design rationale read `PLAN.md`.

## Where things stand right now

- **Nothing is running.** Checked directly (`ps aux`) at 12:12:43 MSK — no python/training/kaggle
  processes alive locally. No Kaggle kernel currently executing either (the one full run already
  completed and was pulled).
- **git is clean**, latest commit `c00265c` on `main`, nothing uncommitted, nothing staged.
- **The user's stated check-in was 10:00 MSK 2026-08-16.** Real time is now 12:12 — past that. There
  was a real, confirmed ~2.5-day gap earlier in this session too (a message assumed "3 days left" when
  only ~8h were actually left — see LOG.md 2026-08-16 02:02) so **do not trust elapsed-time assumptions
  without checking `date` directly** — this project's single biggest recurring lesson.
- **`report.md` is in a complete, internally-consistent, verified state as of commit `c00265c`.** It
  has been read start-to-finish multiple times this session for consistency after each major addition.
  Treat it as done unless the user asks for more.

## The deliverable, in one paragraph

A ≤10M-param looped (weight-tied) transformer on FineWeb, `state_renorm=False`, keeps improving well
past where prior looped-transformer work saturates (Ouro ~4, Loopie ~2, Huginn ~10). Best result:
**46.0M tokens (Kaggle T4), best val CE 3.954 at loop 8, still ahead of loop 1 (4.210) at loop 64**
(2x trained range) — 46% of the 100M-token ceiling. Confirmed on an independent seed (same direction,
0.496 vs 0.746 nats gap). A compute-matched non-looped baseline (81M params, no weight-tying) could
**not** be trained stably at all, regardless of tuning — a real negative result, written up honestly,
not smoothed over, and used to build a mechanistic account (weight-tying as implicit regularizer,
`report.md` §4.4) tying together why the looped, non-renormalizing config specifically avoids the
saturation the design prior originally expected to need renormalization to avoid.

## Key files

- `report.md` — the actual deliverable. §1 (the user's own idea narrative) is **deliberately empty**,
  reserved for them, per the task's own grading split (idea vs. implementation) — do not fill it in.
  §2–§8 are complete: task/constraints, architecture, experiments (screening §4.1, full-budget runs
  §4.2 including the Kaggle scale-up, diagnostics §4.3, compute-matched baseline §4.4), what didn't
  work §5, honest limitations §6, reproducing §7, next steps §8.
- `LOG.md` — append-only chronological ledger, ~700+ lines, every bug/fix/decision with real
  timestamps (self-corrected once already this session when a batch of timestamps turned out
  fabricated — see the 2026-08-13 entries — this project takes raw-log fidelity seriously).
- `PLAN.md` — original design rationale, phased roadmap, has "Outcome" notes appended at key
  decision points (§2 compute path, §5 ablation strategy) rather than being rewritten.
- `README.md` — repo layout + quickstart, kept in sync with `src/` as files were added.
- `checkpoints/` — gitignored except JSON histories/results and eval outputs (`.pt`/`.safetensors`
  excluded via `.gitignore`, tracked results.json/eval_*.json files are small and committed). Best
  checkpoint: `checkpoints/full_no_state_renorm_kaggle/last.pt` (46.0M-token run, 36MB, verified
  clean — 0 non-finite params across all 37 tensors, loaded fresh and checked, not assumed).
- `src/` — `model.py` (the architecture), `train.py`/`eval.py` (core), `run_screening.py`/
  `run_full.py`/`chunked_runner.py` (orchestration, all chunked-subprocess-safe), `baseline_nonlooped.py`
  and `run_second_seed.py` (added this session, see below), `test_model.py` (5 correctness checks, all
  passing, re-verify with `python src/test_model.py` if anything about `model.py` is ever touched).
- `kaggle/main.py` — self-contained Kaggle T4 kernel (used for real this session, not just prepared).

## What's genuinely done vs. explicitly left open

**Done, verified, in the report:**
- Full 5-axis screening sweep (7 arms), independently re-derived via `analyze_screening.py`.
- Full-budget local run (14.60M tokens) and Kaggle run (46.0M tokens) of the winning config.
- Second-seed replication of the headline finding.
- Compute-matched non-looped baseline (negative result, honestly reported).
- Mechanistic synthesis connecting contraction diagnostics + baseline instability.
- A real config-propagation bug (`run_full.py` silently dropping a `TrainConfig`-only arm difference)
  found, fixed, and the affected run relabeled+reused rather than discarded.

**Explicitly, consciously left undone — do not do these without asking the user first:**
- **`.git` history cleanup** (620MB of pre-`.gitignore`-fix checkpoint binaries, `git-filter-repo` is
  installed at `/opt/homebrew/bin/git-filter-repo`). This is a destructive/history-rewriting class
  operation on the only copy of this repo (no remote yet) — deliberately not run unsupervised, twice
  reconsidered and deferred both times. Needs the user's explicit go-ahead.
- **Pushing anything to GitHub or Hugging Face.** Nothing has been pushed anywhere. The project's own
  `CLAUDE.md` (in this directory, distinct from the sibling Huginn project's `CLAUDE.md` one level up
  — they do NOT share rules) says nothing goes to a remote without the user's explicit say-so.
- **§1 of `report.md`** — the user's own idea narrative. Not mine to write.
- Loop sweep past 64 (96/128) — attempted, hit real MPS/GPU driver fragility (see below), documented
  as an open item in `report.md` §8 item 3, not resolved.
- A same-budget, same-clock `state_renorm` on/off comparison (current comparisons differ in token
  count and launch time) — `report.md` §8 item 2.
- A stable training recipe for the compute-matched baseline at loss-level parity — `report.md` §8
  item 1.

## Hard-won operational lessons (read before running anything locally)

1. **This is a shared machine.** Multiple concurrent Claude Code sessions run on it (confirmed via
   `ListAgents` — at least one other active session, `operationalize-porting-directive`, doing RL
   work in a sibling `ccm-intro` project, seen consuming real CPU/memory this session). Memory
   pressure swings are often NOT caused by your own process — check with `ps aux -m` before assuming
   and before killing anything. The user's own standing instruction: prioritize not worsening shared
   swap over any single local experiment ("better OOM than endless swap").
2. **MPS/GPU driver fragility is real and has TWO distinct trigger patterns**, both documented with
   real error signatures in `LOG.md`/`report.md` §6:
   - Sustained load (~700s+) in one process → silent all-zero/NaN output, no exception. Mitigated in
     `train.py`/`chunked_runner.py` via 240s subprocess chunks + a degenerate-output check that raises
     loudly. This mitigation is solid and has now recovered from a real live failure once (see
     `report.md` §4.1's second-seed paragraph).
   - Rapid sequences of short, separate invocations (e.g. several `eval.py` calls back to back with no
     cooldown) → `kIOGPUCommandBufferCallbackError...`, either a data-look-alike NaN or a genuine hang
     requiring `kill -9`. `eval.py` never got the same chunking discipline `train.py` has — this is a
     real, still-open gap, not a one-off. If you need to run several eval-style scripts locally in a
     row, build in real gaps or reuse the chunked-subprocess pattern rather than firing them rapidly.
3. **Always build a memory-pressure guard into any new local driver loop** (check free RAM via
   `vm_stat` before launching each chunk, pause and recheck rather than launch if low — see the
   pattern in the bash driver loops used for `baseline_nonlooped.py` and `run_second_seed.py` this
   session). Don't rely on manual vigilance alone.
4. **Verify raw output, not printed summaries** — caught real bugs this session specifically by
   reading raw JSON/logs instead of trusting a script's own printed "best: ..." line (once a genuine
   summary-computation bug: `.get(1, ...)` on a JSON dict whose keys are strings not ints).
5. **Smoke-test before committing to a long run.** The Kaggle `batch_size=96` OOM and the baseline's
   multiple NaN rounds both happened because a change was pushed/launched without a quick local check
   first. Every fix after that was smoke-tested for real before the next long launch.
6. **No subagents, no Workflow tool for this project** — explicit, standing, hard constraint from the
   user at the start of this engagement. Direct tool use only.
7. **`caveman` mode is active for conversational replies** (a session hook) — terse, fragment-style
   prose in chat responses. This does NOT apply to code, commit messages, or `report.md`/`LOG.md`
   content, which stay in normal, precise prose — this distinction has been maintained consistently.

## If resuming to do more work

1. Check `date` for real elapsed time before assuming anything about remaining budget.
2. Check `git log -1` and `git status` in `tlab-loop-transformer/` to confirm you're picking up from
   `c00265c` (or later, if more happened after this doc was written).
3. Check `ps aux` for anything already running before launching new local work.
4. If continuing research (not just waiting): the report's own §8 (ranked, cheapest-first) is the
   place to look for what's next — item 2 (same-budget state_renorm comparison) or item 3 (past-loop-64
   sweep, needs `eval.py` chunking built first) are the most likely candidates.
5. If the user has responded with feedback/direction: that supersedes everything in this file.
