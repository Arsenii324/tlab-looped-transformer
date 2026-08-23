# RUNS — every job, its ID, and WHAT TO DO when it lands

WORKING doc. Written because job IDs and per-result handling live only in context otherwise, and a
compaction would leave a running job unfetchable. Update on every launch and every landing.

## How to fetch anything here
```bash
export PATH="$HOME/.local/bin:$HOME/yandex-cloud/bin:$PATH"
GRPC_DNS_RESOLVER=native datasphere --profile default project job list -p bt12q57tmrs03pnt8drc
GRPC_DNS_RESOLVER=native datasphere project job download-files --id <ID> --with-logs --output-dir /tmp/x
kaggle kernels status arsen4ikvar/<kernel>;  kaggle kernels output arsen4ikvar/<kernel> -p /tmp/k
```
A CANCELLED/ERROR job may still have recoverable files — the CLI's "Not all files can be downloaded"
warning does NOT mean nothing survived. Always look at what actually landed.

## DataSphere — project `bt12q57tmrs03pnt8drc`, profile `default` (arsen4ikvar), `gt4.1`

| ID | name | status | what to do with it |
|---|---|---|---|
| `bt1g7abps3m3ssi887bm` | **tlab-train-at-L** | EXECUTING, ETA ~04:30 | **THE 04:30 DECISION.** 5 arms L∈{2,4,8,16,32}, 10M tok each, each evaluated at its OWN L. Read from stdout (`log()` prints a TRAIN-AT-L RESULT block) or `results.json`. **If CE is monotone in L → launch a full-budget run at fixed L\*** (§4.9). If it saturates → that is the strongest negative in the project and §4.9 says so. Arms run cheapest-first, so a partial harvest still gives the low-L end. |
| `bt1immlm76fu5d2g3iat` | **tlab-gate-sweep** | EXECUTING | Fixed-gate sweep g∈{0.25,0.5,0.75,1.0}, 6M tok each. **g=1.0 IS the ungated control** (bit-identical, verified) so the sweep is self-controlling. Read the ORDERING of best-CE and optimum-location across g, not pairwise diffs. Fills §4.10. Also: its log has `date +%s` stamps around the torch-install branch and runs on `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` → **tells us whether the preinstalled-torch docker image killed the 218s setup**. |
| `bt1aep4lr92qa7jmn4ef` | **tlab-convex-gate** | EXECUTING | Learned-gate A/B vs control, paired (same seed + data order), 10M tok. Weaker design than the sweep (single pairwise diff) — treat as secondary; it answers "does the model CHOOSE to damp?" |
| `bt1ja3dvec2ait96rur6` | tlab-exit-rules | SUCCESS | **Harvested** → report §4.7. npz lost to output-collection error; numbers came from stdout (saved: `scratchpad/exitdump_run1_stdout.log`). Local regeneration queued in `run_queue.sh`. |
| `bt1tci30t1rif3ui0bi9` | tlab-timing-probe | SUCCESS | **Harvested** → DATASPHERE_NOTES timing table (setup 218.2s = 83.9%, teardown 27.2s, compute 7.2s). |
| `bt1as7hktuqkmqq1ibvl` | tlab-exit-dump | ERROR | Computed fine, failed at output collection. stdout recovered. Superseded by exit-rules. |
| `bt165evrsoo79g5e5ncq` | tlab-exit-dump (dup) | CANCELLED | Accidental duplicate from a non-idempotent resubmit. Nothing recoverable (cancelled during setup). |
| `bt18h81om1bhlvi7d201` | tlab-killtest | CANCELLED | **Deliberate.** Proved mid-compute cancel recovers stdout AND declared outputs. → DATASPHERE_NOTES. |

## Kaggle — `arsen4ikvar/<kernel>`, NOT early-stop-safe, ~9h weekly quota left

| kernel | status | what to do with it |
|---|---|---|
| `tlab-loop-fullrun` | RUNNING, ETA ~06:30 | 90M tok, current config (uniform 4-32, `state_renorm=False`). **This is the headline checkpoint.** On landing: pull output+ckpt, run `paired_eval.py score`, then `eval.py` dense sweep. Expect ~0.25-0.31 nats better than the 46.0M result (bpb 1.7330 → ~1.60). |
| `tlab-loop-normpenalty` | RUNNING, ETA ~09:30 | Same but + norm penalty λ=0.01. **Seed-matched to the control** (verified: both `seed=0`, same `torch.manual_seed` + `np.random.default_rng` paths) so this is a PAIRED training comparison. On landing: `paired_eval.py compare` against the control. **Resolves the pre-registered prediction in §4.6** — if best CE improves, training-time scale control changes the learned path; if only the optimum relocates, the clamp's account extends to training time. |

## Local MPS — serial queue, `./run_queue.sh`, log `queue_run.log`
`run_supervision.py` (3/6 arms done, seed 1 running) → `run_scale_control.py` (4 arms × 2 seeds) →
crossdepth → paired scoring ×2 → residual_scale → local exit dump → argmin_anatomy →
clamp-on-center → sandwich dynamics → 14.6M exit dump → per-arm dense evals.
Every step is idempotent (skips if its output exists) and the queue CONTINUES past a failed step.

## Harvest deadline
**18:00** — cancel and harvest anything still running on DataSphere; hard stop on new compute.

## 2026-08-23 06:09 — tlab-deep-terminal (bt18ei0ieejnek71cmtd)  [supersedes bt1qvkqg1hgo5fiovu10, CANCELLED by me]
**Question.** §4.14 found terminal-only supervision (k=1) puts the depth optimum at 0.89·μ_rec vs
dense's 0.44, at identical μ_rec — the only intervention of eight that changes the *shape* of the
depth curve instead of the model's position on it. Is 0.89·μ_rec a property of the supervision
scheme, or an artifact of μ_rec=18? Six arms: μ_rec ∈ {18,32,56} × k ∈ {1,5}, 2.5M tok each,
paired dense control at every depth so the ratio is measured, not inferred.
**Pre-registered prediction.** optimum ≈ 0.89·μ_rec → ~16 / ~28 / ~50. An optimum near 50 would be
this project's first demonstration of genuinely many useful loops. Saturation near 16-20 regardless
of μ_rec would mean §4.14 is a one-off shift, not a lever — equally worth knowing, and it would
retire the "train deeper" direction.
**Free replication.** dt_mu18_* re-runs §4.14's exact config/budget through a different code path
(DS kernel, not run_supervision_depth.py) at a third seed. If its optimum is not 16/8, §4.14 is
seed-luck and the whole §8.0c conclusion needs revisiting.
**Why the first submit was cancelled 90s in.** I estimated cost at 130k layer-applications/s from
nothing; the run's own first step reported 1864 tok/s at n_loops=9 = **50.3k layer-apps/s, 2.6× lower**.
At 5M tok/arm that is a 17.6h job that could never have finished before the deadline — it would have
burned a GPU slot all day and returned two of six arms. Cut to 2.5M tok/arm (= §4.14's budget, already
proven sufficient to locate an optimum) → 8.8h, ETA ~15:00, harvest guard at 9.5h.
**Rule this produces:** read the run's own first-step tok/s before trusting any cost estimate; a
schedule built on an assumed throughput is a guess wearing a number.

**Prediction amended 2026-08-23 06:30, before any arm of this job finished** (the μ=18 pair was still
training). `src/plateau.py` showed the 0.89·μ_rec figure above is argmin-derived and argmin on these
curves is decided at 3e-4–3e-3 nats. Re-derived on the plateau midpoint, §4.14's real ratios are
**dense 0.63·μ_rec, terminal 0.94·μ_rec** (plateaus [8,16] vs [12,24], bit-identical across seeds).
Amended pre-registration, stated in the robust statistic:
  μ_rec=18 → terminal plateau [12,24], mid ~17   (this is the replication arm; it must reproduce)
  μ_rec=32 → terminal plateau ~[21,43], mid ~30   PREDICTED
  μ_rec=56 → terminal plateau ~[37,75], mid ~53   PREDICTED
and the dense controls should sit at mid ≈ 0.63·μ_rec: ~11, ~20, ~35.
**What would falsify it:** terminal plateau midpoints that flatten toward a constant (~17 at every
μ_rec) instead of tracking 0.94·μ_rec. That would mean §4.14 is a fixed one-grid-step shift, not a
scaling lever, and would retire "train deeper with terminal-only" as a direction.
The original argmin-based prediction (16/28/50) is left above unedited so the amendment is auditable.

**Correction to the cancellation rationale above (written 06:35, after arm 1 finished).** The rule I
wrote — "read the run's own first-step tok/s" — is wrong as stated, and the 17.6h figure it produced
was wrong. `dt_mu18_term` completed 2.5M tokens in **830s**, against the 2683s my 50.3k layer-apps/s
number predicted. The first step reports ~1864 tok/s because it includes warmup and lands on an
unrepresentative loop draw (n_loops=9 of a U[4,32] schedule); steady state is ~3050 tok/s at
n_loops=13, i.e. **~119k layer-applications/s**, close to the 130k I originally assumed. The 5M-token
version would have taken ~7.4h, not 17.6h, and would have fitted the original harvest window.
**Corrected rule: take tok/s from a steady-state step several hundred steps in, and multiply by the
schedule's MEAN loop count — not by whatever n_loops the sampled step happened to draw.**
The resubmit was still the better call, for a reason I should have led with rather than the bad
arithmetic: 2.5M tok/arm is exactly §4.14's budget, which makes `dt_mu18_*` a genuine replication of
that section rather than merely a deeper-schedule probe — and it buys slack (~3.7h for six arms).

## 2026-08-23 06:31 — tlab-k-ladder (bt18m2378fugnu2lsi4h)
**Question.** §4.14 compared exactly two supervision densities (k=1 terminal, k=5 dense) and found the
useful-depth plateau moves [8,16]→[12,24] at identical μ_rec=18. Two points cannot distinguish a
**lever** from a **threshold**: k=1 might be special (nothing anchors intermediate states at all), or
depth might scale smoothly with sparsity. Five arms, k ∈ {1,2,3,5,8}, everything else identical.
**Pre-registered prediction.** Lever → plateau midpoint falls monotonically in k (~19.6, 17, 14, 11.3,
~11). Threshold → k=2..8 cluster near 11.3 and only k=1 stands apart. **Non-monotone → the effect is
inside noise** (§4.15 floor ~0.05 nats, and plateau midpoints are grid-quantised) and §4.14's
two-point result was luck. All three outcomes are informative and the third is a real falsification
route, which is why it is written down before the run rather than after.
**Design note.** Eval grid {1,2,4,8,12,16,20,24,32,48,64} deliberately includes 12 and 20 so this is
directly comparable to §4.14's sweep — the deep-terminal job's grid lacks 12 and forced a
restricted-grid comparison. k=5 is also a fourth independent replicate of the §4.14 dense arm, on CUDA.
**Cost.** ~830s/arm measured → ~70 min for five. Harvest guard 3.0h.

## 2026-08-23 06:36 — tlab-cuda-null (bt1msrvsvn3ob416ka12)
**The null §4.15 is missing.** §4.15 measured a 0.031–0.068 nat run-to-run floor from two accidental
same-config replicates — but both were **MPS**. The CUDA half of the report (§4.9 train-at-L, §4.10
gate + fixed-`g` sweep, §4.13 exploration, §4.4 baselines) currently has an **inferred** floor only
(~0.06, from the non-monotonicity of §4.10's sweep). An inferred floor cannot license "this
difference is real" on a CUDA arm — and §4.9's t/L collapse is judged against exactly that threshold.
**Design.** Three arms, `supervise_k=5`, U[4,32], 2.5M tok, **seed 0 for all three, nothing varied at
all** — verified programmatically before submit that the three TrainConfigs are identical but for
`run_name`. Whatever spread they show *is* the CUDA floor. Three rather than two, because with two
arms a single outlier is indistinguishable from a shift.
**Pre-registered.** If CUDA is deterministic at fixed seed → curves bit-identical, floor ≈ 0, and
several sub-0.05 CUDA differences currently dismissed as noise are actually real (this would require
revisiting §4.10's and §4.13's null readings). If spread ≈ 0.03–0.07 like MPS → §4.15's working rule
stands for both devices as written. Either way the report gains a measured number where it currently
has an assumption.
**Cost.** ~830s/arm → ~45 min. Harvest guard 2.0h. Slot 4 of 4.

**Prediction restated on the job's OWN grid (06:43), since midpoints are grid-dependent.** The
amended prediction above was written in §4.14's grid units; this job sweeps
{1,2,4,8,16,24,32,48,64,96,128} — no 12, no 20. Measured on its own grid, arm 1 (`dt_mu18_term`)
gave plateau **[16,24], mid 19.6**, i.e. **mid/μ_rec = 1.09**. Carrying that ratio forward on the
same grid:
    μ_rec=32 → mid ≈ 35   → plateau plausibly [24,48]
    μ_rec=56 → mid ≈ 61   → plateau plausibly [48,96]
and the dense controls, at §4.14's dense ratio scaled to this grid, should land roughly a factor
1.7 lower: mid ≈ 11 / 20 / 35.
**Falsified if** the terminal midpoints flatten toward a constant near 19.6 regardless of μ_rec —
that would mean terminal-only buys a fixed one-grid-step shift, not a scaling lever, and would
retire "train deeper with terminal-only" as a direction for this architecture.
**A plateau centred near 61 at μ_rec=56 would be the strongest answer this project can give to the
task's actual question** — genuinely many loops, all of them useful, by a mechanism (supervision
sparsity) that is a training choice rather than a fixed table, so it does not stop mattering at scale.

## 2026-08-23 07:36 — tlab-deep2-mu44  [after tlab-deep-terminal (bt18ei0ieejnek71cmtd) died on OOM]
**What the previous job delivered and what it lost.** Four arms completed and were recovered intact
(`checkpoints/deep_terminal_results.json` — the kernel persists results at every eval, so nothing was
lost with the crash; the log-parsed numbers matched the JSON exactly). The μ=56 pair was lost:
`dt_mu56_term` (U[40,72]) OOM'd on its **first forward** — 72 loops of full BPTT against a 14.75 GiB
card — and took the paired dense arm with it, which would have fit.
**Results that stand from it:**
    μ=18: term mid 19.6 CE 5.5242 gain 0.2602 | dense mid 11.3 CE 5.3329 gain 0.1051
    μ=32: term mid 32.0 CE 5.4850 gain 0.8204 | dense mid 22.6 CE 5.4148 gain 0.1733
Pre-registered prediction confirmed (useful depth scales with trained depth; the falsification route
— midpoints flattening near 19.6 — did not occur). **Unpredicted and more useful:** the terminal-only
CE penalty *shrinks* with training depth, **+0.1913 (μ=18) → +0.0702 (μ=32)**.
**This job tests only that second thing.** μ_rec=44 via U[32,56] — not the lost U[40,72], since 72
OOM'd and 40 fit, so 56 is the largest max-loop count with a fair chance. **Batch size held at 8**:
shrinking it to buy memory would change the gradient noise scale and break comparability with the
arms above, which is the entire point of a scaling series.
**Prediction.** Penalty continues to fall (≲0.05 at μ=44) → the price of the one shape-changing
intervention is controllable by training depth, the most task-relevant lead this project has.
Flattens or reverses → the μ=18→32 drop was two points and a line through them, and the direction
closes honestly.
**Hardening carried into this kernel:** per-arm `torch.cuda.OutOfMemoryError` guard so a failing deep
arm is recorded and skipped rather than killing the sweep, and `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` (a fragmentation fix that changes no computed value). If μ=44 also OOMs,
that is now a recorded data point rather than a dead job.

## 2026-08-23 07:32 — tlab-anneal-k (bt196f541abocgh4ki69)  [MY OWN proposal, not the author's idea]
**Motivation, from this project's own measurements.** §4.14 + the CUDA replication: terminal-only
reliably moves the useful-depth plateau up ~1.7× but charges an unpredictable CE cost (inside the
floor on MPS, +0.191 on CUDA). §4.16: the effect is a **threshold at k=1**, so no intermediate
density gives a partial dose. If density cannot be dosed, the remaining axis is **time**: spend most
of training dense (where CE comes from) and only the final phase terminal-only.
**Arms.** k switches 5→1 at 50% / 75% / 90% of steps, plus `an_rev50` (1→5, terminal FIRST) as the
control that separates "the LAST phase decides" from "any exposure to k=1 decides".
**First result — `an_sw50`:** midpoint **17.0** (identical to constant k=1), gain **0.2367** (2.1×
dense), CE **5.3711** vs a pooled dense reference of 5.3369 (n=5) and terminal 5.5512 (n=2) →
**+0.0342 over dense, −0.1801 vs terminal: 84% of the penalty recovered while keeping full depth.**

## 2026-08-23 08:00 — tlab-anneal-rep (bt18vgsamq3qpaqddqeu)
**Fixes the two weaknesses of the above.** (1) Every dense number `an_sw50` was compared against came
from a *different job*, and the measured cross-job dense spread (0.0074–0.0334) is the same order as
the effect claimed — so the control now runs **in the same job**, same shard, same tokenizer. (2) n=1
→ two seeds per condition. Four arms: `a2_an50_s{0,1}`, `a2_dense_s{0,1}`.
**Pre-registered.** an50 midpoint 17.0 at both seeds (depth is the robust part — it has reproduced
identically across 6 arms and 2 devices); an50 CE above dense by ~0.02–0.05, far below the ~0.21 that
constant terminal-only costs. **Falsified if** an50's midpoint returns ~11.3 (dense-like), which
would make the 17.0 a one-run fluke and close the annealing lead entirely.

## 2026-08-23 09:08 — tlab-deep-anneal (bt135694q8u861ealnlq)
**Combines the project's two strongest threads, neither of which has been crossed with the other.**
*Deep-terminal (3 points):* terminal-only's useful-depth midpoint tracks μ_rec at 1.09/1.00/0.98 for
μ_rec 18/32/40; at μ_rec=40 its useful band is loops **32–48**, but CE is 5.6051 vs the dense
control's 5.4170. *Annealing (§4.17):* switching to k=1 only for the final phase keeps the depth
shift and cuts the price (constant terminal +0.21; annealed-50% +0.041 avg over two seeds).
**Arms.** `da_mu40_sw90` / `da_mu40_sw75` / `da_mu40_dense`, all μ_rec=40 via U[32,48] — the deepest
schedule that FITS (U[32,56] and U[40,72] both OOM'd at batch_size=8, and batch size is held fixed so
these stay comparable). Eval grid extended to 128 with a point at 40. One seed each, OOM guard inherited.
**Pre-registered.** sw90/sw75 land at midpoint ≥ 30 (well above dense's 22.6) with CE within ~0.05 of
the in-job dense control. **Falsified if** their midpoints return near 22.6 (annealing does not carry
to deep schedules, and §4.17 would be specific to μ_rec=18), or if CE degrades toward the
constant-terminal 5.6051 (the price returns at depth).
**Why it is the most task-relevant run left:** a useful band near 40 loops at close to dense loss is
the brief's literal objective — low perplexity *by exploiting many loops* — rather than a proxy.

## 2026-08-23 10:15 — tlab-anneal-scale (bt1ke2cf3rkrde54j9q5)  **the budget-robustness test**
**Why this is the most important run left.** §4.17 is confirmed at 2.5M tokens (sw90 beat its in-job
dense control at both seeds, −0.0811/−0.0609). But this project has a hard lesson about screening
budgets: the norm penalty gives **−0.366 nats at 2.5M and only −0.030 at 90M** — a 12× shrinkage.
An effect measured at 2.5M is not evidence about the budget that matters, and §8 currently puts
annealing forward as this report's answer.
**Design.** Same comparison at **10M tokens** (4×), in-job dense control, one seed — the axis under
test is the budget, not seed noise, and 2.5M already has two seeds. ~2.5h.
**Pre-registered, three outcomes.** Advantage holds near −0.03…−0.08 with a deeper plateau → the
mechanism is budget-robust and §4.17 can stand as a design recommendation. Advantage shrinks toward
zero the way the norm penalty did → §4.17 is a small-budget phenomenon and must be labelled so, which
would become the single most important caveat in this report. Advantage reverses → §4.17 is withdrawn
as a recommendation and kept only as a measurement.

## 2026-08-23 10:17 — tlab-deep-anneal2 (bt14t423fvej1dsivv4g)
Seed 1 of the μ_rec=40 annealing arms. Seed 0 gave `sw90` plateau **[24,48]** (mid 33.9) at CE 5.4394
— a useful band spanning loops 24–48 — but that is one seed, and §4.17's own replication showed
`sw75`'s CE advantage **flipping sign** between seeds while its depth held exactly.
**Prediction is deliberately split:** the PLATEAU should reproduce (depth has been the robust quantity
in every replication so far — identical to the digit across 6 arms and 2 devices); the CE gap is the
part that may not. Judging both by the same standard would waste the one thing these replications
have reliably taught.

## 2026-08-23 10:29 — tlab-deep-full (bt1vqefjccioapof5fgh)  **the deep full-budget artifact**
Prompted by an external reviewer's point that every headline so far is a randomized U[4,32] model
swept at INFERENCE, which answers "how deep can this model usefully go" rather than "can a model that
must run many loops be good". Agreed. One arm, all tokens that fit before the 18:00 cutoff.
**Config differs from what was proposed, deliberately.** The proposal was fixed L with CONSTANT
terminal-only. The paired μ_rec=40 data finished after that message was written and says constant
terminal-only is dominated: annealed sw75 reaches plateau **[32,64] mid 45.3 at CE 5.4466 (+0.030 vs
dense)** where constant terminal-only reaches [32,48] mid 39.2 at 5.6051 (**+0.188**). Deeper band,
sixth of the cost. Schedule `U[32,48]`, so **every training step runs ≥32 loops** — trained deep
throughout, not a shallow model swept deep. Chosen over fixed L=40 because U[32,48]+annealing is the
cell actually measured; fixed-L+annealing has never been run and scaling an unmeasured config to full
budget is what this project has a rule against.
**Pre-registered on the plateau, before launch:** midpoint ≥32 (point prediction 45, range 32–64);
**absolute CE ~+0.5 nats WORSE than the 90M headline** (~25M vs 90M tokens; 0.398 nats/e-fold →
CE ≈ 4.12, ppl ≈ 62 vs 3.6146/37.14) — *this is the expected outcome, not a failure of the config*;
**falsified if** the midpoint returns near 22 (dense-like), i.e. annealing does not survive a 10×
budget — the live risk, with precedent in the norm penalty's −0.366 → −0.030 shrinkage from 2.5M to 90M.

## PRE-REGISTRATION for `tlab-anneal-scale`, written 2026-08-23 11:55 — BEFORE the control landed
**Why the existing falsifier is insufficient.** It asked only whether the CE advantage *shrinks toward
zero*. The norm penalty shows that is the wrong question: decomposed, it does not shrink, it
**reverses**. ΔCE@1 goes **−0.2196 at 2.5M → +0.2263 at 90M**, a sign flip of 0.45 nats, while
ΔCE_best merely shrinks (−0.366 → −0.030). That is a **regime change**, and §4.12 supplies the
mechanism: loop gain has to *emerge* with tokens. At 2.5M the control's gain is 0.1056 — almost no
depth utility exists — so the penalty acts as a general optimisation aid and helps everywhere. At 90M
the control's gain is 0.3047 — depth utility exists — and the penalty becomes depth-*specialisation*
pressure, buying the deep end by selling the shallow one.

**This strips the discriminating power from the 2.5M annealing numbers.** At 2.5M, `sw90` gives
(ΔCE_best, ΔCE@1) = (−0.0811, −0.0416) and (−0.0609, −0.0277) — **both endpoints improve, which is
exactly the shape the norm penalty had at 2.5M before it reversed.** At the screening budget the two
interventions are indistinguishable in shape. So this run is not a magnitude check; it is the first
look at whether annealing follows the penalty into reversal.

**Read the result on `(ΔCE_best, ΔCE@1)` against the in-job control — never on `Δgain` alone.**
Reference trajectory: norm penalty `(−0.366, −0.220)` at 2.5M → `(−0.030, +0.226)` at 90M.

| outcome | signature | what it means |
|---|---|---|
| **A** | both endpoints still improve (`ΔCE@1 < 0`) | annealing is **not** the penalty's phenomenon. §3.5 stands and strengthens — this is what distinguishes a real advance from depth-specialisation |
| **B** | `ΔCE_best` holds, `ΔCE@1` **flips positive** | **same regime change as the penalty.** Annealing is a **trade**, not a free win. §3.5 survives for a task that scores depth, but must say so plainly, and the μ_rec=18 "better on both axes" claim is a small-budget artifact |
| **C** | both collapse toward zero | small-budget phenomenon; §3.5 rewrites around a deep dense schedule |

**B is the outcome the penalty's evidence makes most likely, and the previous falsifier did not name
it.** Recorded now so it cannot be chosen after the fact.

## SECOND PRE-REGISTRATION for `tlab-anneal-scale`, written 2026-08-23 12:40 — still before the control landed
**A second, independent discrimination the run can settle, proposed by the reviewer and recorded before
the data.** §4.17's rule is a **fraction** (terminal for the last 10–25% of steps). The mechanism-derived
alternative is a **token-keyed** rule, and the two are not the same thing:

*§4.12 measures that loop gain **emerges** over an absolute token scale — 0.02@0.44M → 0.10@1.92M →
0.20@9.24M → 0.24@13.57M, flattening near 10–15M. Building the loop machinery therefore has an
absolute cost. Releasing the anchor (§4.17's terminal phase) re-shapes a trajectory the model already
has, which is plausibly also absolute rather than proportional. If both are absolute, a **fraction** is
the wrong parameterisation and only looks right at the budget it was fitted to.*

| | fraction rule (last 25%) | token rule (dense until ~12M, then terminal) |
|---|---|---|
| 2.5M | terminal from 1.9M | budget < build time → **predicts annealing should not work at all at 2.5M** |
| **10M** | terminal from 7.5M | terminal from ~10M → **predicts a WEAKER effect at 10M than at 2.5M** |
| 90M | terminal from 67M | terminal from ~12M → 83% of training terminal |

**The awkward part, stated because it is the honest one:** the token rule predicts the 2.5M result
*should not exist*. It does exist and it replicates at two seeds. So either the build requirement is
softer than §4.12's curve implies, or "release" is not extending a built machine but something closer
to removing a regulariser. Both readings are interesting and the second is the more surprising.

**What this run discriminates.** Under the **fraction** rule the effect at 10M should hold at roughly
the 2.5M magnitude. Under the **token** rule it should be **weaker at 10M than at 2.5M**, because 25%
of 10M puts the switch at 7.5M — later in absolute terms but still inside the build phase.
Read alongside the A/B/C outcome registered at 11:55; these are orthogonal questions (that one asks
*which endpoints move*, this one asks *whether the fraction parameterisation transfers*).

---

## 2026-08-23 13:30 — PRE-REGISTERED READ for `tlab-deep-full`, written before it lands

**Why this is being written now.** The job's own description records its config: **`sw75`, μ_rec=40**.
Both of those are, as of today, the settings where annealing is measured to be *damage-driven*:

- **Switch fraction.** §3.5 was narrowed today from "~10–25% of steps" to **`sw90` specifically**,
  because the four-pair decomposition shows `sw75` is damage-driven at seed 0 (ΔCE_best −0.0656,
  ΔCE@1 **+0.0185**) and **worse on CE_best outright** at seed 1 (**+0.0906**). Only `sw90` improves
  the ceiling with loop 1 undamaged, at both seeds.
- **Schedule.** At μ_rec=40 both annealed arms are damage-driven against their in-job dense control
  (ΔCE@1 +0.0749 / +0.1749), and both ΔCE_best values sit *inside* the 0.0541 CUDA terminal replicate
  floor — so at that schedule the ceiling gain is not resolvable while the loop-1 damage is.

**So this artifact is the weaker variant at the harder schedule.** That was not known when it was
launched (the switch-fraction decomposition was run today, ~6h after submission). Stating it now so
the harvest is not read more favourably than the design supports.

**How to read it when it lands — committed in advance:**

| outcome | reading |
|---|---|
| plateau midpoint **≥32** and CE_best beats the dense reference | the deep half of §3.5's table stands. Still report ΔCE@1 alongside; a deep band bought with loop-1 damage is a trade, not a win |
| plateau midpoint **≥32** but CE_best within the replicate floor | **the expected outcome given the above.** Report as *band relocation without a resolvable ceiling gain* — consistent with every other rate-intervention in §4.6/§4.10/§5.0, and NOT as evidence annealing wins at depth |
| plateau midpoint returns near **22** (dense-like) | the deep half of §3.5 is **withdrawn**, as registered at 10:29 |
| job returns curves only *(certain — its config predates the `outputs:` fix)* | no weights, so no exit-rule or angular follow-up on it; §4.7a's matched pair already covers the exit question at 2.5M |

**What it cannot settle.** It is one seed with no in-job dense control at the same budget — the
comparison would be against a *different* run, which is the unpaired-comparison error this report has
already made twice (§6.0). Its useful claim is about the **band**, not about ΔCE against anything.

**Not relaunching as `sw90`.** ~6h of T4 time is already spent, the remaining window does not fit a
second deep run, and the band question it answers is genuinely schedule-level rather than
switch-fraction-level. Recorded as a decision, not an oversight.

---

## 2026-08-23 14:00 — PRE-REGISTERED: the scale clock (`scale_clock`), written before any arm runs

**What changed about the idea before it was implemented.** Proposed by the reviewer as *breaking a
fixed point* of the induced sphere map `G(u)=F(u)/‖F(u)‖`, on the premise that `u_t → u*`. **That
premise was tested first and is false** — `src/angular_convergence.py` shows `u` drifts
logarithmically (log-drift R² 0.986 with one parameter, power law 0.748 with two; 0.18 rad of motion
still accumulating over loops 129–384). There is no fixed point to break. The intervention is run
anyway on the surviving, weaker motivation: **the block cannot see `‖h‖`, so it has no way to behave
differently late than early**, and logarithmic drift means readout-visible progress per loop decays
as 1/t. Precedent that a *trained* scale coupling can move the path where an inference-time one
cannot: §4.6's clamp relocates without raising the ceiling, while §4.6b's norm penalty is the only
arm whose ρ crosses below 1 and whose drift constant falls (C 0.154 → 0.102).

**Implementation.** `h_in *= 1 + w·(log rms(h_t) − log rms(h_1))`, per token, `w` zero-initialised —
**+448 params (0.0049%), zero extra FLOPs, bit-identical to the current model at step 0** (verified,
max|diff| = 0.0). A function of state, not a table over `t`, so §3.4-compliant and extrapolates.

**Arms** (local MPS, 2.5M tokens, seed 0, in-job control so the comparison is paired):
`sc_ctrl` (clock off) · `sc_clock` (clock on) · `sc_clock_sw90` (clock + annealing).

**Read, in this order — geometry first, because CE at 2.5M has already reversed between budgets:**

| rank | quantity | baseline | what confirms the mechanism |
|---|---|---|---|
| **1** | drift constant **C** (`angular_convergence.py`) | 0.1539 (90M ctrl) / measure `sc_ctrl` in-job | **C rises** — the block is using the coordinate to keep moving |
| **1b** | consecutive-step exponent | ≈ −1.0 | moves **away from −1** toward 0 |
| 2 | learned ‖w‖ | 0 at init | **non-zero** — if it stays ~0 the model declined the clock and every other reading is moot |
| 3 | plateau midpoint / onset, grid-matched | `sc_ctrl`'s own | extends |
| 4 | CE_best, ΔCE@1, loop-1 share via `gain_decomp` | `sc_ctrl`'s own | reported, not used to decide |

**Pre-registered falsifier, and it is the informative outcome.** If **C rises and the plateau does
not extend**, then readout-visible supply was never the binding constraint and demand is — which
would be the cleanest statement of the supply/demand split in this report, and would justify §3.5's
supervision-based method on mechanism rather than on measurement alone. **If ‖w‖ ≈ 0**, the model
declined a strictly-larger hypothesis class and the honest report is "no effect, and the model would
not even take the parameter."

**Not pre-registering a CE threshold**, deliberately: 448 params against a measured MPS replicate
floor of 0.031–0.068 nats means any CE difference at 2.5M is unresolvable by construction. This run
is a geometry experiment.

---

## 2026-08-23 ~14:10 — `tlab-anchor-tokenkey` · **`bt1hp97su48dc6096sqn`** · DataSphere
Kernel: **`ds_scaleclock/`** (in-repo, scrubbed; the launched copy lives in the session scratchpad
because its `config.yaml` carries the live wandb key — `DATASPHERE_NOTES.md` rule).
Arms, cheapest first so an early cancel still answers the falsifier:
`ak_sw90_k5` / `ak_sw90_k3` / `ak_sw90_k2` / `ak_dense_k5` (2.5M each) → `tk_frac90_10M` /
`tk_tok225_10M` (10M each). Read against the pre-registration in this file at 14:00 and §4.18.
Harvest: `datasphere project job download-files --id bt1hp97su48dc6096sqn --with-logs --output-dir …`
(prefix every DS call with `GRPC_DNS_RESOLVER=native`). Also on wandb, project `tlab-loop-transformer`.

## 2026-08-23 ~14:40 — `tlab-seed-extension` · Kaggle `arsen4ikvar/tlab-seed-extension`
Kernel: **`kg_seeds/`** (in-repo). Extends the in-job (sw90 − dense) paired difference to **seeds 2
and 3**, giving n=4 estimates of the quantity §3.5 rests on rather than a replicate "floor".
**Pre-registered:** if the four paired differences straddle 0, or their mean falls inside the 0.0541
CUDA terminal floor, §3.5's annealing recommendation is withdrawn to "not resolved at this budget".
Harvest: `kaggle kernels output arsen4ikvar/tlab-seed-extension -p <dir>`.

**Provenance note, because this is the first time it has applied.** All 26 earlier `ds_*/` kernels
were tracked before launch. These two were launched from the scratchpad (live key) and their scrubbed
copies committed after the fact — so the tracked file is byte-identical to what ran **except** for
`WANDB_API_KEY`. §4.7a's inputs (`exitdump_*.npz`, 2 × 278 MB) remain scratchpad-only and are NOT
recoverable after the session; the derived exit-rule outputs are persisted in
`checkpoints/exit_rules_annealed_pair.json` so the published numbers stay traceable to an artifact.

---

## 2026-08-23 17:40 — PRE-REGISTERED READ for `od_depth_gate` (arm 4/4, running locally)

Gemini's `depth_gate_mode="state"`: stack all `n_loops` states, score each with `Linear(H,1)`,
softmax over loops **per token**, mix, and replace the final readout with the mixture.
**+448 params.** This is the `gate_state` experiment two reviewers independently asked for, and it is
the only remaining candidate for a *positive* result. Two caveats that change how it must be read,
recorded before the number exists:

**1. Step 0 is NOT the control.** The head is zero-initialised, so all gate logits are 0 and the
softmax is **uniform over loops** — i.e. the arm begins at the *uniform mixture*, not at the
final-loop readout the control uses. So this is not a strictly-larger-hypothesis-class test the way
the scale clock and gated injection were. §4.7c measured uniform[1,16] at −0.0015 against the best
single depth, so the starting point is not badly off — but "beats the control" here mixes the gate's
benefit with the uniform-start offset, and cannot be attributed to the gate alone.

**2. Its eval curve is not a depth curve.** The gate applies whenever the full state stack is
available, so evaluating at `n_loops = r` mixes over loops `1..r`. **The arm's "plateau" is therefore
over mixture-window size, not over depth**, and is not directly comparable to every other plateau in
this report. Compare it to the control on CE; do not put its band in the §4.16b/§4.17 band tables.

**Read, in order:** (a) do the learned gate weights concentrate or stay near uniform — if near
uniform, the model declined the parameter and it reduces to §4.7c's null; (b) if concentrated, on
which loop, and is it token-dependent (that is §4.7's unreachable signal becoming reachable, which
would be the project's first positive on that axis); (c) CE against `od_control`, floor 0.0527;
(d) **do not** read the plateau as a depth band, per caveat 2.

**Prior, stated honestly:** §4.7c found no *static* mixture helps, and §4.7's headroom is per-token
while a static mixture is global — so a per-token gate is exactly the instrument that could reach it,
and E1's null is a *lower* bound on this, not an upper one. But four instrument classes have now
failed on this headroom, so the base rate is not encouraging.

## 2026-08-23 18:19 — PRE-REGISTERED READ for `tlab-duocausal-s0` / `-s1` (launched, no data exists yet)

Two DataSphere T4 jobs, seeds 0 and 1, **four in-job paired arms each**, 3.5M tokens/arm, one grid
(11-point), `U[4,32]`, `supervise_k=5`:
`dc_control_s{0,1}` · `dc_w2_s{0,1}` · `dc_w3_s{0,1}` · `dg_norm_s{0,1}`.

**Why these two mechanisms and not another.** Every instrument this project has aimed at the per-token
depth headroom is **readout-side** -- label-free rules (§4.7), static readout mixtures (§4.7c), the
annealed retest (§4.7a), the oracle-depth cache (§4.8b), and the learned gate (§4.22). All five read
the *finished* trajectory and select or blend it. **None changes what the block sees at loop t.**

1. **Duo-causal attention, `kv_window` = 2 and 3 — ZERO added parameters.** At loop t each layer
   attends over the K/V of its own inputs from loops t-W+1..t, concatenated on the key axis under a
   token-causal mask replicated across depths (Think-at-Hard, arXiv 2511.08577, verified from
   tarball). Recurrence-side. Motivated by this project's own §4.3 anchor result: the forcing bias
   exceeds the model's per-step motion from ~loop 2, so the update is largely history-independent --
   and history is what the block cannot compute from `h_t` alone.
2. **Scale-invariant depth gate, `state_norm` (+449 params).** §4.22 measured the existing gate as
   unable to express a mixture: its logits are `w·h_t` on the RAW state, ‖h‖ grows 1.8-4.0x per pass
   and ~1e3 over training, so the softmax saturates to a hard argmax (effective loops mixed
   1.01-1.05 of r). This scores the DIRECTION (`w·h/‖h‖`) times a learned scalar temperature, so the
   model chooses its own sharpness. It is the two-line version that actually tests the hypothesis.

**Correctness gates run BEFORE launch** (a null from a broken arm is meaningless, not informative):
`kv_window=1` is **bit-identical** to the untouched model (max|diff| = 0.000e+00); `W=2` and `W=3`
provably change the forward (7.6e-01, and W3≠W2); duo-causal adds **zero** parameters
(9,064,608 → 9,064,608); loop-1 logits are unchanged at W>1 (no history exists yet); gradients reach
`k_proj` through the extra keys; `state_norm` = `state` + exactly 1 param; **both gates start at a
uniform mixture** (effective loops 8.000/8 at init). All four arms construct and forward finite from
the *generated* driver, not just from `src/`.

**Reads, in order, decided now:**

- **(a) PRIMARY, task axis — the plateau band vs the in-job control, grid-matched.** This is the axis
  the brief scores and the one **nothing in this report has ever widened** (eight interventions).
- **(b) PRIMARY, mechanism — `cos(du_t, du_{t−1})`, post-hoc on the returned checkpoint.** §4.3
  measures it at **0.9999**: the state travels a near-straight ray. If the block can finally see its
  own history, that increment should stop being a near-constant vector. **This read is independent of
  CE**, which matters because CE at screening budgets has reversed on this project twice.
- **(c) CE_best vs in-job control**, against the 0.0150 CUDA-dense floor.
- **(d) Dose-response across W = 1 → 2 → 3.** Monotone is the signature of a real effect; non-monotone
  across a swept parameter is the signature of noise (§4.10's own reasoning). **This is why W=3 is in
  the sweep rather than W=2 alone.**
- **(e) Gate: effective loops mixed.** ~1.0 again ⇒ saturation was never scale-driven. ~r with no CE
  gain ⇒ it declined to discriminate and this reduces to §4.7c's null. Intermediate **and** band
  widens ⇒ the first positive on the per-token axis.

**Falsifiers, written before the data:**
- Band unmoved at **both** seeds and `cos` unchanged ⇒ **the eighth and ninth nulls**, and the
  report's central negative becomes materially stronger: it would then span **both** readout-side and
  recurrence-side families rather than only the one it has tested.
- `cos` falls but the band does not widen ⇒ **supply-side fixed, demand binding.** That is the
  cleanest statement of the supply/demand split this report has, and a *better* result than a small
  CE win.
- CE regresses > 0.05 at both seeds ⇒ kill, and it becomes the fourth instance of "the model is given
  the mechanism and gets worse".
- Any effect that appears at one seed and reverses at the other ⇒ **not reported as a result.** This
  project has withdrawn two claims for exactly that.

**Explicit outputs, named file by file, never a glob** — §6.0 row 34 cost this project the DataSphere
depth-gate weights this morning, and read (b) requires the checkpoints.

### Addendum to the 18:19 pre-registration — source verified myself, and one honest difference

Checked in `papers/sources/2511.08577` rather than relayed (§6.0 row 22's rule):

- `3_method.tex:105,168` — duo-causal *"lets tokens attend across both previous positions and shallower
  iteration depths, maintaining 2D causality."* That is exactly the mask implemented here: token-causal,
  replicated across depths.
- `3_method.tex:174` — *"When all tokens iterate only once (as in standard transformers), this reduces
  to regular causal attention."* **This is the property the pre-launch gate checks**, and `kv_window=1`
  reproduces the untouched model at `max|diff| = 0.000e+00`.
- `3_method.tex:163-165` — the dilemma is *"requiring up-to-date context from all previous tokens
  [while] maintaining parallel training where depth-d computations cannot depend on uncomputed deeper
  states."* Attending only to **shallower** depths is what preserves full training parallelism, which
  is why this is affordable at all.
- `4_experiment.tex:277-280` — their own ablation: replacing duo-causal with *"attending only to the
  first iteration"* costs **5.4%**, with *"attending only to the current iteration"* (what this project
  does everywhere) costs **8.5%**. That is their measured size for the mechanism, at 1.7B.

**The difference between their mechanism and this arm, stated before the data.** Theirs attends to
**all** shallower depths; this arm attends to a **window** of the last `W-1` (W = 2, 3), because
storing every depth's K/V is the O(r) memory that already OOM'd §4.22's gate at a 13.04 GiB cap.
**So a null here bounds the windowed form, not the full triangle** — and W = 2 vs 3 is in the sweep
precisely so the dose-response says whether more window is buying anything before anyone pays for the
full version. If W=3 > W=2 > W=1 monotonically, the full triangle is worth someone's compute; if
W=3 ≈ W=2 ≈ W=1, it is not, and that is a more useful negative than one window would give.

**Job IDs (recorded immediately — a compaction loses these):**
`tlab-duocausal-s0` = **bt1qvi35v7gsejmvn1it** · `tlab-duocausal-s1` = **bt1lkbri6cqj6q9fssoa`**
(launched 18:19, gt4.1 T4, ~2h). Kaggle LoRA scale-up = slug **arsen4ikvar/tlab-lora-scaleup**
(pushed 18:23, 2 in-job arms × 12M tokens; **Kaggle returns output only on completion**, so it must
finish — MAX_SWEEP_SECONDS 4.2h is the internal guard).
Harvest with `datasphere project job download-files --id <id> --output-dir <dir>`; the checkpoints
are declared **by explicit filename**, so unlike this morning's job they will actually come back.

## 2026-08-23 18:33 — `tlab-recmethod-s2` (**bt1s4mag4kdvsvts536m**), pre-registered before data

**Purpose: the method §3.5 recommends has no checkpoint. This makes one.** Two in-job arms,
10M tokens each, seed 2, gt4.1: `rec_dense_s2` (control) and `rec_sw90_s2`
(`supervise_k_final=1`, `supervise_switch_frac=0.90`). Outputs declare **both `.pt` files by explicit
name** (§6.0 row 34).

**Verified before launch that the intervention is not inert** — this project has already had one
annealed run that silently did nothing (§6.0, the 727-second no-op). `effective_k` traced through the
*generated* driver: total 4,882 steps, switch at 4,393; dense returns k=5 at every step; sw90 returns
k=5 through step 4,344 and **k=1 from step 4,394**. It engages.

**Pre-registered read.** Primary purpose is the **artifact**, not a CE verdict: `tlab-anneal-scale`
already ran this comparison in-job at 10M (outcome A, ΔCE_best −0.0764) and returned no weights. So
(a) the deliverable is `rec_sw90_s2_last.pt` — the first weights of the recommended configuration at a
non-screening budget; (b) the paired ΔCE_best is a **third budget point** for a claim whose CE half is
withdrawn at n=4 and is read as such, not as a resolution; (c) the band is read grid-matched against
its own in-job control. **It is not a perplexity headline** — 10M tokens against the shipped 90M
checkpoint, ~0.5-0.8 nats behind by this report's own scaling interval, by construction.

### 2026-08-23 18:53 — HARD INSTRUMENT GATES added to the 18:19 pre-registration, still before the data

Raised by an external reviewer and adopted, because it is **this project's own §5 house rule applied
to the instruments I just caught failing**: *no hypothesis may be retired or confirmed by an
instrument that has not itself passed a null.* The running arms can each produce a CE number that
would be uninterpretable, and without these gates it would be interpreted anyway.

**GATE 1 — the scale-invariant depth gate (`dg_norm`).** Its CE is **uninterpretable as a mixture
result** unless the mixing actually happens. Registered threshold: **effective loops mixed
`exp(H(gate_weights))` ≥ 1.5**, measured on the returned checkpoint the same way §4.22 measured the
broken one (which came back at **1.01–1.05 of r**, 95–98% of tokens above 0.99 top-weight).
- **< 1.5 ⇒ it is a hard selector again**, scale was not the binding constraint, and **no CE claim is
  made from that arm** — it reports as a second instrument failure, not as a depth result.
- **≥ 1.5 ⇒ the instrument works** and its CE and band are readable as a per-token mixture test.

**GATE 2 — duo-causal (`dc_w2`, `dc_w3`).** The primary read is **`cos(du_t, du_{t−1})` against
§4.3's 0.9999**, not CE. If the cosine does not move, the block did not use the history it was given,
and **a CE null is a null on a mechanism that did not engage — not a null on the hypothesis.** Those
are different findings and only the second is worth anything.

**Why this is registered rather than applied afterwards:** both gates can only *disqualify* an arm's
CE, never rescue it, so writing them down now removes the option of reading a favourable CE from an
arm whose mechanism is inert. That option is exactly what §6.0 rows 3, 5 and 16 are about.

### 2026-08-23 19:02 — GATE 2 AMENDED: three cases, not two. Still before the duo-causal arms finish.

The 18:51 gate named two outcomes for `cos(du_t, du_{t−1})` — unchanged, or falls. **That is
incomplete, and the missing case is the most interesting one.** Registered now:

| outcome | reading |
|---|---|
| **cos ≈ 0.9999, unchanged** | the block did not use the history it was given. A CE null here is a null on a **mechanism that did not engage**, not on the hypothesis |
| **cos FALLS** | engaged as intended — the update stopped being a near-constant vector. CE and band are then readable as a test of the hypothesis |
| **cos RISES above 0.9999** | **engaged and pushed the wrong way** — feeding the block its own recent history made the update *more* self-similar, not less. This is a stronger negative than "no movement" and is the same shape as the scale clock (§4.19): the model takes the mechanism and the trajectory gets worse. It would also *explain* a CE regression rather than just accompanying one |

**Why this matters for the interim already in hand.** At step 244 seed 0's `dc_w2` best sits at
**r = 4** against the control's **r = 8** — the band moved *inward*, not merely down. Combined with
+0.1632, that is the signature of an intervention making the trajectory settle **sooner**, which is
what the third case predicts. Registered before the arms end so the third case cannot be adopted
after seeing which way the cosine went.

**A confound to state whichever way it comes out — this arm may be handicapped rather than refuted.**
(1) RoPE here is **depth-invariant by construction** (positions are token-indexed, as Think-at-Hard
specifies), so the extra keys carry **no marker of which depth they came from**; the block must infer
depth from content alone. A learned per-depth key offset would fix that and costs parameters, which
this arm deliberately does not spend. (2) Think-at-Hard applies duo-causal as a **fine-tune on a
pretrained 1.7B backbone with LoRA adapters**, not from scratch at 9M parameters on 3.5M tokens — and
the pathway has **zero dedicated capacity** here, so the model must repurpose existing weights to use
it. **A null at this budget therefore bounds "windowed duo-causal, from scratch, at 3.5M tokens, with
no depth marker and no added parameters" — not the mechanism.**

## RUN -> CODE provenance (recorded 19:09; the frozen drivers are archived IN the repo)

DataSphere jobs carry a **frozen inlined copy** of `src/model.py`, so a job is immune to later
edits of the tree — which is why the 18:57 change of `cond_fixed_branch` from bool to int could
not affect the already-running diversity job (verified: its frozen driver pins to branch 0, the
pin2 driver pins to 2, each self-consistently). Archiving the drivers so a run can be re-derived.

| job | id | driver md5 | archived at |
|---|---|---|---|
| `ds_dc_s0` | bt1qvi35v7gsejmvn1it | `3cc015eb5307` | `runs_frozen/ds_dc_s0/` |
| `ds_dc_s1` | bt1lkbri6cqj6q9fssoa | `67376853c404` | `runs_frozen/ds_dc_s1/` |
| `ds_recmethod` | bt1s4mag4kdvsvts536m | `8be2244e2ab8` | `runs_frozen/ds_recmethod/` |
| `ds_diversity` | bt1ps6o54qhrecg40etf | `038d23fe5509` | `runs_frozen/ds_diversity/` |
| `ds_pin2` | pending | `18bc3ef487bc` | `runs_frozen/ds_pin2/` |

Tree SHA at archive time: `b218613`. `src/model.py` at that SHA is the source the last of these was generated from.

## 2026-08-23 19:15 — PRE-REGISTERED: `tlab-xsa-s0`, and the prediction comes from this report's OWN regularity

**Exclusive Self Attention** (arXiv 2603.09078, verified in `papers/sources/`):
`z_i = y_i − (y_iᵀv_i)·v_i/‖v_i‖²`. **Two lines, ZERO parameters.** Two in-job arms, 2.5M tokens,
seed 0, gt4.1: `xsa_control_s0` vs `xsa_on_s0`.

**Why it is worth a slot even though the outcome is predicted.** §4.3 measured *their* phenomenon
here: `cos(y_i, v_i)` rises monotonically with loop index (0.29 → 0.36 across loops 2→64, all three
layers), so the quantity XSA removes **is present and growing in this regime**. It is also a **third
axis** the taxonomy did not have — not readout-side, not recurrence-side, but **inside the operator**,
changing what attention writes.

**THE PREDICTION, from this report's own eight-intervention regularity, registered before the run:**
> **CE improves; neither band edge moves.** Operator diversity did exactly this (−0.0857 replicated,
> both edges identical in 5 of 5 pairs, ~90% of the gain at `r = 1`). XSA is another *block*
> improvement, and §8's dissociation says block improvements do not buy depth.

- **Confirmed (CE down, band unmoved)** ⇒ the **ninth** instance, from a published zero-parameter
  operator whose outcome was predicted in advance. That makes the dissociation very hard to dismiss.
- **Violated (band widens)** ⇒ **the most interesting result of the day**, and the first thing in this
  project to move the band without touching the loss schedule.
- **CE regresses** ⇒ a fourth "the model takes the mechanism and gets worse", alongside the scale
  clock, gated injection, and (interim) duo-causal.

**Pre-launch gates.** `xsa=False` is **bit-identical** to the untouched model (max|diff| = 0.000e+00);
`xsa=True` changes the forward (2.38); parameter count unchanged to the digit (9,064,608 both); and
the operator is verified against the paper's equation directly — after projection
**`cos(z, v) = 2.85e-09`** and `max|⟨z,v⟩| = 2.2e-06`, i.e. the self-value direction really is removed.

### 2026-08-23 19:20 — `tlab-xsa-s0` prediction SHARPENED by its own null, before any result exists

The arm launched 19:15; this null was measured 19:20; its results land ~20:05. **Amending the
registered prediction now, with the timestamps stated, rather than reinterpreting afterwards.**

`src/attn_self_bias.py` on an **untrained** model of the same config: `cos(y_i, v_i)` reaches
**0.83–0.85** by loop 64, against the trained model's **0.35**. The attention-similarity bias is
**architectural, and training already suppresses most of it.**

**So the sharpened prediction is that XSA is near-NULL ON CE TOO, not merely on the band** — there is
much less self-value component left for its operator to remove than the untrained geometry (or the
1.3B model in their Figure 1) would suggest. The original registration said "CE down, band unmoved";
**the null says the CE half is now doubtful and the band half is unchanged.**

- **Near-null on both** ⇒ consistent, and the *reason* is measured rather than assumed.
- **CE down anyway** ⇒ the original prediction, ninth instance of the dissociation.
- **Band widens** ⇒ still the most interesting outcome available, unchanged.

### 2026-08-23 19:22 — `dg_norm` is now a JOINT test of §4.7e, registered before it runs (arm 4/4, ETA ~20:30)

§4.7e landed *after* the 18:51 gate and makes the scale-invariant gate sharper than that gate alone.
**Depth keys span ~1.6 of 32 dimensions**, so even a *working* mixture has almost nothing to
discriminate between. The registered outcomes are therefore **two-dimensional**, not one:

| effective loops mixed | CE vs in-job control | reading |
|---|---|---|
| **< 1.5** | *(uninterpretable)* | selector again; scale was **not** the binding constraint. Second instrument failure, **no CE claim** (18:51 gate) |
| **≥ 1.5, near r** | **no gain** | **THE POINTED PREDICTION.** Mixing engages and buys nothing ⇒ **§4.7e's rank collapse is the binding constraint**, and this reduces to §4.7c's static-mixture null *with the mechanism identified*. Stronger than either result alone |
| **≥ 1.5, near r** | **gain > floor** | §4.7e is wrong or incomplete — rank ~1.6 was not the constraint, and the per-token headroom is reachable after all. **The most consequential outcome available tonight** |
| ≥ 1.5, **CE worse** | — | fourth "model takes the mechanism and gets worse" |

**Registered now because the joint reading is what makes it worth anything**, and adopting it after
seeing which cell fired would be precisely §6.0 rows 5 and 16.

**All six job IDs, one place (a compaction loses these):**
`tlab-duocausal-s0` **bt1qvi35v7gsejmvn1it** · `-s1` **bt1lkbri6cqj6q9fssoa** ·
`tlab-recmethod-s2` **bt1s4mag4kdvsvts536m** · `tlab-diversity-control-s0` **bt1ps6o54qhrecg40etf** ·
`tlab-pin2-control-s0` **bt1b76se42lip6987fb9** · `tlab-xsa-s0` **bt15egv862odi4o20qtn** ·
Kaggle **arsen4ikvar/tlab-lora-scaleup**. Harvest all six: `./harvest_duocausal.sh`.

---

## `tlab-untie-s0` — registered 2026-08-23 21:48 MSK, BEFORE the job was submitted

**The question.** §4.7e says the depth-mixing family fails because a token's depth keys span an
effective rank of ~1.6 of 32. §4.23's `dg_norm` supports it: a *working* soft mixture (7.58/8,
14.96/16, 29.84/32 effective loops) gains **−0.0012 / +0.0023** — nothing. **But that is a
correlation between low rank and no gain.** §4.28 shows rank is `≈1.6 × (number of distinct
projections)` and prices the fix. **This job is the causal test: raise the rank, keep the gate, see if
the gate acts.**

**Arms — three, in one job** (so §4.27's 0.0914 cross-job drift cannot touch the comparison):

| arm | config | params |
|---|---|---|
| `ut_ctrl_s0` | tied `W_K`, no gate | 9,064,608 |
| `ut_b4_s0` | `W_K` in **4** loop-index buckets, no gate | **9,967,776** |
| `ut_b4_gate_s0` | 4 buckets **+ scale-invariant depth gate** | 9,968,226 |

Seed 0, 3.5M tokens/arm, μ_rec 18 (`U[4,32]`), `supervise_k=5`, T4.

**Pre-launch gates, all verified before submission:**
- `kv_untie_buckets=1` is **bit-identical** to the unpatched frozen kernel: max\|diff\| = **0.000e+00**.
- Buckets 0 and 1 produce **different** keys (max\|diff\| 1.223e-01) — the mechanism is wired.
- Parameter counts measured by instantiating, not computed: all **under the 10M cap**.
- `outputs:` names every file explicitly, no globs (§6.0 row 34).

**FALSIFIERS, written before any data exists.**

**GATE A — did untying do what §4.28 predicts?** Measure the trained `ut_b4_gate_s0`'s depth-key
effective rank with `src/depth_key_rank.py`. **It must exceed ~4.** If it comes back near 1.6, the
buckets did not take and **nothing below is decided** — instrument failure, not a result.

**GATE B — the causal claim.** Let `Δgate_hi = CE(ut_b4_gate) − CE(ut_b4)` (the gate's contribution
*at high rank*) against the already-measured `Δgate_lo = −0.0012 / +0.0023` (its contribution at rank
1.6, §4.23).

| outcome | what it decides |
|---|---|
| GATE A passes **and** `Δgate_hi` is materially negative (beyond the 0.0150 floor) | **§4.7e is confirmed CAUSALLY.** The rank collapse was the binding constraint, and it is fixable |
| GATE A passes **and** `Δgate_hi` is null like `Δgate_lo` | **§4.7e is INCOMPLETE.** Rank is not what binds; the explanation the report leans on is at best partial |
| GATE A fails | nothing decided |

**GATE C — where does any gain sit?** If `ut_b4` or `ut_b4_gate` lowers CE, decompose
`Δgain = ΔCE@1 − ΔCE_best`. **On this report's own regularity (§4.24) I predict 78–101% at `r = 1`,
i.e. capacity rather than depth-mixing** — the same shape as LoRA, XSA and duo-causal. **If instead
the gain concentrates past `r = 8` and the useful band widens, that is the first genuine depth
mechanism in this project and the report's central negative is wrong.**

**Stated cost, so this is not read as free:** the untied arms spend **+10.0%** of the parameter budget
on `W_K` alone. Even a positive result here is *not* a recommendation for the submitted architecture —
it would be a mechanism finding, and §4.28 records that 8 buckets does not fit at this budget at all.

**Procedural amendment to `tlab-untie-s0`, written 21:52 — BEFORE any data, and it changes no
falsifier.** I registered GATE A as "measure the trained arm's depth-key rank with
`src/depth_key_rank.py`". **That job will return its weights WITHOUT its tokenizer**, like every
DataSphere job here (§4.27's quarantine note), so a local forward pass will feed it token ids that
mean something different from what it trained on. **This does not invalidate GATE A, and the reason
should be on record before the number exists:**

- GATE A is a **relative** comparison — `ut_b4_gate_s0`'s key rank against `ut_ctrl_s0`'s — and **both
  arms are evaluated on the identical local token stream**, so the mismatch is common-mode.
- Effective rank of the depth-key stream is a **structural** property: §4.7e measures it on
  **untrained** models, where no tokenizer relationship exists at all. It is not a capability readout.
- **GATES B and C are untouched**: they read the **in-job** `val_curve`, computed inside the job with
  that job's own tokenizer.

**What I should have done and did not:** added `tokenizer.json` to this job's `outputs:` before
submitting. It is one line, I knew about the failure ninety minutes earlier because I quarantined an
artifact produced by exactly it, and I still did not apply it to the job I launched afterwards.
**Recorded as a process failure rather than fixed retroactively**, because the job is already running
and editing the registration to hide the omission is the thing this file exists to prevent.

### `tlab-untie-s0` — GATE A pre-flight, run 22:22, BEFORE the job returned

**Why now rather than at harvest:** GATE A ("the trained arm's depth-key rank must exceed ~4, else
nothing is decided") is the gate that can waste the whole run. It is checkable **untrained**, on the
exact kernel that is executing, for zero GPU — so it was, rather than discovering at harvest that the
buckets never took.

| kernel config (untrained, `ds_untie/main.py`) | depth-key effective rank |
|---|---|
| `kv_untie_buckets=1` (tied — what the control is) | **2.729 / 32** |
| `kv_untie_buckets=4` (the arm) | **8.818 / 32** |

**Two things this establishes.**

1. **The instrument is the same one §4.7e used.** The tied figure **2.729** reproduces §4.7e's
   independently-measured untrained tied rank of **2.73** to three digits, on a different code path
   (the DataSphere kernel rather than `src/depth_key_rank.py`). *That is a cross-implementation
   agreement on a load-bearing number, which this project had not had for this quantity.*
2. **The mechanism is wired and diversifies at initialisation — 3.2×.** GATE A is therefore expected
   to pass, and the job should return an interpretable result rather than an instrument failure.

**One prediction attached, so this is falsifiable too:** §4.7e measures that **training makes the tied
rank worse** (2.73 → 1.83). If that also applies to the bucketed arm, its *trained* rank should land
**below 8.818 — plausibly 5–6 — while still clearing GATE A's threshold of 4.** *If the trained
bucketed rank comes back at or below ~4, GATE A fails and nothing about §4.7e is decided by this run.*

**What each landing outcome means, written now so it is not reasoned out under time pressure:**

| GATE A | `Δgate_hi = CE(ut_b4_gate) − CE(ut_b4)` | reading |
|---|---|---|
| passes | **materially negative** (beyond 0.0150) | **§4.7e confirmed causally.** The rank collapse *was* the binding constraint on the whole depth-mixing family, and it is fixable for +10.0% of the parameter budget |
| passes | **null**, like `dg_norm`'s −0.0012/+0.0023 at rank 1.6 | **§4.7e is INCOMPLETE.** Rank is not what binds; the explanation the report leans on is at best partial, and the honest move is to say so in §4.7e rather than defend it |
| fails | either | **nothing decided.** Instrument failure, exactly as in §4.22 |

**A second, independent question the same job answers, worth reading even if GATE B is null:**
`ut_b4` vs `ut_ctrl` is *untying's own CE effect*, with no gate involved — **does spending +10.0% of
the parameter budget on three extra `W_K` matrices buy anything at all?** On this report's own
regularity (§4.24) I expect any gain to sit mostly at `r = 1`, i.e. capacity rather than depth — and
**§4.28 already prices the alternative: at 8 buckets it does not fit under the 10M cap at all.**
