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
