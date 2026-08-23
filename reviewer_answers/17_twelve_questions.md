# Reply — 2026-08-23 ~18:00 · your twelve questions, answered from artifacts

Every number below was re-derived just now, not recalled. Where a question exposed a live error, the
fix is already committed and named.

---

## Q1 — Token-keyed annealing: what is `n`? **n = 1. You caught a real error, and it was mine.**

One seed (0), one budget (10M), one schedule, in-job pair. And §3.5 **did** state it as a
recommendation: *"§3.5's recommendation should be read as token-keyed."*

**That is the identical failure this project withdrew six hours earlier**, made in the same document,
on a larger claimed effect, within two hours of writing the withdrawal. Knowing the lesson did not
prevent repeating it. What caught it was you asking "what is n?".

**Fixed and committed.** §3.5 now carries a `⚠ THIS IS n = 1, AND IT IS A LEAD, NOT A RECOMMENDATION`
block that (a) states the seed count, (b) names the repetition explicitly rather than quietly editing
it away, (c) puts the −0.2208 against this project's own measured seed spread on this exact class of
paired difference, **sd = 0.0640** — so the effect is 3.4× the spread, which is why it is a strong
lead and still one draw — and (d) says plainly that replication at ≥3 seeds is what would make it a
recommendation and **was not run**.

The surviving claim is narrower: the fraction rule was validated where it switches *before* loop gain
emerges and extrapolated to where it switches deep *post*-saturation; that gap is real and
mechanically motivated (§4.12); one 10M in-job comparison favours token-keying by 0.22.

## Q2 — The one-sentence answer to "what method did you build", verbatim from §3.5 today

> *"A Qwen3-style 3-layer block, weight-tied and applied `r` times, with **no prelude and no coda**,
> **no inter-loop normalisation**, additive re-injection of the embedded input at every loop, and
> `1/√(2·n_loop_eff)` output-projection scaling at init. Trained at a **deep loop schedule** (every
> step samples `r ~ U[32,48]`), with a **loss applied to a sparse subset of loops for most of training
> and to the final loop only for the last ~10% of steps** — *supervision annealing*."*

**And the honest caveat that now belongs beside it:** the CE half of annealing's justification is
withdrawn (n=4). What survives is that it relocates the useful band at 4/4 seeds. So the sentence
above is a method whose *depth* claim is replicated and whose *loss* claim is not resolved.

## Q3 — Has anyone read `report.md` end to end since the retractions? **No.**

That remains unknown-known #7 and it is now worse than when it was written, because two retractions
have landed since. I have done *targeted* propagation with mechanical verification (T10: checked all
8 remaining mentions of the withdrawn number; T15: §4.20's retraction propagated to the two sections
citing it as live). **That is not the same as an end-to-end read and I will not claim it is.** It is
the single largest outstanding risk on the artifact being graded.

## Q4 — Generated text: **run, and it passes.** Nobody had done this.

`full_control90_kaggle`, 8 loops, shipped tokenizer:

> *greedy:* "The history of the United States is a very small and very small and very small. The first
> thing that I've seen is that I'm not afraid of the fact that I'm afraid of the fact that…"
>
> *T=0.8:* "The history of the United States Medical Center contains intimidation for our collective
> tasks. The objectives for the federal government and their own population can help to provide those
> who need to make the best possible choice for safe use through…"

Recognisably English, grammatical, prompt-anchored, with the greedy repetition a 9M model at 90M
tokens should show. **No tokenizer/decoder defect.** Written into §6.0a — it is the cheapest
defect-detector in the project and it went unrun all session, which is itself the finding.

## Q5 — `agy` job B's ten findings: **sampled three, and one is real and non-trivial.**

- **Finding 3 — CONFIRMED, and it is a live confound.** In §4.6b's readout comparison, the `raw` and
  `final_only` arms train with gradients **pinned at the clip for 100% of logged steps** (raw norms
  26.10 and 85.07) while the `norm` control **never clips** (max 0.84). At `grad_clip = 1.0` those two
  arms are effectively trained with normalised gradient direction and fixed step size, and the control
  is not. The intervention *causes* the clipping, so it cannot be separated by re-running at the same
  clip. §4.6b now states that conclusions there are about "that readout **under saturating
  clipping**", not the readout alone.
- **Finding 10 — CONFIRMED but already known**: the 33-layer untied baseline diverged to NaN at step
  144 (296,960 tokens). Its own `eval_final.json` note says so; §4.4's live comparison uses the CUDA
  run, not this one.
- I did not verify the other eight and am not asserting them.

So: the round number was worth being suspicious of, and the set was still worth sampling.

## Q6 — Citation cross-check: **done mechanically, as the substitute for the job that never ran.**

28 arXiv-shaped strings in `report.md`; 2 are regex false positives. **15 verifiable from tarball on
disk. 11 not on disk — and all 11 are already flagged second-hand / relayed / unverified within ±6
lines of their citation**, checked mechanically rather than from memory. **No claim in the report
rests on an unflagged unverifiable source.** Recorded in `VERIFICATION.md`.

## Q7 — Push and upload: **neither has happened, and both are the user's call.**

- **No git remote is configured.** 0 uncommitted files, branch `review`, all work committed locally.
- **No HF upload has ever run.** `upload_checkpoint.py` was fixed today (it previously shipped weights
  *without* `configs/tokenizer.json`, which would have made every downloaded checkpoint evaluate at
  chance ≈ 8.32) and dry-run verified, but never executed against the network.
- The fresh-clone dry run passed at 11:00; the tree has changed substantially since (new `src/`
  probes, `model.py` gains `cond_mode`/`depth_gate_mode`), so **it should be re-run before shipping**.

## Q8 — Known unknown: the number I'd least like a grader to check

**The 0.398 nats/e-fold token-scaling constant.** It is used to correct token-mismatched arms (§4.1's
seed-spread correction) and to argue what the unspent 8% of budget is worth, and it is fitted from
this project's *own* small number of budget points. If it is off by 30%, several corrected
comparisons shift. It is load-bearing, it is used across sections, and it has never been given its own
error bar.

## Q9 — Unknown known: something true, in the artifacts, never written down

**The eval loop sweep is not the same grid across the arms this report compares.** §4.15 and
`plateau.py` both warn that plateau midpoints are grid-conditional (17% swing on the same checkpoint),
and the headline table now names its grid. But the *screening* arms use
`(1,2,4,8,12,16,24,32)` while today's DataSphere arms use `(1,2,4,8,12,16,20,24,32,48,64)` — an
11-point grid with two extra shallow points. Every within-experiment comparison is safe (arms share
their grid). What has never been written down is that **`plateau_mid` values are not comparable
between the screening series and the 2026-08-23 series**, and both appear in this report. I have not
audited whether any cross-series midpoint comparison actually occurs — that is now the top item on
the stack.

## Q10 — Unknown unknown: the most likely reason this is judged poorly, excluding §6

**That the report reads as a methodology audit rather than as an answer to the question asked.** The
task is: get low perplexity *by exploiting many loops*. The document's centre of gravity has moved to
instruments, retractions, floors and nulls — much of which is genuinely good work — and its single
most-defensible claim is now a **negative** (saturation without convergence; seven interventions,
zero ceiling improvements). A grader looking for "here is the method, here is the number" finds a
withdrawal, a lead at n=1, and §1 empty. **The work is honest and the framing is not obviously
answering the brief.** That is not in §6 because §6 catalogues errors, and this is not an error.

## Q11 — The plan, 18:00 → 23:30, in hours

| window | task |
|---|---|
| **18:00–18:45** | Harvest `tlab-deep-full` (lands ~now), read as §4.16b band replication, write up. Harvest `od_depth_gate` (arm 4/4) with the caveats pre-registered in `RUNS.md` 17:40 |
| **18:45–20:30** | **End-to-end read of `report.md`** — Q3, the largest outstanding risk. Also resolves Q9's grid audit |
| **20:30–21:15** | Re-run the fresh-clone dry run; execute `upload_checkpoint.py` **if the user authorizes**; push **if a remote is configured and authorized** |
| **21:15–22:30** | §8 writing items still owed (W1/W2/W4/W5); tighten §3.5's opening sentence |
| **22:30** | **HARD STOP on editing.** Whatever exists ships |

**§1 is the user's and is still empty.** If it is empty at 20:00 that is recoverable; at 22:00 it is
not, and criterion 1 has no other carrier now that the method's headline number is withdrawn.

## Q12 — Think-at-Hard and 2608.09444

**2511.08577 re-verified from source**, which matters because §6.0 row 22's retraction was about this
exact paper: `3_method.tex:206` reads *"over 73\% of next-tokens are correctly predicted at the first
iteration"* — the reviewer's figure, not the 85% a summarising fetch produced. The report's other
quote from it is verbatim at `3_method.tex:207`.

**2608.09444 is not cited anywhere in `report.md`**, so there is nothing to log. Your proposed
reconciliation is right and I'd have used it: a ragged cache is safe here *because* the per-depth
states are nearly identical in the directions attention reads — the same dilution (§4.3) that kills
depth utility. That **unifies §4.3 and §4.8 rather than threatening either**, and I've noted it in
`VERIFICATION.md` against the day someone does cite it.

---

## Surfaced while answering, not asked about

- **`od_lora_r2` landed**: +0.0941 CE, band unmoved. Empirically confirms §4.20's retraction — the arm
  was built to fix what turned out to be a shared-residual artifact.
- **§6.0b now states the `n_loop_eff = 24` limitation**: the init constant is fixed at 24 while
  schedules ran at mean 18 and 40 (ratios 1.15×, 0.77×). In-job pairs are unaffected (both arms share
  the same wrong constant); cross-schedule comparisons carry it.
- **`od_depth_gate`'s read is pre-registered with two caveats** that change its interpretation: it
  zero-inits to a *uniform mixture*, not to the control, so "beats control" is confounded with the
  uniform-start offset; and it mixes over loops `1..r` at eval, so **its plateau is over
  mixture-window size, not depth**, and must not enter the band tables.
