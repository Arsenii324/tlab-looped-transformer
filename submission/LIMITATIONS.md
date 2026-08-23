# What this project does not have

*`FAILURES.md` lists errors that reached a number. This lists **absences** — measurements never made,
comparisons never run, choices never screened. It is separate because "we got this wrong and caught
it" and "we never looked" are different kinds of weakness, and only the first is covered by a failure
log. Nothing here is hedging: each line is a concrete thing a reviewer could ask for and we could not
produce.*

---

## 1. The apparatus, stated plainly so the rest can be priced

**Every number in this submission was produced by code written by one coding agent, run on
infrastructure it configured, scored by an evaluation harness it wrote, and checked mostly by it.**
There is no independent implementation of anything. **Nothing outside this repository has ever scored
this model** — no public benchmark, no third-party harness, no other person's eval.

**The tokenizer is ours** (4,096-token byte-level BPE trained here), so perplexity is not comparable
to any published number, and bits/byte becomes comparable only through a `bytes/token = 3.3358`
constant that is also measured here.

**The headline is one 90M-token run.** *Almost every architectural claim is measured at 2.5–3.5M
tokens* — 3.6% of that budget — and §4.24 shows loop gain roughly **triples** between the two regimes.
**The interventions are measured where the loop is worth least.**

## 2. Evaluation

- **No downstream task of any kind.** No LAMBADA, no cloze, no probing suite, nothing. **Every claim
  here is next-token cross-entropy.** The phrase "the model is better" always means "CE went down".
- **No held-out test set** distinct from the validation shard used for every decision in the project.
  Split-half checks exist within it (§4.7); a genuinely untouched split does not.
- **No human evaluation.** Generation was sampled once as a sanity check (`RESULTS.md` §5.3) and is
  explicitly not a claim.
- **One validation shard, one position in one stream.** If it is unrepresentative every number moves
  together, and **none of our three consistency gates would notice** — they compare numbers to each
  other, not to ground truth.
- **Teacher-forced throughout, 256-token sequences.** §4.8 notes its cross-depth measurement "bounds
  from below rather than settles"; we have not checked how much more broadly that caveat applies.

## 3. Comparisons we did not run

- **No non-looped baseline at the headline budget.** The compute-matched baseline exists at 46M and
  took six-plus attempts to train stably (`FAILURES.md`).
- **No sparse / MoE arm.** `SCALE.md` §3 argues the parameter cap makes MoE the wrong comparison — but
  that is **arithmetic, not measurement**, and it is the alternative most likely to threaten
  "loop a dense block many times".
- **No width scan and no layers-per-loop scan.** One hidden size (448), one 3-layer block, one
  vocabulary. **§4.7e's mechanism is claimed to hold at any width and is measured at one width.**
- **No seed replication of the headline run.** The 90M control is n = 1.
- **The noise floor at full budget is unmeasured.** 0.0150 / 0.0541 come from 2.5M-token replicates
  and every "×the floor" statement in the report inherits them. §4.27 has already shown the *related*
  cross-job figure was understated **2.7×** when it was finally measured deliberately.
- **That 0.0914 cross-job spread is *unexplained*, which is the worse of the two possibilities.** It
  is **not** a tokenizer artifact — every DataSphere kernel's `train_tokenizer` is byte-identical
  (md5 `1dab774d…`), so the three controls share a vocabulary and were config-, seed- and
  step-identical. **An explained spread would be benign; an unexplained one at 6.1× the in-job floor
  bounds every cross-job statement in this project.** Candidates named in §4.27 — shard ordering,
  non-deterministic CUDA kernel selection, allocator differences — and none established.

## 4. Method

- **No hyperparameter search worth the name.** §6.0b lists the choices *inherited and never screened*:
  LR schedule, batch size, sequence length, RoPE theta, initialisation, and an `n_loop_eff = 24`
  constant that is wrong for every schedule actually run.
- **No pre-registration before the final day.** The falsifiers in `RUNS.md` are real and were written
  before their data existed — but they cover roughly one day's arms, not the project.
- **No power analysis.** Sample sizes were "what fit in the quota". Several claims rest on n = 1 or
  n = 2 and say so in place.
- **Instrument parameters were not swept until the last hours.** `plateau`'s tolerance is the clearest
  case: set once, never varied, and **tighter than the noise floor it exists to absorb**; sweeping it
  withdrew three claims (§4.25).

## 4b. The one thing we did test across budgets, and it failed

**This is not an absence — it is a measurement, and it belongs here because it prices everything
above.** The only intervention in this project tested at more than one budget is loop-cycled LoRA:
**−0.0936 at 2.5M tokens across five arms and three platforms, +0.0077 at 12M tokens in-job**
(§4.29). `[POSTHOC-LORA-RANK]` *(that −0.0936 already rested on a post-hoc rank restriction whose
all-arms interval covers zero, with no dose–response above the threshold — the scale failure is the
sixth deflation, not the first.)* Sign reversed, magnitude inside the replicate floor.

**So when this submission says an intervention "lowers the loss", that is a 2.5–3.5M-token statement,
and the one time it was checked at 5× it did not hold.** We have **no replicated CE improvement at
scale.** *The band results are the exception and are stated as such: supervision annealing's widening
reproduces with an identical edge decomposition at 2.5M and 10M, which is the only depth claim here
that has been tested across budgets and held.*

## 5. The two readings of our retraction record, and we cannot separate them

**Twelve claims were retracted in this project, three on the final day by their own pre-registered
falsifiers.** We present that as evidence of process.

**A reader is entitled to read it instead as evidence that the measurements are fragile**, and
**nothing in this repository distinguishes those two readings from the outside.** The honest position
is that both are true: the retraction rate is high *because the checking was aggressive* **and**
*because many effects here sit near the noise floor.* They are not separable with what we have.

## 6. The gates check consistency, not correctness

`src/headline.py`, `src/check_caveats.py` and `src/check_crossref.py` exist because self-review kept
failing, and they now pass `--strict`. **They verify that numbers agree with each other and that
deflated claims carry their caveats. A number that is wrong *everywhere* passes all three.**

---

## 7. What we would run next, costed from measured wall-clock

*Costed rather than listed, because "future work" is cheap to write and the numbers are available:
this project's arms run at **≈ 460 s per million tokens per arm on a T4** (measured — `divx` did 3
arms × 2.5M in 3,429 s; the Kaggle 12M arms took 5,648 s and 6,563 s).*

**The two short runs that would relate arms we already present** — both are the same shape: they
*connect* existing measurements rather than adding a new mechanism, which is where this project's
uncertainty actually sits.

| # | run | cost | what it converts |
|---|---|---|---|
| **1** | **Three config-identical controls in ONE job**, 2.5M each | **~58 min** | §4.27's **0.0914 cross-job spread is unexplained**, and it currently bounds every cross-job statement here. The *in-job* floor rests on **two accidental replicates found while auditing something else**. Three deliberate in-job replicates would separate "between-job drift" from "run-to-run noise" — and if the in-job spread is ~0.015 against 0.0914 between jobs, the report's in-job pairing discipline is **quantitatively vindicated** instead of merely asserted |
| **2** | **One DataSphere control that saves its `tokenizer.json`** | **~20 min** | Every DataSphere arm is currently in its own tokenizer family and its absolute CE is comparable to nothing outside that family (§4.27). **One anchored arm converts all of them**, retroactively, and it is a one-line change to `outputs:` — the same line whose absence cost an artifact tonight |

**The measurements that would move a conclusion, and why they were not run tonight:**

| # | run | cost | what it decides |
|---|---|---|---|
| 3 | **LoRA × annealing at 12M** | ~3.1 h | §4.29 withdrew LoRA at scale **for rank 4 under dense supervision**. The largest 2.5M positive combined it with annealing (−0.1172) and **that cell is unmeasured**. §4.17 found these were the only two interventions here that improved *both* endpoints |
| 4 | **A budget ladder** — one control at 2.5M / 5M / 10M in one job | ~2.2 h | Turns this project's "12× shrinkage" regularity from an **inference across jobs** into a **measurement within one**, which is exactly the assumption §4.24's scope condition rests on |
| 5 | **One same-config replicate at 90M** | ~9 h | The floor every "×the floor" claim in this report uses is measured at 2.5M and applied at 90M |
| 6 | **A second width** (e.g. 320 or 640) | ~1 h at 2.5M | §4.7e's mechanism is claimed width-independent and measured at **one width**. §4.28 makes it a dose–response prediction, so a second width is a real test rather than a replication |

**Running as this was written:** `tlab-untie-s0` — the causal test of §4.7e (tied control · `W_K` in 4
loop-index buckets · buckets + the scale-invariant gate), with GATES A/B/C registered in `RUNS.md`
before submission and `harvest_untie.sh` written before the data landed.

**Nothing else was started.** At the time of writing, a 12M scale test costs ~3.1 h against ~1.5 h
remaining, and this project's own §6.0 has two rows about launching under time pressure — including
one from tonight, where a job went out 24 minutes after we learned what its missing `outputs:` line
would cost. *Naming the runs and their prices is the honest deliverable here; starting one that
cannot land is not.*

## 8. What would change our mind fastest, if a reviewer has budget

1. **Any downstream task, however small** — the entire capability claim is currently "CE went down".
2. **Run #1 above (~58 min)** — it is the cheapest thing that would tighten a number bounding the
   whole report.
3. **An independent re-implementation of `src/eval.py`** — the single point of failure behind every
   number here.
4. **A second width** — see #6.
