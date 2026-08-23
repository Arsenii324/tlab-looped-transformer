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

## 7. What would change our mind fastest

1. **One same-config replicate at 90M** — settles the floor every "×the floor" claim uses.
2. **Any downstream task, however small** — the capability claim is currently "CE went down".
3. **A second width** — tests §4.7e's width-independence, which is asserted from a mechanism and
   measured at one width.
4. **An independent re-implementation of `src/eval.py`** — the single point of failure behind every
   number here.
