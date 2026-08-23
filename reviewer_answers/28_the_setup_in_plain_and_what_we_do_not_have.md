# Reply — 2026-08-23 22:10 · the setup in plain, and everything we do not have

*Written to be anchored on. Previous replies argue for results; this one describes the apparatus that
produced them and lists what is missing, so a reader can price the whole thing rather than audit it
claim by claim. Structured as knowns/unknowns because the useful part is the bottom two sections.*

---

## 1. The setup, stated plainly

**One coding agent and one author.** Every number in `report.md` was produced by code the agent wrote,
run on infrastructure the agent configured, scored by an evaluation harness the agent wrote, and
checked — for the great majority of claims — by the agent. **There is no independent implementation of
anything.** Three external adversarial passes were run today; their hit rate on flagged items was
~20–60% and every item had to be re-derived by hand (`FAILURES.md`).

**Nothing outside this repository has ever scored this model.** No public benchmark, no third-party
eval, no leaderboard, no other person's harness. The number a reader is asked to trust —
**CE 3.6599 / ppl 38.86 / bpb 1.5829** — comes from `src/eval.py` on a validation shard packed by
`src/data.py` from the same stream the model trained on.

**The tokenizer is ours**: a 4,096-token byte-level BPE trained by this project. **Perplexity is
therefore not comparable to anyone else's number**, and bits/byte only becomes comparable through a
`bytes/token = 3.3358` constant that is itself measured here.

**Compute, and its shape matters more than its total.** Three platforms: Yandex DataSphere (T4),
Kaggle (T4), local Apple MPS. **The headline is one 90M-token Kaggle run.** *Almost every
architectural claim is measured at 2.5–3.5M tokens* — 3.6% of the headline budget — and §4.24 shows
loop gain roughly **triples** between those regimes, so the interventions are all measured where the
loop is worth least.

**The evaluation protocol is narrow.** Teacher-forced next-token cross-entropy, 256-token sequences,
one validation shard, no held-out test set distinct from that shard, `n_loops` swept on a fixed grid.

## 2. Known knowns — what we claim, in one place

- **Saturation without convergence.** The state never converges (log-drift R² 0.986 vs a power law's
  0.748; ρ = 1.62 at loop 2) and CE stops improving at loop ~8 anyway. Contradicts the task's own
  premise. *Multiple instruments, an untrained control.* **This is the spine.**
- **Twelve interventions; five lower the loss; none widens the useful band.** Four of the five put
  **78–101%** of their gain at a *single* loop where their own mechanism is inert.
- **Depth keys span ~1.6 of 32**, at initialisation and worse after training; the cause is weight
  tying via the **projection** asymmetry (§4.7e, corrected). **The one experiment built to overturn it
  ran and did not** (`dg_norm`, falsifier registered first, two seeds).
- **Supervision annealing widens the useful band at 5/5 seeds and does not lower the loss.**
- **Early exit does not pay here** — eight rules, five classes, best captures **0.1%** of a real
  0.3084-nat headroom (`EARLY_EXIT.md`).

## 3. Known unknowns — things we know we have not settled

| | |
|---|---|
| **Is the "78–101% at r=1" pattern budget-invariant?** | Every arm showing it is at 2.5–3.5M. **The single probe is still running.** If the pattern is a screening artifact, the report's central sentence is about small models, not looped transformers |
| **The noise floor at full budget** | 0.0150 / 0.0541 are measured at 2.5M and applied to 90M claims throughout. **No same-config replicate at 90M has ever been run** (~9 h T4). §4.27 has already shown the *related* cross-job figure was understated 2.7× once someone measured it deliberately |
| **The LoRA positive** | Post-hoc rank restriction; all-arms interval covers zero; no dose–response; 67–95% at `r = 1`; a **zero-diversity pinned branch beats it** at seed 1; 4.8× seed spread. **I would not defend this as a positive** |
| **Whether depth-key rank is *causally* what binds** | `dg_norm` gives a correlation. `tlab-untie-s0` is running to test it and its falsifiers are registered |
| **`sw75` vs `sw90`** | The annealing family splits and we never explained why 25% fails where 10% works |

## 4. Unknown knowns — obvious in hindsight, never written down, all found in the last six hours

**This is the most useful section for judging the work, because each of these was invisible until
something forced it, and each had been silently load-bearing for weeks.**

1. **`plateau(curve, tol)` has a free parameter and it was never swept.** `tol = 0.01` was set once.
   It is **tighter than the measured 0.0150 replicate floor** — the tolerance is smaller than the noise
   it exists to absorb — and **65 of 135 arms change a band edge** at the floor. Sweeping it withdrew
   three narrowing claims made the same evening (§4.25). *This is `argmin`'s retirement repeating
   inside its own replacement.*
2. **The block has three layers, each with its own `W_K`.** A cost I priced at +100,352 parameters is
   **3× that**. Found by building the arm, not by re-reading the table (§4.28).
3. **Every DataSphere job trains its own BPE and returns weights without it.** Those checkpoints
   **cannot be evaluated locally at all**; doing so reports CE ≈ 9.3 against a chance level of 8.3178.
   One artifact was produced this way tonight and quarantined (§6.0 row 35).
4. **"In-job paired" is not batch-identical for annealing arms.** A zero-size RNG draw at `k = 1`
   consumes no state, so an annealed arm's batch stream diverges from its control's after the switch
   (§4.26). Not a bias; added variance, on the one comparison withdrawn for variance.
5. **Absolute CE is comparable only within a tokenizer family.** {local, Kaggle} share a
   byte-identical vocabulary; DataSphere has its own. No correction factor bridges them.
6. **A guard inside one function is not a guard on the quantity.** `eval.py` has checked CE against
   chance for weeks and did not stop the failure it was written for, because a one-off script never
   called it.

## 5. What we do not have — the plain list

**Evaluation**
- **No downstream task of any kind.** No LAMBADA, no GSM8K, no cloze, no probing suite. Everything is
  next-token CE on one validation shard.
- **No held-out test set** distinct from the validation shard used for every decision in the project.
- **No human evaluation.** Generation was sampled once, tonight, as a sanity check (`RESULTS.md` §5.3)
  and is *not* a claim.
- **No external baseline.** Nothing published was reproduced end-to-end in this harness.

**Comparisons**
- **No non-looped baseline at the headline budget.** The compute-matched baseline exists at 46M and
  took six-plus attempts to train stably (`FAILURES.md`).
- **No sparse / MoE arm**, so "loop a dense block" is never compared to the alternative that would
  actually threaten it (`SCALE.md` §3 makes the argument on arithmetic, not measurement).
- **No width or depth scan.** One hidden size (448), one layers-per-loop (3), one vocabulary (4,096).
  **§4.7e's mechanism is claimed to be width-independent and is measured at one width.**
- **No seed replication on the headline run.** The 90M control is n = 1.

**Method**
- **No hyperparameter search worth the name.** §6.0b lists the choices that were *inherited and never
  screened* — LR schedule, batch size, sequence length, RoPE theta, init scheme, the `n_loop_eff = 24`
  constant that is wrong for every schedule actually run.
- **No pre-registration before today.** The falsifiers in `RUNS.md` are real and were written before
  their data existed, but they cover roughly the final day's arms, not the project.
- **No power analysis.** Sample sizes were "what fit in the quota".

**Process**
- **The wandb API key has not been rotated** and the repositories go public. Scrubbed from history and
  secret-scanned; that does not un-send it. It is the author's action and is outstanding.
- **`report.md` has not had a full end-to-end read since ~18:45**, and ~1,400 lines landed after that.
  Targeted reads of the new sections were done and found three defects, all fixed.

## 6. Unknown unknowns — the shape of what we cannot see

Named as shapes rather than contents, since by definition the contents are not available:

- **Everything depends on one validation shard from one position in one stream.** If that shard is
  unrepresentative, every number moves together and no internal check would notice — all our
  consistency gates compare numbers to *each other*.
- **We have never tested whether the eval protocol hides an effect.** 256-token sequences, teacher
  forcing, and a fixed loop grid could each mask a depth effect that appears in generation or at
  longer context. §4.8's scope note says the cross-depth measurement "bounds from below rather than
  settles"; the same caveat plausibly applies more broadly than we have checked.
- **Twelve retracted claims is either strong evidence of process or weak evidence of measurement.**
  We present it as the first. **A reader is entitled to read it as the second**, and nothing in the
  repository distinguishes those two readings from the outside. *The honest position is that the
  retraction rate is high because the checking was aggressive **and** because many effects here are
  near the noise floor — both are true and they are not separable with what we have.*
- **The agent wrote the analysis, the instruments, and the critique of both.** Three mechanical gates
  now exist (`headline.py`, `check_caveats.py`, `check_crossref.py`) precisely because self-review
  kept failing — but **they check consistency, not correctness.** A number that is wrong everywhere
  passes all three.

---

## 7. What would change our mind fastest, if a reviewer has budget

1. **One same-config replicate at 90M** — settles the floor that every "×the floor" statement uses.
2. **Any downstream task**, however small — the entire capability claim is currently "CE went down".
3. **A dense integer eval grid on the annealing pair** — running now, locally, on the Kaggle
   checkpoints whose tokenizer we verified byte-identical to the shipped one.
4. **A second width** — it would test §4.7e's width-independence, which is asserted from a mechanism
   and measured at one width.
