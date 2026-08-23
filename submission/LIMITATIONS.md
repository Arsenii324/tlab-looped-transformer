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
  Split-half checks exist within it (§4.7); an untouched split does not.
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
  vocabulary. **§4.7e's mechanism is measured flat across widths 224–896 at initialisation** (12× the parameters, spread 0.062, no trend — §4.31), **but whether the *trained* collapse is width-independent is not measured**: §4.30 measured training's effect at one width only.
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

## 6b. Numbers here that will **not** reproduce from the shipped artifacts, and why

*Read this before recomputing anything. Every row is a trap we walked into ourselves, and each
divergence is legitimate — but a reviewer who hits one without warning would reasonably conclude the
report is wrong.*

| if you recompute… | you will get… | why |
|---|---|---|
| **any DataSphere checkpoint, evaluated locally** | CE ≈ **9.3** — *above* chance (8.3178) | DataSphere kernels train their **own** BPE (NFKC normalizer, `unk_token`, 5,000 docs) and **return no `tokenizer.json`**; `src/train_tokenizer.py` uses no normalizer. **Never evaluate a DataSphere checkpoint against `configs/tokenizer.json`.** Kaggle's tokenizer **is** byte-identical to the shipped one, so Kaggle checkpoints can be. **And the DataSphere vocabulary itself now rebuilds locally** — its recipe is deterministic and byte-identical across all four frozen kernels — so use `configs/tokenizer_datasphere.json` (`src/rebuild_ds_tokenizer.py`), against which `rec_dense_s2` scores **CE 4.4252**, not 9.3 |
| **ρ from `checkpoints/jacobian_spec_results.json`** | **1.2273 / 1.0801**, not **1.6227** | The report's figure is the **90M control**; that JSON holds the **2.5M donor** checkpoints. All are > 1 — only the magnitude is checkpoint-specific |
| **log-drift vs power-law R² from `angular_convergence.json`** | **0.9885 / 0.8341**, not **0.986 / 0.748** | Same split: report quotes the 90M control, JSON holds 2.5M donors. Log-drift wins on every checkpoint measured |
| **the cv of oracle depth** | **0.95–1.18**, not 0.798 | 0.798 is the **angular budget at each token's oracle depth**, not the cv of the depth itself. The report mislabelled this until it was caught; the true figure is *larger*, so the argument strengthens |
| **oracle headroom** | **0.3084**, **0.3083**, or **0.2008 / 0.2032** | Three legitimate quantities — 46M test-split, 46M full-set, and the 2.5M annealed pair. Each is labelled where it is used |
| **the tail fraction on a 32-loop dump** | **~30.9%**, not 27.9% | 27.9% is *oracle depth > 32* on a **64**-loop sweep. On a 32-loop dump "past 32" is 0% by construction, and 30.9% is *fraction at the cap* — a different quantity |
| **any band, at a different `plateau` tolerance** | different edges for **65 of 135 arms** | `tol = 0.01` is **tighter than the 0.0150 replicate floor.** §4.25 sweeps it; four of eleven paired verdicts are tolerance-dependent and are marked as such |
| **absolute CE across two jobs** | up to **0.0914** apart at identical config, seed and step | Real, unexplained cross-job drift (§4.27). **In-job Δ always; absolute CE only within a tokenizer family** — {local, Kaggle} is one family, DataSphere is another |

### Which checkpoint carries which claim

The shipped artifact and the most-measured artifact are **not the same model**, which is stated in
place throughout but is easier to hold as a table:

| claim family | measured on |
|---|---|
| headline CE / ppl / bpb / band [6,17] | **90M control** (`full_control90_kaggle`) — the shipped artifact |
| trajectory geometry: drift law, readout gain, ρ, radial clamp | **46M** `full_no_state_renorm_kaggle` |
| per-token depth demand, the eight exit rules, ragged KV cache | **46M** `full_no_state_renorm_kaggle` |
| depth-key rank, and the tied-vs-untied contrast | **2.5M** screening checkpoints, plus untrained models |
| every paired intervention in §4.23 | **2.5–3.5M** in-job pairs |
| annealing's band result | **2.5M** (seeds 0–3) and **10M** (two arms: `as_10M_sw90`, `rec_sw90_s2`) — 6 of 6 |

## 7. What we would run next, costed from measured wall-clock

*Costed rather than listed, because "future work" is cheap to write and the numbers are available:
this project's arms run at **≈ 460 s per million tokens per arm on a T4** (measured — `divx` did 3
arms × 2.5M in 3,429 s; the Kaggle 12M arms took 5,648 s and 6,563 s).*

**The two short runs that would relate arms we already present** — both are the same shape: they
*connect* existing measurements rather than adding a new mechanism, which is where this project's
uncertainty actually sits. **One of the two turned out to need no run at all**, which is recorded
below rather than quietly deleted.

| # | run | cost | what it converts |
|---|---|---|---|
| **1** | **Three config-identical controls in ONE job**, 2.5M each | **~58 min** | §4.27's **0.0914 cross-job spread is unexplained**, and it currently bounds every cross-job statement here. The *in-job* floor rests on **two accidental replicates found while auditing something else**. Three deliberate in-job replicates would separate "between-job drift" from "run-to-run noise" — and if the in-job spread is ~0.015 against 0.0914 between jobs, the report's in-job pairing discipline is **quantitatively vindicated** instead of merely asserted |
| ~~2~~ | ~~**One DataSphere control that saves its `tokenizer.json`**~~ | ~~~20 min~~ | **SOLVED at 22:38 without a run, and this row is kept to show why.** The plan was to spend 20 minutes of GPU anchoring one DataSphere arm so the rest became interpretable. Unnecessary: the kernel's `train_tokenizer()` is **deterministic and byte-identical across all four frozen kernels**, so the vocabulary simply **rebuilds locally** (`src/rebuild_ds_tokenizer.py`). Verified against the thing it had to reproduce — `rec_dense_s2`, whose local evaluation had produced a quarantined CE-9.27 artifact, now scores **CE 4.4252** against chance 8.3178 and its own in-job 4.4907. **Every DataSphere checkpoint in this project is now locally re-evaluable, at zero compute cost.** *The one-line `outputs:` fix is still the right thing for any future job — it removes the need to reconstruct anything* |

**The measurements that would move a conclusion, and why they were not run tonight:**

| # | run | cost | what it decides |
|---|---|---|---|
| 3 | **LoRA × annealing at 12M** | ~3.1 h | §4.29 withdrew LoRA at scale **for rank 4 under dense supervision**. The largest 2.5M positive combined it with annealing (−0.1172) and **that cell is unmeasured**. §4.17 found these were the only two interventions here that improved *both* endpoints |
| 4 | **A budget ladder** — one control at 2.5M / 5M / 10M in one job | ~2.2 h | Turns this project's "12× shrinkage" regularity from an **inference across jobs** into a **measurement within one**, which is exactly the assumption §4.24's scope condition rests on |
| 5 | **One same-config replicate at 90M** | ~9 h | The floor every "×the floor" claim in this report uses is measured at 2.5M and applied at 90M |
| 6 | **A second width** (e.g. 320 or 640) | ~1 h at 2.5M | §4.7e's mechanism is claimed width-independent and measured at **one width**. §4.28 makes it a dose–response prediction, so a second width is a real test rather than a replication |

**`tlab-untie-s0` landed at 23:00 and its registered GATE A FAILED**, so **the causal test of §4.7e is
undecided and is reported that way** — the rank explanation still rests on `dg_norm`'s null, which is
a correlation. *This is a named absence, not a result we are holding back:* four distinct key
projections were predicted to give rank ~5.7 and gave **1.73** trained, so the high-rank arm the test
needed does not exist. **The failure was informative anyway** (§4.30: rank is 8.818/32 at
initialisation and 1.74 after training, which locates the collapse in the objective rather than only
in the architecture) — but that is a different claim from the one the job was built to decide, and it
has one seed.

**Nothing else was started.** At the time of writing, a 12M scale test costs ~3.1 h against ~1.5 h
remaining, and this project's own §6.0 has two rows about launching under time pressure — including
one from tonight, where a job went out 24 minutes after we learned what its missing `outputs:` line
would cost. *Naming the runs and their prices is the deliverable here; starting one that
cannot land is not.*

## 8. What would change our mind fastest, if a reviewer has budget

1. **Any downstream task, however small** — the entire capability claim is currently "CE went down".
2. **Run #1 above (~58 min)** — it is the cheapest thing that would tighten a number bounding the
   whole report.
3. **An independent re-implementation of `src/eval.py`** — the single point of failure behind every
   number here.
4. **A second width** — see #6.
