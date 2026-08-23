# Reply — 2026-08-23 11:00 MSK (D1–D3, S1–S3, T1)

**D1 was live, and asking it caught two real defects.** Answering in your priority order.

---

## D1. Tokenizer shipping — **the trap was live; now closed and verified**

**(a) Was the exact vocab serialised anywhere?** Yes for the *local* one (`configs/tokenizer.json`,
in-repo since 13 Aug). **No for the Kaggle runs** — you were right. `kaggle/main.py` trains the BPE
fresh from a FineWeb stream, and its only outputs were `results.json` and the checkpoint. **It never
saved the tokenizer.** The vocabulary behind the headline weights existed solely as a side-effect of
a run, and identity with the local vocab had been *inferred from the eval looking coherent* — which,
as you said, is not a check, because a mismatch reports CE ≈ ln(4096) = 8.32 rather than raising.

**Now verified rather than inferred.** `src/check_tokenizer_identity.py` scores a checkpoint on local
data with the local vocab and compares against the number the *producing* run reported:

| checkpoint | local CE@1 (32 batches) | producing run | \|diff\| | gate |
|---|---|---|---|---|
| 90M control | 3.9642 ± 0.0593 | 3.9192 | 0.0450 | **PASS** |
| 46M headline | 4.2148 ± 0.0562 | 4.2580 | 0.0432 | **PASS** |

Both are ~1.4 nats from chance on the decisive comparison. **The vocabularies are the same object**,
so no published number changes. Fixed structurally anyway: `kaggle/main.py` now saves
`tokenizer.json` beside its checkpoint, because "it happened to match" is not a property to rely on twice.

**(b) Will it ship with the weights?** Yes — `configs/tokenizer.json` is in the submission tree, and
the model card will name the identity gate and the expected CE@1 so a grader can run it in one command.

**(c) Does `eval.py` rebuild one?** No — it reads a pre-tokenized `uint16` memmap, so it never touches
a tokenizer. `data.py` **loads** `configs/tokenizer.json` (it does not retrain).

**But your question found a second, worse defect that (c) alone would have hidden.** The README
quickstart read:

```
python src/train_tokenizer.py     # <-- OVERWRITES configs/tokenizer.json
python src/data.py
```

**A grader following the README would retrain the vocabulary, overwrite the shipped file, and then
evaluate every released checkpoint at chance** — silently, looking like a broken model rather than a
broken setup. That is your named failure, verbatim, sitting in my own quickstart. The README now
leads with a warning, separates the "evaluate a released checkpoint" path from the "reproduce from
scratch" path, and puts the identity gate in the normal path. **This is the single most valuable
thing to come out of the reviewer thread.**

---

## D2. Fresh-clone dry run — **done, and it would have failed for a different reason entirely**

Not "the scripts exist" — actually run from clean. Two findings:

**The push would have been rejected outright.** The working repo has a **2.0 GB `.git`**, and three
tracked `.npz` files of **564 MB each**. GitHub hard-rejects any file over 100 MB. A `git push` would
simply have failed, and discovering that at 22:00 would have been very bad.

**Resolved without rewriting history** (the user's standing constraint is that previous work stays
intact): a clean submission tree — **193 files, 2.9 MB** — built from the tracked set minus binaries,
with a `.gitignore` that keeps them out. Dry run from a fresh clone:

```
fresh clone OK
configs/tokenizer.json present — sha16 e32e4cd74ca3935b (identical to the working repo's)
python src/test_model.py    -> ALL CHECKS PASSED   (9 gates)
python src/test_plateau.py  -> ALL PASS            (8 gates)
python src/eval.py --help   -> ok
```

Not yet exercised end-to-end: `data.py` needs to stream FineWeb (network, ~90s) to regenerate
`val.bin`, which is deliberately not in the repo. That path is code-verified (it loads the shipped
tokenizer) but not run from cold; it is on the list before any push.

---

## D3. Which checkpoint is the deliverable — **not decided, and deliberately not yet**

Three candidates, and the honest answer is that two numbers land first:

| candidate | val ppl | status |
|---|---|---|
| 90M **norm-penalty** | **36.03** | best perplexity measured |
| 90M **control** (the plain method) | 37.14 | the config §3.5 describes, minus annealing |
| `tlab-deep-full` (μ_rec=40 annealed, ~32.6M tok) | pending, ETA 17:30 | the *method's* demonstration; expected ~+0.5 nats worse by construction |

Both 90M figures come from the Kaggle kernel's own eval; `run_eval90.sh` re-scores them under the
local protocol when the GPU frees, and only then does `headline.py set` swap anything.

**Your point about the mismatch is taken and will be handled explicitly.** If the uploaded weights are
the 90M control while the report proposes annealing, the model card and §3.5 will both say so in
plain words — *"these weights demonstrate the architecture; the annealing schedule that §4.17
recommends was validated at 2.5–32M tokens and is not in this checkpoint"* — rather than leaving a
grader to notice.

---

## S1. Yes — the head is PALBERT's own best configuration, not Ouro's

Built to their spec, with their sentence in the docstring: `[h_t, h_{t−1}]` concatenation
(`src/qexit.py`, the feature build) **and** a tanh hidden layer, not a single linear on a single
state. So §4.7's negative rules out **PALBERT's best row**, which is the stronger claim you wanted.

## S2. No longer n=1 — and you identified the exact missing arm

`tlab-deep-anneal2` (launched 10:17, before your message) re-runs the **annealed** arms and the dense
control at seed 1. You spotted what it lacked: **no constant-terminal arm**, so the load-bearing
45.3-vs-39.2 gap still could not be replicated. `tlab-term-seed1` (launched 10:40) is that arm alone,
with the falsifier written down. On your throughput worry — DataSphere gives each job its own node, so
this costs `tlab-deep-full` nothing; it is running at 1333 tok/s and now tracking **~32.6M tokens**.

## S3. §3.5 exists — written, and marked provisional

*"The final method, named — and why this form."* One paragraph of method, a table of the four
load-bearing choices each with the measurement that decided it, the three convergent nulls, the scale
argument aimed directly at the task's value-embedding counter-example (the mechanism adds **zero
parameters**, so there is no table to outgrow), and both settings of the architecture with the tension
stated. Carries a **PROVISIONAL** banner naming the two in-flight runs that can falsify it.

## T1. Cutoff and writing state

- **Compute cutoff 18:00**; `tlab-deep-full` harvests ~17:30 and is the last number that can enter.
- **Last number for a *load-bearing* claim: ~12:45** (`tlab-anneal-scale`, the budget test that can
  falsify §3.5). Everything after that is replication.
- **Writing state: sections, not outline.** ~3,300 lines, zero placeholders, 21 sections in §4–§6
  including three written today (§4.15 statistics, §4.16/§4.16b supervision, §4.17 annealing) plus
  §3.5 and §6.0. §1 is untouched and stays that way.
