# Where the submission actually stands — my read, not the jury's

*Written 2026-08-23 ~21:20 MSK by the coding agent, for the author, ahead of a 23:59 deadline. This
is **not** a submission document and should not go in `submission/`. It is my honest assessment of
what is solid, what I would attack, and what nobody has looked at. Where I am uncertain I say so
rather than rounding to confident.*

---

## 1. State by component

| component | state | my confidence | note |
|---|---|---|---|
| **The artifact** (90M control, 9,064,608 params) | shipped, on HF, gate passes against the **downloaded** copy | **high** | Both caps verified from the weights, not a config. Model generates fluent English (checked tonight, first time ever) |
| `submission/METHOD.md` | complete | **high** | Recommendation now qualified *at the recipe*, not three paragraphs later |
| `submission/RESULTS.md` | complete | **medium-high** | Count convention fixed; the band-direction claims were downgraded tonight (§4.25) |
| `submission/EXPERIMENTS.md` | generated from artifacts | **high** | 135 arms; the coverage block is now generated too, so it cannot drift |
| `submission/SCALE.md` | complete | **medium** | §5's mechanism is good; §2's weak joint is real and stated |
| `submission/NEGATIVE_RESULTS.md` | complete | **high** | This is the strongest document in the folder |
| `submission/FAILURES.md` | complete | **high** | Three named patterns, the third added tonight |
| `submission/EARLY_EXIT.md` | **new tonight** | **medium-high** | Closes the brief's optional clause, which had no owner. Written fast; would benefit from one more read |
| `report.md` §0–§8 | ~6,900 lines | **medium** | §4.23–§4.26 are ~4 hours old. §4 as a whole has never been fully re-read since ~18:45 |
| The three gates | green | **high** | `headline.py`, `check_caveats.py`, `check_crossref.py` all pass `--strict` |

## 2. What I actually believe, as distinct from what is written

**I would bet on these:**
- **Saturation without convergence.** Multiple independent instruments, an untrained control, and it
  contradicts the task's own premise. This is the spine and I think it holds.
- **The depth-key rank collapse (~1.6 of 32) is real and is caused by weight tying.** The
  projection-asymmetry argument is structural, provable at initialisation, and survived the one
  experiment built to kill it (`dg_norm`, two seeds, with the falsifier registered first).
- **Supervision annealing widens the useful band and does not lower the loss.** 5/5 seeds, identical
  edge decomposition at 2.5M and 10M, and — as of tonight — robust to halving the plateau tolerance.
  This is the best-supported positive depth claim in the project.
- **Early exit does not pay here, and the reason is measured.** Eight rules, five classes, 0.1%.

**I would not bet on these, and they are in the submission:**
- **The LoRA positive.** Post hoc rank restriction, all-arms interval covers zero, no dose–response,
  67–95% of the gain at `r=1`, a zero-diversity pin *beats* it at seed 1, and the cycled arm ranges
  −0.0261…−0.1251 across two in-job pairs at one budget. **I think this is close to nothing and the
  submission should probably say so more bluntly than it does.**
- **XSA's −0.24.** It replicates and it is large, but 84–91% is at `r=1`, it is a *generic* attention
  operator with no loop-dependence, and its band claim already died at the second seed. It belongs in
  the report; it is not evidence about looping.
- **Anything about band *direction* for the loss-lowering arms.** Downgraded tonight and rightly.

**The thing I am least sure about and cannot resolve:** whether the "78–101% at r=1" pattern is
budget-invariant. Every arm showing it sits at 2.5–3.5M, and loop gain triples by 90M (§4.24). If the
pattern is a screening-scale artifact, the report's central sentence is a statement about small
models rather than about looped transformers. **The Kaggle 12M arm is the only probe and it can only
show "shifted a lot" or "did not".**

## 3. Concerns, ranked by what they would cost if a grader hit them

1. **~~The parameter cap looks violated~~ — FIXED tonight.** `state_dict` sums to 10,899,616 against a
   10M cap; the real count is 9,064,608 (tied embedding counted twice). It was explained only in
   `report.md` §6.0 row 27 — **nowhere a grader looks.** Now in `submission/README.md`, `METHOD.md`
   and the HF card. *This was the single most costly possible misreading and it was live until 21:05.*
2. **`report.md` has not been read end-to-end since ~18:45**, and ~1,000 lines have landed since. The
   two full reads this project has done found 12 defects each, none grep-reachable. **A targeted read
   of §4.23–§4.26 was done tonight and found 3 defects in my own text; §4.1–§4.22 has not been
   re-read.**
3. **The submission is written in a confident register about results that are one or two seeds at one
   budget.** Every scope condition is stated somewhere, but the *prose* reads stronger than the
   evidence in places. A hostile reader could fairly call several sections over-claimed on tone.
4. **The LoRA row is the weakest thing presented as a positive** (see §2). If a grader attacks one
   number, it is this one, and the honest answer is "you are right".
5. **Nothing verifies `report.md`'s numbers against the artifacts** beyond the headline. `headline.py`
   covers the headline; `check_crossref.py` covers submission→report; **report→JSON is unchecked for
   several hundred figures.** I spot-checked §4.23's 36 CE figures tonight and found 2 wrong (derived
   from rounded deltas rather than read). That rate over §4 as a whole would be dozens.
6. **The wandb key is unrotated** and both repos go public. The author has said they will handle it.

## 4. Unknown unknowns — five checked tonight, and what is still dark

**Checked, and they paid:**

| # | the thing nobody knew was a problem | outcome |
|---|---|---|
| 1 | **Does a `state_dict` sum look like a cap violation?** | **YES — 10,899,616.** Fixed on three surfaces |
| 2 | **Has anyone ever looked at what the model writes?** | **No, in 6,900 lines.** Sampled it: fluent grammatical English, semantically incoherent, no degeneracy. **The model works** |
| 3 | **Was the plateau tolerance ever varied?** | **No — set to 0.01 once.** Sweeping it downgraded the narrowing claims and *upgraded* annealing's widening (§4.25) |
| 4 | **Do "in-job paired" arms really see the same batches?** | **Yes, except annealing pairs** — `k=1` short-circuits the shared rng. Not a bias; a *mechanism* for the variance that got annealing's CE claim withdrawn (§4.26) |
| 5 | **Is `bytes/token = 3.3358` measured or asserted?** | **Measured, and I reproduced it independently** over 2M val tokens: 3.3358, bits/byte 1.5828 vs published 1.5829. The only externally-comparable number holds |

**Still dark — I would look here next:**

- **`report.md` §4.1–§4.22 has never been checked figure-by-figure against the stored JSON.** §4.23's
  spot check found a 2/36 error rate. **This is my top unchecked risk.**
- **The eval grid is `{1,2,4,8,12,16,20,24,32,48,64}` — geometric and coarse past 24.** Every "end"
  band edge past 24 is resolved by *one* grid point. `end 16 → 24` is a single-interval move, and the
  annealing claim rests on it at 5 seeds. **A denser grid between 16 and 32 would either confirm or
  dissolve the strongest depth claim in the project, and it has never been run.**
- **`BYTES_PER_TOKEN` is duplicated as a literal in four files.** All four currently read 3.3358, so
  no live defect — but this is the one-implementation-per-quantity rule violated on the number the
  submission calls "the only externally comparable figure".
- **Every geometry claim is on ONE checkpoint** (the 46M `no_state_renorm`). The rank collapse, the
  drift law, the readout gain. `dg_norm` tests the *consequence* at two seeds, but the underlying
  measurement is n=1 in checkpoints.
- **No arm was ever trained twice with the same config and different seeds *at the full budget*.**
  The floors (0.0150 / 0.0541) come from 2.5M-token replicates. Whether they transfer to 90M is
  assumed, not measured — and every "×the floor" statement inherits that.

## 5. Ablations I would run, in the order I would run them

None of these are launched. Costs are T4-hours.

| # | ablation | cost | what it decides | why it is not running |
|---|---|---|---|---|
| **A1** | **Dense eval grid 16–32 (every integer) on the annealing pair at 10M** | **~0.3 h**, no training — re-eval of `rec_dense_s2` / `rec_sw90_s2` | Whether `end 16 → 24` is a real edge or one grid interval. **This is the cheapest check on the project's strongest depth claim and I would run it first** | Time; needs only the returned `.pt` files, which we have |
| **A2** | Plateau tolerance sweep over **all 135 arms**, not the 7 pairs | ~0 h, pure computation | Whether §4.25's conclusion generalises past the pairs I checked | Partially done tonight |
| **A3** | **One unshared `W_K` per loop-index bucket** (partial untying), 2 seeds × 2.5M | ~2 h | The single mechanism §4.7e/`SCALE.md` §5 points at: if depth keys collapse because one `W_K` sees a collinear stream, giving 4 buckets their own `W_K` should raise the rank. **Directly tests the report's central mechanism rather than another consequence of it** | Not implemented; ~30 lines in `model.py` |
| **A4** | Same-config replicate at **90M** (2 seeds) | ~9 h each | Whether the 0.0150 floor transfers to full budget. Every "×the floor" claim depends on it | Cost; would consume the remaining quota |
| **A5** | The r=1 share at **12M and 30M** for one mechanism (XSA is cheapest — zero params) | ~3 h | §4.24's open question, with more leverage than the single Kaggle LoRA arm | Time |
| **A6** | Generation-based eval (not teacher-forced) on the shipped checkpoint | ~0.5 h | Everything in the report is teacher-forced. §4.8's ragged-cache result explicitly bounds a *generating* exiter from below only | Time; sampling tonight was qualitative only |
| **A7** | `report.md` §4 figure-by-figure re-derivation from stored JSON | ~1 h scripted | Concern #5. Would either clear §4 or find a defect class | Time |

**If I had exactly one more hour of compute: A1.** It is cheap, needs no training, and it is the only
one that can strengthen *or* dissolve the claim the submission leads with on depth.

**If I had one more hour of *my* time and no compute: A7**, then a full read of §4.1–§4.22.

## 6. What is running, and what I am waiting on

- **Kaggle `tlab-lora-scaleup`**, 12M/arm, ETA ~21:52. The only budget probe of §4.24. **Whichever way
  it goes it is one arm, one seed, one mechanism** — it cannot settle the scope condition, only
  indicate.
- Nothing else. Four DataSphere jobs landed tonight and are written up (§4.23–§4.23e).

## 7. My overall read

**The submission is in good shape and its strength is the thing the brief actually rewards.** It has a
spine that survives every withdrawal; a measured mechanism under the central negative; a falsifier
that was registered before its data existed, ran, and held; and a failures document that is evidence
rather than apology. **Twelve claims have been retracted and every one is visible with its
correction** — including three tonight, two of which were mine and one of which was a retraction of my
own earlier retraction.

**What would make me nervous in front of a grader** is not the science but the **consistency
surface**: seven submission documents, ~30 root documents and a 6,900-line report, all edited
concurrently under deadline, where tonight's three worst defects lived. Three mechanical gates now
cover parts of it, and they caught real errors tonight — but they cover *number co-presence*, not
*claims restated in prose*, which is precisely the failure mode both end-to-end reads found and
neither grep could.

**If one thing gets attention before submission, make it A1** — a dense eval grid between 16 and 32
would put the strongest depth claim in the project on a resolved edge instead of a single interval.
