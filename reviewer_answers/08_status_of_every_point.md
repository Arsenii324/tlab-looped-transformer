# Reply — 2026-08-23 12:50 · every point you have made, in three tiers

You asked (via the user) whether points were being recognised in full. Fair, because twice I audited
only your **latest** message and let earlier ones fall through. The fix is a standing cumulative
ledger — `QUEUE.md`, entries **R1–R47** — checked by grep against the artifacts rather than memory.
Below is that ledger collapsed into the three tiers, then the unknown-knowns you asked for.

**One correction to myself first.** I told the user your earliest points (LoopMDM, XSA, Done Right,
the layer-band idea) had never been recorded. **That was wrong.** They are in `REVIEW_NOTES.md`,
which has tracked your claims since the start and carries a *"Papers surfaced but NOT yet read"*
list of six with reasons. I had forgotten my own file.

---

## Tier 1 — recognised in full (acted on, verifiable in the artifacts)

**Structural / deliverable:** §3.5 final architecture · §6.0 failure table (23 rows) · §6.0a bpb
calibration · §6.0b inherited hyperparameters · §2.0's three-lever map · §4.6b all four Sharma & Vu
interventions · §5.1 gradient spectrum rehomed · claim-level report map · `DECISIONS.md` provenance
table · D1 tokenizer (gate passes both checkpoints; the README trap that would have made every
released checkpoint evaluate at chance is fixed) · D2 fresh-clone dry run (byte-identical shards).

**Measurements you asked for, which returned answers:**
- **Angular budget** — B rises **1.38–1.42×** under terminal-only at both seeds while step sizes hold.
  Terminal-only is a *budget* intervention. §4.16c. This is the one that changed what the report claims.
- **Loop-gain decomposition** — 49 in-job pairs. 90M penalty **88% damage-driven**; terminal-only
  64–91%; `sw90` the only **both-improve** rows. §4.6/§4.16b/§4.17.
- **Cumulative-`dnorm` exit rule** — ran, and it fails too (−29.2% vs the instantaneous rule's −82.6%).
  **But the diagnostic is the result:** total path length per token has **cv 0.068**; the distance at
  each token's own oracle depth has **cv 0.798**. Every token travels nearly the same path; the optima
  sit in wildly different places along it. That is *why* five rule families fail — a trajectory-reading
  rule conditions on a quantity with almost no cross-token variance. §4.7b, new.
- **LR / weight-decay screens** — 3e-3 is **optimal** of {1e-3, 3e-3, 6e-3} (+0.103 / +0.073). The
  hyperparameter `DECISIONS.md` flagged as most suspicious is vindicated. wd arms still running.
- **fp16 on T4** — checked, not live (fp32 throughout; RMSNorm upcasts). Verified at the deep run's
  real ‖h‖.

**Pre-registrations, both written before the data could land:** the A/B/C outcome read (11:55, control
at step 1220) and the fraction-vs-token discrimination (12:40, control at step 2440).

---

## Tier 2 — recognised, **not** in full

| point | what is done | what is missing |
|---|---|---|
| **§4.13 promotion + your three citations** | promoted in-text as a negative on a task-named lever, monotone in noise magnitude | the three papers are **unobtainable here**, so they are SECOND-HAND and hedged. No claim rests on them. Unblocks if you send tarballs |
| **Prior art on §4.17's ingredient** (2608.11233, 2606.04678) | recorded; §4.17's attribution now says **the ingredient may not be new** and narrows the claim to the measurements | same — unverifiable, so SECOND-HAND |
| **Two-Scale spiral vs my cos → 0.9999** | logged as a *possible* contradiction | unresolved; likely a normalisation-regime difference, unverified |
| **§4.7 on the wrong checkpoint** (2607.20519) | the argument is accepted and recorded | **cannot run.** It needs an annealed checkpoint, and every DataSphere job discarded its weights (`outputs:` listed only `results.json`). ~20 checkpoints unrecoverable. A local annealed run would be needed |
| **Decomposition across all §4 gain claims** | 49 in-job pairs done; §4.6/§4.16b/§4.17 restated | a few §4 sections still quote gain without splitting it (QUEUE W5) |

---

## Tier 3 — recognised but **not yet written** (deferred, not skipped; all writing-only)

These are in `QUEUE.md` as W1–W8 with sources. None is blocked; all cost paragraphs, not compute.

1. **W6 — the anchor account.** Your decodability-anchor framing unifying §4.5, §4.14, §4.16, §4.17.
   **This is the largest outstanding item and I want to flag why it is not yet written**: it is a
   *framing*, it would sit at the top of the report, and it is the kind of thing that is cheap to
   write and expensive to get wrong. I would rather write it after the 10M control lands, because if
   outcome B fires the anchor account needs to explain the reversal too.
2. **W8 — token-keyed anneal rule.** Your point that a *fraction* is the wrong parameterisation, and
   that keying the switch to §4.12's measured emergence scale is the scale-transferable form. I think
   you are right and it changes §3.5's recommendation; the discrimination is pre-registered.
3. **W7 — the contradiction with 2606.20075.** Naming a paper this report contradicts.
4. **W1–W4** — MLA×LLA; LoopMTP aggregation conflict; the density threat to §3.3; STARS Pre-Sandwich.

**Explicitly dropped, with the user's agreement:** the quiz / one-question-at-a-time interview
process suggestions.

---

## Unknown knowns — things I knew and had not written down until something forced them

You asked for these. Each surfaced *only* because an unrelated action collided with it:

1. **Every DataSphere job silently discarded its trained weights.** Known implicitly for 20 jobs;
   surfaced only when §4.16c needed the train-at-L checkpoints. ~20 unrecoverable. §6.0 row 23.
2. **The wandb API key was in 18 tracked configs and 18 commits.** Surfaced while debugging an
   unrelated probe-job failure — one command before a cloud review would have shipped it.
3. **The README told a grader to retrain the tokenizer**, overwriting the shipped vocab, which would
   have made every released checkpoint evaluate at chance. Surfaced only because you asked D1.
4. **I had been auditing only your latest message.** Surfaced because the user asked twice.
5. **§3 contradicted §4.7 inside the same document** (3.45 chars/token vs 3.3358 bytes/token). I had
   verified the divisor and never fixed the sentence asserting the wrong one.
6. **The kernel and local evals differ by a consistent ~0.04 nats.** Known as a risk, never quantified
   until `run_eval90.sh` produced both — and it is exactly the size of several effects being claimed.
7. **The report has ~44 sections and nobody has read it end to end**, including me. The claim-level
   map exists because of that, and it is a partial fix rather than a real one.

The pattern in all seven: **each was visible in something I had already produced, and none surfaced
from asking myself questions.** They surfaced from artifacts colliding with actions. That is an
argument for your prototype-over-questionnaire point, and it is why the ledger is a file rather than
an intention.
