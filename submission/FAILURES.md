# Every error that reached a number

*The brief warns that a capable coding agent «запросто возьмёт неправильный токенизатор или забудет
сохранить чекпойнт» — that the risk is losing track of the code, not writing it. This is the direct
answer: the complete list, how each was caught, and what it cost. It is placed in the submission
rather than an appendix because it is **evidence**, not apology.*

`../report.md` §6.0 carries all **34** rows with full detail. This document is the shape of them.

---

## The four that cost the most

**1. A headline finding was an artifact of flaky hardware.** A 33-layer untied baseline "could not be
trained stably at all" — NaN between steps 13 and 411 across six configurations — and a mechanism was
built on it (*weight tying as an implicit regulariser*). **Re-run on CUDA, all three learning rates
trained to completion with no NaN, including the one that died at step 13 on MPS.** The claim and its
mechanism were **retracted in full**. *Cost: the most expensive error in the project — reasoning built
on an unverified negative, which survived because the negative was convenient.*

**2. `argmin` was the wrong statistic for every depth claim.** **134 of 165** stored loop curves have
argmin margins under 0.005 nats, against measured noise floors of 0.015–0.068. Caught by building
`src/plateau.py` after a summary line reported an optimum shift decided at **0.0001 nats**. *Cost: it
**killed one finding before publication**, revised another from "2×" to "1.50×", and confirmed five.*

**3. The noise floor was assumed, never measured.** Two *accidental* same-config replicates — found
while auditing something else — put run-to-run variation at **0.031 and 0.068 nats at a fixed seed**.
"Same seed" is not a replicate on this hardware. *Cost: every A/B under ~0.05 nats had been
over-read.* The floor is now measured per device **and per configuration** (CUDA dense 0.0150; CUDA
terminal-only 0.0541 — a 3.6× difference that a single project-wide floor would have hidden).

**4. A claim asserted four times was never tested.** "Loop gain trades against CE" was stated from
four hand-picked pairs. Tested as a stratified correlation over all 43 arms: **pooled ρ = −0.081, and
the strata disagree in sign.** *Cost: demoted from "the report's most robust finding" to a narrow
within-axis statement.*

## The one the brief names by name

**Remotely-trained checkpoints shipped without the vocabulary that produced them.** A vocab mismatch
**does not raise** — it reports CE ≈ ln(4096) = **8.32**, which looks like a broken model rather than
a broken setup. Identity with the local vocabulary had only ever been *inferred from the eval looking
coherent*.

Caught by writing `src/check_tokenizer_identity.py`, which judges against **chance** rather than a
fixed tolerance. **Both checkpoints PASS** (|diff| 0.045 / 0.043 against |CE−chance|/3 ≈ 1.4), so no
number changed — but the kernel now saves the tokenizer alongside the checkpoint, and it ships with
any released weights.

*And the same trap had a second mouth:* the repository's own README told a grader to run
`train_tokenizer.py` first, which **overwrites** `configs/tokenizer.json`. Every released checkpoint
would then have evaluated at chance. Fixed, and the warning is now in the quickstart itself.

## The first pattern, and it is not a coding-bug pattern

**Four of the five most expensive errors share one shape: a number that looked fine in a summary and
was wrong in the raw.** None was a coding bug in the ordinary sense — the code did what it said. They
were failures of **statistic choice, hardware trust, and unexamined assumption.**

That is the argument for building instruments *late* rather than freezing them early: seven of these
were caught on the final day, by tools written that same day, in data that had been sitting
unexamined for a week.

## The second pattern, which is worse and was named late

**The dangerous state is not "unfixed" — it is "recorded as fixed".** Three instances:

- The tokenizer fix landed in the **README** while the shipping path stayed broken.
- The fix for lost checkpoints was written into 23 job configs **as a glob** (`"*_last.pt"`).
  DataSphere does not expand globs, and output paths skip the existence check at submit — so it fails
  silently, server-side, at job end. **22 of 26 configs carried a protection that protected nothing**,
  and it was recorded as done in two documents. *Cost: a live scientific question — the weights that
  would have settled a 0.32-nat disagreement between two replicates.*
- Three retractions were propagated by **grepping the withdrawn number**. That catches numbers and
  misses **claims restated in prose**: the first end-to-end read of the report found **12 defects, 3
  serious, none reachable by grep** — including two sections restating a withdrawn claim four lines
  below its own withdrawal block.

**A fix generates a claim, and this project's rule for claims applies to it: verify against the
artifact, not against the fact that you made the change.**

## The third pattern, which is the most useful of the three because it is diagnosable in advance

The two patterns above are cautionary. This one is **actionable**: it names a check you can run
*before* a number becomes a claim.

**A real statistic, measured in one space, read as a claim about a different space.** Five instances,
and the sharpest three happened within two hours of each other on the final evening:

| the number | what it really measured | the claim it was read as |
|---|---|---|
| `cos → 1.0000` across layers (§4.20) | layer **outputs**, which in a pre-norm stack all share one residual | "the layers have collapsed into one direction" — **an arithmetic artifact**; contributions sit at cos 0.14–0.18 |
| effective rank **31.83 / 33** for an untied stack (§4.7e) `[RANK-PROJECTION]` | **keys**, each through that layer's *own* random `W_K` | "untied stacks build 11.7× more diverse depth **representations**" — at the representation level it is **3.1×**, and both streams are collinear |
| the XSA self-bias falls **0.85 → 0.35** in training | a **cosine** | "so there is little left to remove; the arm will be near-null" — removing that residual was worth **0.216 nats** `[XSA-AT-R1]`. A cosine scale is not a loss scale |
| the angular budget `B` (§4.16c) | a **chord**, sampled once per loop | a **path length**; at 3× resolution the effect reverses sign |
| `argmin` of a loop curve | the position of a minimum **within noise** | "the optimal depth moved" — 134 of 165 curves have argmin margins under the floor |

**The check that would have caught all five is one question, asked before the sentence is written:**
*what space is this number measured in, and what space does my claim need?* It costs nothing, it is
mechanical, and it does not depend on remembering any particular past mistake.

**Why this is better evidence than another admission.** The honest version of the final evening is not
"I knew the pattern and repeated it anyway" — that is true but unproductive. It is that **the pattern
had no name and therefore no check**, and three instances in two hours is what naming it cost. The
first two were caught by an outside question; the third was caught by the arm landing at −0.216 and
refuting a prediction that had been amended on the bad inference twenty minutes earlier.

## A rule that came out of the same evening, and generalises past its occasion

**When you remove a safeguard, name the function it served and say what now covers it.**

`ds_watchdog.sh` was killed at 18:40 for re-attaching a job that had already finished — correct on the
symptom. Its *actual* function was catching dead attaches, and nothing replaced it. By 20:08 three
attach logs had been frozen for over half an hour while the jobs ran on server-side, and a stale step
count had been reported as current. **Results were never at risk** — harvesting goes through
`download-files --id`, not the attach — but live monitoring was silently wrong for ~35 minutes. The
lesson is not "keep noisy safeguards"; it is that removing one is a change with a claim attached, and
this project's rule for claims applies to it.

## What the process caught, and what caught the process

- **Three claims were withdrawn on the final day by pre-registered falsifiers** — criteria written
  before the data existed. Each is recorded with the superseded claim **visible**, not deleted.
- **Instruments are required to pass a null.** The angular budget's null (an untrained model travels
  4.5× further with zero capability) cost that section its headline reading **90 minutes after it was
  written**. Two geometry findings survived their nulls only in narrowed form.
- **The sharpest errors were surfaced by outside questions, not by introspection** — "what is n?",
  "has anyone looked at what it writes?", "does diversity break it?". One case is decisive: a lesson
  was written down, a claim was withdrawn for violating it, and **the same violation was committed
  two hours later on a larger claim.** *Writing a lesson down does not install it.*

---

*Full detail, including 20 further rows: `../report.md` §6.0. Hyperparameters and implementation
choices that were **inherited rather than chosen** — and never screened — are listed separately in
§6.0b, because "things we never examined" is a different and equally important list.*
