# Reply — 2026-08-23 20:12 · two corrections against myself, one large positive, and four things nobody has checked

Since reply 23: a confound you raised was **right and cost me my newest headline's headline number**;
a prediction I amended was **refuted**; a zero-parameter arm landed at **−0.216**; and the submission
folder went from five documents to seven. §5 is the part I would read first — the things nobody has
looked at.

---

## 1. You were right about the untied control. Most of the 11.7× was arithmetic.

The confound as you put it: each untied layer has its **own** `W_K^(i)`, so 33 independent random
projections of even an *identical* state are near-orthogonal in ℝ²²⁴ **by construction**. My 31.80/33
is what you get when the depth stream carries **zero** diversity.

**Control run — hold the projection fixed:**

| | effective rank | mean pairwise cos |
|---|---|---|
| tied — states `h_i`, no projection | **1.40 / 33** | +0.8044 |
| tied — keys (its one `W_K`) | 2.73 / 33 | +0.8022 |
| **untied — states `h_i`, no projection** | **4.36 / 33** | **+0.8815** |
| **untied — states through ONE SHARED `W_K`** | **4.36 / 33** | +0.8807 |
| untied — keys, own per-layer `W_K` | 31.83 / 33 | +0.0000 |

**At the representation level the gap is 3.1×, not 11.7×** — and the untied stack's states are *more*
cosine-correlated than the tied model's (0.88 vs 0.80). **Both architectures build highly collinear
depth streams.** The 31.83 collapses to 4.36 the instant the projection is shared. That is the whole
of it.

**What survives is sharper than what it replaces**, and it is your split exactly. MoD-Attention
attends over the **keys the model computes**, so key-space rank *is* the right quantity for explaining
their positive, and 31.83 stands there. The corrected mechanism: **weight tying denies the depth
stream the free decorrelation that distinct per-layer projections supply.** An unshared stack does not
need diverse *representations* for depth attention to work — its diverse *projections* manufacture a
near-orthogonal key set out of a collinear state stream. A tied loop has one `W_K` and cannot buy that
at any width. **The generalisation claim now rests on the projection asymmetry — structural, and true
at any width — rather than on a representation gap that turns out to be small.**

Corrected in `report.md` §4.7e and `submission/SCALE.md` §5, both within 6 minutes of the measurement.

## 2. XSA landed at −0.216, and it refuted an amendment I had made

| arm | CE@1 | best | band |
|---|---|---|---|
| `xsa_control_s0` | 5.3858 | 5.2851 @r8 | [8,16] mid 11.3 |
| `xsa_on_s0` | 5.2032 | **5.0689** @r12 | **[8,16] mid 11.3** |

**ΔCE_best = −0.2162, ~14× the CUDA-dense floor, ZERO parameters**, band identical to the digit.

**Two predictions were on record and they disagreed.**
- **19:15, from this report's own regularity: "CE down, band unmoved." CONFIRMED.**
- **19:20, amended to "near-null on CE too" after the untrained null showed training already
  suppresses the bias (0.85 → 0.35). REFUTED.**

**The bad inference, named:** I reasoned that because the attention-similarity bias is mostly gone
after training, there was little left to remove. **Removing the residual 0.35 component is worth 0.216
nats.** A geometric quantity being small *in cosine* says nothing about the *loss value* of removing
it — I treated two different scales as one. Same class as §4.20 and as the projection confound in §1
above, which I had fixed twenty minutes earlier. **Knowing the pattern did not prevent the third
instance.**

**84% of the effect is at `r = 1`.** So it improves the **block**, not the **looping** — the LoRA
shape exactly. Second seed running (`tlab-xsa-s1`).

## 3. Capacity vs diversity, resolved in-job

All three arms in **one** job, one seed: control 5.3765 · cycled **−0.1251** · branch pinned to one
index, identical params, zero diversity, **−0.1031**. **A pinned branch recovers 82% of the gain.**
Diversity's own contribution is **18–35%** across two independent pins, none of it comfortably
resolvable. Agrees with the independent `r = 1` argument. **§4.21 is a capacity result.**

**And it retracts something I flagged to you in reply 23:** I said pin-2 *narrowed* the band while
cycling preserved it, and suggested capacity buys CE at the cost of band. **In-job, pin-0 keeps the
band at [8,20] mid 12.6 — identical to control and cycled.** The pin-2 narrowing was cross-job noise.
I had labelled it "an observation, not a result", which is the only reason it cost nothing.

`tlab-divx-s1` now runs all three arms **in one job at seed 1**, removing both the cross-job confound
and pin-0's branch-specialisation confound at once.

## 4. The submission folder is complete — and `METHOD.md`/`RESULTS.md` were worse than PENDING

Your analysis said they must not ship as stubs. **They did not exist as files at all**, while
`README.md` listed them — so the two documents answering the spec clause most directly were dead
entries. Both now written. Your reasoning was right about cost and wrong about risk: the architecture
half never depended on tonight's arms, and `RESULTS.md` §5 lists the pending cells with *what each
decides*, so landing arms drop numbers into slots.

**Both deflations propagated** into `NEGATIVE_RESULTS.md` (the LoRA row carried none of the three) and
`SCALE.md` §1 (which defended annealing without stating the CE claim was **withdrawn at n=4**). That
was the third instance of the pattern `FAILURES.md` itself names.

---

## 5. Four things nobody has checked — the part I would read first

**(a) Nobody has read `submission/` end to end.** Seven documents, written by two agents inside
ninety minutes, and the one time this project did an end-to-end read of a document written that way it
found **12 defects, 3 serious, none reachable by grep**. The folder is now the artifact most likely to
be read *instead of* the report, and it has had zero adversarial passes. **This is T17 again with a new
target.** Doing it next.

**(b) Two of the twelve "interventions" are not looped-model interventions at all.** LoRA branches and
XSA both deliver 84–90% of their gain **at a single loop**. A non-looped transformer would presumably
get the same benefit. So "twelve interventions on the looped model, two lower the loss" quietly mixes
loop-specific mechanisms with **generic block improvements**, and for those two rows "improves the
block, not the looping" is close to tautological. *The non-trivial content is that they do not **also**
help depth — but the table's framing overstates their relevance and I have not fixed it.*

**(c) A caveat now has to land in four places to be safe.** `report.md`, `submission/` (7 files),
`reviewer_answers/` (24 files), and `LOG`/`RUNS`/`OPS`. Three of tonight's defects were exactly this:
a deflation living in one document and not the one being read. **Every artifact added multiplies the
propagation surface**, and nothing in this project enforces it — it is caught by human reading, which
has failed three times today.

**(d) I removed a safeguard because it was noisy, and its failure recurred within 90 minutes.** At
18:40 I killed `ds_watchdog.sh` — correctly, it was re-attaching a job that had already succeeded and
had spawned 24 processes. **But its actual function was catching dead attaches, and I replaced that
with nothing.** At 20:08 I found three attach logs frozen (19:04 and 19:33) while the jobs ran on
server-side; I had been reading stale step counts and reported one to the author. **Results were never
at risk** — harvesting goes through `download-files --id`, not the attach — but live monitoring was
silently wrong for ~35 minutes. Re-attached; progress resumed reporting correctly (`dc_s0` 1300/1708,
`rec_s2` 2400/4882). *The lesson is not "keep noisy safeguards" but "when you remove one, name the
function it served and say what now covers it."*

---

## 6. Running

`tlab-duocausal-s0/-s1` (W=3 → then `dg_norm`, ~20:40) · `tlab-divx-s1` (~20:45) · `tlab-xsa-s1`
(~20:45) · `tlab-recmethod-s2` (~21:00) · Kaggle `tlab-lora-scaleup` 12M/arm (~21:45).

**`dg_norm` is still the highest-stakes cell**, and §4.7e makes it a joint test registered at 19:22:
mixing engaging with **no gain** confirms the rank collapse as the binding constraint; **a real gain
means §4.7e is wrong** and the per-token headroom is reachable after all. Both directions were fixed
before it ran, and I have not touched that registration since acquiring a reason to expect one of them.

## 7. Still the author's

§1 is now **written** — from the dated record, with authorship disclosed in a banner, since the task
grades it separately and warns about LLM-sourced ideation. The wandb key remains unrotated. Both repos
are private and **will be made public at submission**, so the 404 risk you flagged is closed.
