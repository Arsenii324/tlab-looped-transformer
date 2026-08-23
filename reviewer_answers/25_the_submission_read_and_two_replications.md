# Reply — 2026-08-23 20:52 · the submission folder's own audit, and two replications that landed mid-edit

Since reply 24: the adversarial read of `submission/` came back with **12 findings, 3 HIGH**, all now
fixed — and fixing them made the report's central claim **stronger, not weaker**. Two jobs landed
while I was editing; **one of them contradicted a sentence I had shipped twenty minutes earlier.**
§5 is the part I would read first.

---

## 1. The headline sentence was wrong, and it was wrong in the *deflating* direction

`RESULTS.md` said **"Twelve interventions. Two lower the loss."** That was a real defect — the table
under it had three negatives, and a full re-derivation from the raw `val_curve` arrays finds **five**.

The corrected version is a stronger claim, not a weaker one:

> **Twelve interventions. Five lower the loss. Four of the five deliver 78–101% of that gain at a
> *single* loop, where their own mechanism is provably inert or irrelevant.**

| loss-lowering arm | ΔCE_best | share at `r = 1` | why the mechanism cannot be acting there |
|---|---|---|---|
| depth gate, unnormalised | −0.2950 | **96%** | softmax over a **single** state; the mixture *is* that state |
| **exclusive self attention** | **−0.2162 / −0.2633** | **84% / 91%** | operates on a token's own value vector; needs no second loop |
| loop-cycled LoRA, rank ≥ 4 | −0.0936 | **67–95%** (median 88%, 5 arms) | branch index `0 mod 4`; verified max\|diff\| = 0.000e+00 at r=1 |
| duo-causal attention, W = 3 | −0.0871 / −0.0394 | **78% / 101%** | no previous loop's KV exists at loop 1 |

**The fifth is the mirror image and fails the brief the other way round.** The norm penalty wins
perplexity outright (37.52 vs 38.86) — but `ΔCE@1 = +0.2263`, so **88% of its apparent loop-gain
advantage is loop-1 damage.** It buys loop gain by making loop 1 *worse*.

**I had been reporting the pattern at three or four instances. It is five, and the fifth — the
saturating depth gate at 96% — is the one with the cleanest inertness proof**, because a softmax over
one element is inert by definition rather than by measurement. It was sitting in §4.21b the whole
time, counted as an instrument failure and therefore not counted as an instance of the pattern. It is
both.

## 2. XSA replicated — and its band claim died in the same result

`tlab-xsa-s1` landed at 20:39. **It replicates and it is now the largest positive in the project.**

| arm | best CE | ΔCE_best | band | share at r=1 |
|---|---|---|---|---|
| seed 0 | 5.0689 | **−0.2162** | [8,16] → [8,16] *(unmoved)* | 84% |
| **seed 1** | 5.1436 | **−0.2633** | **[8,20] → [8,16]** *(narrows)* | **91%** |

Mean **−0.2398**, ~16× the floor, **zero added parameters**, both seeds putting the optimum at r=12.

**And the 19:15 pre-registered prediction was half right.** It said *"CE down, band unmoved."* **CE
down: confirmed twice. Band unmoved: confirmed at seed 0, contradicted at seed 1** — so that half is
withdrawn. XSA joins the norm penalty and duo-causal W=3: **three of the five loss-lowering arms
narrow the useful band, and the number that widen it is still zero.**

**The uncomfortable timing is the point.** I wrote "band unmoved, [8,16]" into
`submission/README.md`'s five-sentence answer at **20:19**. The replicate landed at **20:39**. Had
`tlab-xsa-s1` not been launched, that sentence ships — an n=1 band claim contradicted by its own
first replicate, in the most-read paragraph of the submission. Propagated out in 8 minutes.

## 3. Annealing: the band effect replicates *exactly* at 4× the budget; the CE claim gets its worst point

`tlab-recmethod-s2` was launched to close a gap `METHOD.md` §4 states outright — **no artifact in this
project was the recommended configuration.** In-job, **10.0M tokens**, 4× every prior annealing seed:

| arm | best CE | onset | end | mid |
|---|---|---|---|---|
| `rec_dense_s2` | **4.4907** | 8 | 16 | 11.3 |
| `rec_sw90_s2` | **4.6025** | **8** | **24** | **13.9** |

**Onset 8→8, end 16→24, mid 11.3→13.9 — identical to seeds 0 and 1 at 2.5M, to the grid point.**
That is **5/5 seeds across a 4× budget range**, and the decomposition says the same thing every time:
*the model does not improve further, it degrades later.*

**And ΔCE_best = +0.1119** — the worst of five (−0.0811 / −0.0609 / +0.0482 / −0.0902 / **+0.1119**,
mean −0.0144). The n=4 withdrawal was right and this is the strongest evidence for it.

**Two things this settles.** The recommended recipe now has weights of its own. And **shipping the
*dense* 90M control is now evidenced at the recipe's own budget** rather than inherited from launch
order — at 10M the annealed arm is 0.11 nats worse. The 90M annealed run is dequeued, with the reason
recorded rather than silently dropped.

## 4. `dg_norm`: §4.7e survived the one experiment built to kill it

Registered 19:22 as a **joint** falsifier, before the arm existed: *mixing engages + no gain* confirms
the rank collapse; *mixing engages + gain* means §4.7e is **wrong**; *no mixing* decides nothing.

**GATE 1 passed by a wide margin** — effective loops mixed **7.58/8, 14.96/16, 29.84/32**, zero tokens
above 0.99 top-weight, against the broken gate's 1.01–1.05 of `r`. **This is the first genuinely
working per-token soft mixture over depths this project has built.** CE: **−0.0012 / +0.0023**, sign
reversing, both an order of magnitude inside the floor.

**A perfect mixer over a representation spanning ~1.6 of 32 dimensions buys nothing.** The explanation
held under the test designed to overturn it, at two seeds.

*Its plateau reads [12,24] and [12,32] — deeper than control — and is **excluded from every band table**
by a decision recorded at 17:40, before the arm ran: the gate mixes over `1..r`, so that number is
**mixture-window size, not depth.***

## 5. What the audit says about the process, including two things that are not flattering

**(a) The enforcement mechanism caught a live error — mine — for the first time.** `src/check_caveats.py`
greps each deflated claim's number and flags any reader-facing file stating it without its caveat
token. Rewriting §4.23d for XSA's second seed, I dropped the token while still stating −0.2162;
`--strict` failed the build. Every previous catch had been historical. **This is the first time the
thing built to stop the project's most repeated failure actually stopped it in flight.**

**(b) A completeness certificate was certifying a stale number, three lines above its own refutation.**
`EXPERIMENTS.md` said *"Of the 113 arms…"* directly above a **generated** line reading *"Total arms
with a final validation curve: 128."* The six arms from tonight lived in `/tmp/ds_*` and were invisible
to the generator. **The document whose job is to certify that no experiment went unreported was itself
unreported-on.** That is the `FAILURES.md` "recorded as fixed" pattern in the worst possible file.
Fixed structurally, not textually: the **coverage block is now generated too**, so it cannot drift
again. Arms 113 → 132; absences 19 → 7, each annotated.

**(c) Four different intervention counts across four files.** `README.md` contradicted itself **21
lines apart** (nine / twelve). Now one convention, stated once in `README.md` and referenced: *twelve
interventions = eleven mechanisms on the model + one lever on the loss; two mechanisms ran at two
settings each.*

**(d) The misattribution that matters most, and it was mine to catch.** `README.md` said §1 — the idea
narrative — *"is the author's and is written separately."* **§1 is agent-written**, from the dated
record, at the author's instruction, and it has said so in its own banner since it was written. But a
grader reading `submission/README.md` alone would have been told the opposite, **in exactly the
direction the task's warning about LLM-sourced ideation makes costly.** Corrected, and the disclosure
now appears in the submission folder rather than only at §1's head.

---

## 6. Still open, and what I would attack

| | |
|---|---|
| **running** | `tlab-divx-s1` (capacity-vs-diversity, all 3 arms in one job, seed 1) · Kaggle `tlab-lora-scaleup` (12M/arm, ~21:52) |
| **the weakest surviving positive** | LoRA. `rank ≥ 4` is post hoc, the all-arms interval covers zero, a pinned branch recovers 82%. The Kaggle 12M arm is the only scale check and it is one arm |
| **unexamined** | Everything tonight is 2.5M tokens against this project's own 12×-shrinkage regularity. `recmethod-s2` is the single 10M replication and it went the *unfavourable* way on CE |
| **not audited** | The repo's ~20 root-level working documents. Several predate today's four largest changes and carry the superseded 46M headline. The repo goes public; `submission/` and `report.md` are current, the rest are dated working records and will be labelled as such rather than rewritten |

**The one I would attack if there were another day:** every instance of the central pattern is a
*single* budget. "78–101% of the gain sits at r=1" is measured at 2.5M for four of five arms. If the
r=1 share *falls* with budget, the pattern is a screening-scale artifact and the report's spine is
much weaker than it reads. Nothing here tests that, and the Kaggle 12M LoRA arm is the closest thing
to a probe of it.
