# Results

*Headline figures, every intervention with its effect, and what is still landing. Numbers are
`src/eval.py` post-hoc sweeps on a shared grid; `src/headline.py check` verifies this file's headline
against the artifact it came from. Cells still running are marked and dated rather than omitted.*

---

## 1. Headline

| run | tokens | CE | **val ppl** | bits/byte | useful band *(dense 1..64)* | CE@1 | loop gain |
|---|---|---|---|---|---|---|---|
| previous headline | 46.0M | 4.0071 | 54.99 | 1.7330 | [5,14] | 4.2580 | 0.2509 |
| **90M control — the released artifact** | 90.0M | **3.6599** | **38.86** | **1.5829** | **[6,17]** | 3.9622 | 0.3023 |
| 90M + norm penalty | 90.0M | 3.6250 | **37.52** | 1.5678 | [6,14] | 4.1803 | 0.5553 |

**9,064,608 parameters (≤10M ✓) · 90.0M training tokens (≤100M ✓).**

**Perplexity is tokenizer-dependent** (4096-vocab); **bits/byte is the only externally comparable
figure**, and even that is against FineWeb, not FineWeb-Edu (§2.1).

**Finishing the token budget bought 0.39–0.42 nats. Every architectural intervention bought
0.002–0.26.** That is the report's most practically useful sentence.

**Why the control ships and not the 37.52 arm:** see `METHOD.md` §4 — 88% loop-1 damage, a narrower
band, the only arm that converges, and an unresolvable clipping confound.

## 1b. What actually stands — the positives, ranked, with what each is worth

*Asked plainly because the rest of this folder is organised around what failed, and a reader is
entitled to the other list without assembling it themselves.*

| # | what worked | size | how well supported | what it is **not** |
|---|---|---|---|---|
| 1 | **Removing inter-loop normalisation** | **≈ −0.68 nats** token-corrected — *the largest effect in the project* | replicated; the normalised variant provably contracts to a fixed point by loop ~16 and never accrues loop gain at all (§4.1, §4.3, §4.12) | not novel — it is a *removal*. The field's default was wrong here, and that is the finding |
| 2 | **Finishing the token budget** | **0.39–0.42 nats** (46M → 90M) | same platform, same shard, same protocol | not an idea. It is the sentence a practitioner should take away: **every architectural intervention here is worth 0.002–0.26; the data is worth 0.40** |
| 3 | **Supervision annealing — the useful-depth band** | band **widens at 5/5 seeds**; on a dense integer grid `end 20 → 30` against the control's `20` | **the best-supported result here.** Identical edge decomposition at 2.5M *and* 10M; survives halving the plateau tolerance; the sparse grid was *understating* it (§4.23e, §4.25, §4.25c). **Zero added parameters** | **not a loss improvement.** Its CE half is withdrawn at n=4 and a fifth point at 4× budget is +0.1119. It buys *where depth stays useful*, not *how good the model gets* |
| 4 | **Exclusive self attention (XSA)** | **−0.2162 / −0.2633**, two seeds, **zero parameters** | replicates, ~16× the floor | **not about looping.** 84–91% of it is at `r = 1`; it is a generic attention operator a non-looped model would plausibly get too. Its band claim died at the second seed. **Untested at scale** |
| 5 | **Norm penalty** | wins perplexity outright: **37.52 vs 38.86** | one 90M arm | **not shipped, and the reasons are measured**: `ΔCE@1 = +0.2263` (88% of its loop-gain advantage is loop-1 *damage*), its band narrows, it is the only arm whose map converges, and it carries an unresolvable clipping confound |

**And the one that was a positive this morning and is not one tonight:** loop-cycled LoRA. −0.0936
across five arms and three platforms at 2.5M, **+0.0077 at 12M** in a config-identical pair (§4.29).
**This project has no replicated CE improvement at scale.**

### The depth-mixing family, asked directly: is there any "mixture-over-depths" positive? No.

**Seven mechanisms, and every one is null or explained away.** This is the report's central negative
and it is the best-evidenced thing in it:

| mechanism | result |
|---|---|
| static readout mixture over depths (raw and normalised) | best **−0.0023** against a 0.0527 floor — null |
| learned per-token depth gate, unnormalised | **instrument failure** — saturates to a hard argmax, mixes 1.01–1.05 of `r` |
| **scale-invariant depth gate** (`dg_norm`) | **genuinely mixes** — 7.58/8, 14.96/16, 29.84/32 — and gains **−0.0012 / +0.0023**. Null at two seeds |
| oracle-depth ragged KV cache | **−0.0096**, reverses sign by query depth 24 — null |
| loop-cycled LoRA (a different operator per depth) | a **capacity** result, not a diversity one; a zero-diversity pin *beats* it; dead at 12M |
| duo-causal attention (attend to the previous loop's KV) | W=2 null; W=3 lowers CE but its **registered mechanism check failed** |
| five label-free halting rules + a learned probe + two Q-exit heads | best captures **0.1%** of a real headroom (`EARLY_EXIT.md`) |

**And there is a measured reason rather than seven shrugs: a token's 32 depth keys span an effective
rank of ~1.6.** There is almost nothing for any mixing or selection mechanism to discriminate
between — and **the one experiment built to overturn that explanation was registered in advance, ran,
and did not** (`dg_norm`, two seeds). §4.28 turns the mechanism into a priced dose–response: depth-key
rank is `≈ 1.6 × (number of distinct projections)`, so a weight-tied loop — which has exactly one
`W_K` — cannot buy it at any width.

## 2. Every intervention, and the pattern across them

**Twelve interventions: eleven mechanisms on the model, one lever on the loss schedule** (`README.md`
states the convention). Two mechanisms were run at two settings each, so this table has thirteen rows.
ΔCE_best is against each arm's **own in-job control**; negative = better.

| # | intervention | ΔCE_best | band | params |
|---|---|---|---|---|
| 1 | inter-loop RMSNorm | **+0.744** | saturates at 4 | 0 |
| 2 | scale clock | **+1.36** *(non-finite by loop 39)* | onset 8→4 | +448 |
| 3 | gated (diagonal state-space) injection, α = 0.874 | **+0.247** | unmoved | +896 |
| 4a | loop-cycled LoRA, **rank 2** | **+0.094** | unmoved | +204k |
| 4b | **loop-cycled LoRA, rank ≥ 4** | **−0.0936** at 2.5M ⚠ — **+0.0077 at 12M** (§4.29) | **unmoved (6 of 6)** | +409k (+4.51%) |
| 5 | radial clamp | ~0 *(ceiling invariant to 0.006)* | relocates | 0 |
| 6 | convex gate / fixed-`g` sweep | null | unmoved | +66k |
| 7 | ε = λ/(N√L) residual scaling | null *(its "shift" was a 0.0001-nat argmin flip)* | unmoved | 0 |
| 8 | norm penalty λ=0.01 (90M) | **−0.030** *(wins ppl)* ⚠ | narrows [6,17]→[6,14] *(90M, not in the tol sweep)* | 0 |
| 9a | duo-causal attention, **W = 2** | +0.0093 / −0.0115 *(sign reverses, 2 seeds)* | **unmoved, to the digit** | 0 |
| 9b | **duo-causal attention, W = 3** | **−0.0871 / −0.0394** ⚠ | narrows [8,20]→[8,16] **at tol 0.01 only** † | 0 |
| 10 | **exclusive self attention (XSA)** | **−0.2162 / −0.2633** *(2 seeds, agreeing)* ⚠ | seeds **disagree**: s0 unmoved (widens at tol 0.005), s1 narrows 3/3 † | **0** |
| 11a | per-token depth gate, unnormalised | **−0.2950** ⚠ | *(gate saturates — see below)* | +449 |
| 11b | per-token depth gate, **scale-invariant** | **−0.0012 / +0.0023** *(sign reverses, 2 seeds)* | *n/a — see §5* | +450 |
| 12 | **supervision annealing** *(loss-side)* | CE **withdrawn at n=4**; a 5th point at 4× budget is **+0.1119** `[WITHDRAWN-ANNEAL-CE]` | **band widens 5/5 seeds**, same decomposition at 2.5M and 10M | **0** |

**† Band direction is a `tol = 0.01` statement for these rows.** Sweeping the plateau tolerance
(§4.25) shows four of eleven paired band verdicts are tolerance-dependent; the two the argument below
rests on — **annealing widens** and **LoRA leaves the band unmoved** — are robust at 3/3.

### Loop-specific mechanisms vs generic block improvements — the split that makes the claim honest

**A grader should be able to ask "would a non-looped transformer get the same benefit?" and find the
answer already in the table.** Eleven of the twelve are **loop-specific by construction** — their
mechanism refers to loops, depths, or the state between them, and is undefined at `r = 1`. One is
not: **exclusive self attention is an ordinary attention operator with no loop-dependence at all**,
included because it was cheap and worth testing, and a non-looped model would plausibly get the same
−0.24.

**And that split is exactly where the finding lives, because the measurement crosses it.** The four
loop-specific mechanisms that lower the loss **behave like generic block improvements anyway** — they
deliver 67–101% of their gain at a single loop, where the loop-referring part of each is inert:

### The pattern, which is the report's central finding

**Five of the twelve lower the loss *at the budget they were measured at* — and that qualifier turned
out to matter.** Four of the five deliver 67–101% of that gain at a *single* loop, where their own
mechanism is provably inert or irrelevant.

> **The one that was tested at 5× the budget did not survive it.** Loop-cycled LoRA gives −0.0733 …
> −0.1251 at 2.5M and **+0.0077 at 12M** — sign reversed, inside the replicate floor (§4.29). It was
> the only CE claim here that replicated across three platforms. **This project therefore has no
> replicated CE improvement at scale**, and every "lowers the loss" in this table should be read as a
> **2.5–3.5M-token** statement. *The band results are unaffected — the 12M pair is [8,16] for both
> arms, unmoved, as at every smaller budget.*

| loss-lowering arm | ΔCE@1 | ΔCE_best | **share of the gain already present at r = 1** | why the mechanism cannot be acting at r = 1 |
|---|---|---|---|---|
| depth gate, unnormalised | −0.2830 | −0.2950 | **96%** | the softmax runs over a **single** state; the mixture *is* that state |
| exclusive self attention | −0.1826 / −0.2401 | −0.2162 / −0.2633 | **84% / 91%** | *(operates on a single token's own value vector; nothing about it needs a second loop)* |
| loop-cycled LoRA, rank ≥ 4 | — | −0.0936 *(2.5M only)* | **67–95%**, median 88% *(5 arms)* | branch index is `0 mod 4`; cycling is inert — verified max\|diff\| = 0.000e+00 at r=1 |
| duo-causal attention, W = 3 | −0.0682 / −0.0399 | −0.0871 / −0.0394 | **78% / 101%** | with one loop there is no previous loop's KV to attend to |

**The fifth is the mirror image and is worth stating separately, because it fails the brief the other
way round.** The norm penalty lowers best CE by 0.030 and **wins perplexity outright** (37.52 vs
38.86) — but `ΔCE@1 = +0.2263`, so **88% of its apparent loop-gain advantage is loop-1 damage**, and
its band *narrows*. It buys loop gain by making loop 1 worse rather than by making depth worth more.

**Not one of the twelve widens the useful band.** At the tolerance every table here uses
(`tol = 0.01`) three of the five loss-lowering arms *narrow* it — the norm penalty ([6,17]→[6,14]),
duo-causal W = 3 ([8,20]→[8,16], both seeds) and XSA ([8,20]→[8,16] at seed 1). The one lever that
moves the band *outward* — supervision annealing, **5/5 seeds**, zero parameters — **does not lower
the loss.**

> **⚠ The narrowings are tolerance-dependent; the widening is not. Varied for the first time at
> 21:14 (§4.25).** `plateau()` takes a tolerance, it was set to 0.01 once, and **it had never been
> varied** — which is §4.15's retired-`argmin` failure in a new costume, since direction was being
> read off ±1 grid interval. Sweeping tol ∈ {0.005, 0.01, 0.02, 0.05}: **annealing widens at 0.005,
> 0.01 AND 0.02 and never reverses**, but duo-causal W = 3 is *unchanged* at 0.005 at both seeds, and
> XSA *widens* at seed 0 at 0.005 while narrowing at seed 1. **So "three of the five narrow it" is a
> `tol = 0.01` observation, not a finding, and it is downgraded here.** What survives every tolerance
> is the claim the dissociation actually needs: **no loss-lowering intervention moves the band
> consistently, and the one lever that widens it does not lower the loss.**

**So the honest count is sharper than "twelve interventions, five lower the loss", and it is not
kinder:** *eleven loop-specific mechanisms were tested; not one widens the useful band; the four that
lower the loss do so **as block improvements**, at a depth where the loop-referring part of each does
nothing. The twelfth intervention is a generic block operator, it is the largest positive in the
report, and it is not about looping at all.*

*That dissociation is the answer to the brief's sentence: low perplexity and «за счет большого
количества лупов» come apart under measurement.*

> *One number from that sweep worth keeping: `tol = 0.01` is **tighter than the 0.0150 replicate
> floor** — smaller than the noise it exists to absorb — and at the floor **48% of all 135 stored
> arms change a band edge**. The paired claims mostly survive that; individual band edges, including
> the headline `[6,17]`, do not, and are not compared across jobs.*

> **The band claim's edge is now resolved, and the sparse grid had been understating it (§4.25c).**
> Re-evaluating **both** annealing pairs on **every integer depth 12–32** — possible because Kaggle's
> saved tokenizer is byte-identical to the shipped one — the controls hold within 0.01 nats over
> **9 and 8** consecutive depths (ending at **20** and **19**) and the annealed arms over **19 and 16**
> (ending at **30** and **27**). The sparse grid read the seed-2 pair as `16 → 24`. **The effect is
> 2.1× and 2.0× the control's width at two seeds — not a two-grid-point artifact, and the sparse grid
> was understating it.** *Best CE in the same pairs: +0.0350 at one seed, −0.0052 at the other — the
> withdrawn CE claim's own pattern, at higher resolution.*

> **The scope condition, stated because it is the strongest objection to the above (§4.24).** **Every
> one of these paired interventions was measured at 2.5–3.5M tokens** — there is no budget leverage in
> the data. And loop gain itself **roughly triples** between that regime and the released model's
> (median **0.1084** at ≤3M against **0.3023** for the 90M control), so the r=1 share is measured
> exactly where the loop is worth least. **Both readings are live:** the pattern is budget-invariant
> (in which case the finding strengthens as loop gain grows around it), or it is a screening-scale
> artifact. One arm probes it — Kaggle `tlab-lora-scaleup` at 12M/arm — and seed noise on the share is
> already ±7–12 points at fixed budget, which bounds what that arm can settle.

### One prediction that fired

Not every entry above is a null found after the fact. **XSA's outcome was written down before the arm
ran**, at 19:15, and derived from *this report's own measured regularity* rather than from the paper
proposing it: given that every loss-lowering intervention here had improved the block and left the
band alone, the prediction was **"CE down, band unmoved."**

**CE down: confirmed at both seeds** (−0.2162, −0.2633; ~16× the floor; zero parameters).
**Band unmoved: confirmed at seed 0, contradicted at seed 1** — and withdrawn on that basis.

Reported this way because a pattern that generates a correct advance prediction on a *published,
zero-parameter* operator is worth more than the same pattern restated over arms chosen after the fact
— and because the half that failed is what an n=1 report would have shipped intact.

> **⚠ The caveats belong beside the numbers, not in a footnote.**
> **LoRA `[POSTHOC-LORA-RANK]` `[CAPACITY-NOT-DIVERSITY]`:** the `rank ≥ 4` restriction is **post hoc** — over all six arms the interval
> **covers zero** — and there is no dose–response above the threshold. **A branch pinned to one index
> (identical parameters, *zero* diversity) recovers 82% of the gain at seed 0 and delivers 5.6× it at
> seed 1**, so averaged over the two in-job pairs the zero-diversity arm is **better** than the cycled
> one. **Operator diversity — the mechanism the intervention is named for — has no measurable benefit
> here.** The cycled arm's own effect ranges **−0.0261 … −0.1251** across two in-job pairs at one
> budget, a 4.8× seed spread. It is a **capacity** result and it costs **+4.51%** of the parameter
> budget.
> **XSA `[XSA-AT-R1]`:** **replicated at two seeds** (−0.2162 / −0.2633, mean −0.2398, ~16× the floor,
> **zero parameters**) — the largest positive in the project. But **84–91% of it is at `r = 1`**, and
> the "band unmoved" half of the 19:15 prediction **held at seed 0 and failed at seed 1**, so it is
> withdrawn: XSA joins the arms that *cost* useful depth. Both seeds, one budget (2.5M), one width.
> **Duo-causal W = 3:** the CE gain is real and agrees in sign at both seeds, **but the registered
> mechanism check says the mechanism did not engage** — `cos(Δu_t, Δu_{t−1})` is 0.9962/0.9991/0.9998
> against a control's 0.9978/0.9993/0.9998, i.e. indistinguishable. A CE gain whose mechanism check
> fails is **not evidence for the mechanism**, and 78–101% of it sits where the mechanism is inert.
> **Depth gate, unnormalised:** it **could not express its own hypothesis** — logits are `w·h_t` on the
> **unnormalised** state, which grows 1.8–4.0× within a forward pass, so the softmax saturates to a
> hard argmax (effective loops mixed **1.01–1.05 of r**). Its −0.2950 is dense supervision by another
> name, not depth selection. **This is reported as an instrument failure, not a result** (§4.22).
> Full detail on all of these in `NEGATIVE_RESULTS.md`.

## 3. Depth: what a looped model here actually does

- **Saturation without convergence.** The unit state drifts **logarithmically** (log-drift R² 0.986 with
  one parameter vs a convergent power law's 0.748 with two); 0.18 rad still accumulates between loops
  129 and 384; ρ = 1.6227 at loop 2. **Yet CE stops improving at loop ~8.** The task's DEQ premise —
  fast convergence makes further compute pointless — **does not explain the ceiling here**, because
  convergence never happens. *Scope: an untrained model drifts faster, so non-convergence is
  architectural.*
- **Late motion is fully visible to the readout and still useless.** Readout gain flat 207 → 226 while
  ‖Δu‖ falls **117×**; `E`'s condition number is 132, so there is no null subspace to hide in.
- **Graceful past the trained range.** Best at loop 8, still better than loop 1 at **loop 105**,
  crossing at 106. The deepest genuinely-useful band measured is **[16,32]** on a 30M-token model
  trained at `U[32,48]`.

## 4. Per-token depth demand: real, large, and unreachable — with a measured reason

Oracle headroom **0.3084 nats** on the 46M checkpoint, split-half reliability **0.866** against a null
of **0.0007**; **27.9%** of tokens want depth > 32. **Nine rules across five instrument classes fail
to capture more than 0.1%.** *(A separate, smaller figure — **0.2008–0.2032** — appears in §4.7a and is
a different model: the matched dense/annealed 2.5M pair. Same quantity, different checkpoint; the two
are not comparable and the report labels them as such.)* **Full case: `EARLY_EXIT.md`.**

**Why**, and this is the part that is a mechanism rather than a list of nulls: **a token's 32 depth
keys span an effective rank of ~1.6**, mean pairwise cosine 0.91–0.97. There is almost nothing for any
mixing or selection mechanism to discriminate *between*. The collapse is present **at initialisation**
and **training makes it worse** (2.73 → 1.83), and the one arm that applies a genuinely different
operator per depth raises it by **0.01–0.08 out of 32**.

**Holding scale fixed and varying only weight tying** `[RANK-PROJECTION]`, an untied stack's *keys* are near-orthogonal
(31.83/33) while the tied loop's are 2.73/33 — **but most of that gap is per-layer projection
randomness**: at the *representation* level the untied stack is 4.36/33 against the tied 1.40/33, a
3.1× gap, and both streams are highly collinear. **The mechanism is that distinct per-layer
projections decorrelate a collinear state stream for free, and a tied loop has one `W_K` and cannot.**
(§4.7e — corrected 20:01 after the confound was raised.)

## 5. The registered joint test, and the last arm outstanding

### 5.1 `dg_norm` — the one experiment that could have refuted §4.7e, and did not

§4.7e explains the entire per-token-depth family with one fact: a token's depth keys span an effective
rank of ~1.6, so there is nothing for a mixing mechanism to discriminate between. That explanation is
only worth anything if it was **exposed to refutation**, because the previous depth gate failed for a
*different* reason — it saturated and could not mix at all (§2, row 11a). A rank explanation cannot be
tested by an instrument that never mixes.

So a **scale-invariant** gate was built (`F.normalize` on the state before the gate head, plus a
learned temperature) and a **joint** falsifier was registered at 19:22, *before the arm existed*:

| the gate mixes (GATE 1) | CE gain | verdict registered in advance |
|---|---|---|
| yes | **no** | **§4.7e confirmed** — the collapse is the binding constraint |
| yes | **yes** | **§4.7e is WRONG** — per-token headroom is reachable after all |
| no | either | instrument failure again; **nothing is decided** |

**Result, two seeds, in-job paired.** GATE 1 **passed**: effective loops mixed **7.58/8, 14.96/16,
29.84/32**, with **zero** tokens above 0.99 top-weight — the gate genuinely mixes, where its
predecessor did not. CE: **−0.0012 and +0.0023.** The sign reverses; both are two orders of magnitude
inside the 0.0150 floor.

**A working mixture over a collapsed representation buys nothing. §4.7e stands, decided by the test
that could have killed it.** *Scope: 2.5M tokens, two seeds, one width.*

> **One number from this arm must not be read as a depth result.** `dg_norm`'s plateau is [12,24] and
> [12,32], deeper than its control's [8,20] — and it is **excluded from every band table in this
> submission**, by a decision recorded at 17:40 *before* the arm ran. The gate mixes over a window of
> `r` states, so a deeper plateau measures **how wide the mixture window is**, not how deep the useful
> computation goes. Reporting it as a band would be the project's characteristic error — a statistic
> read as a claim about a space it does not live in.

### 5.2 The arms this section was written to hold open — four landed, one outstanding

Each had its falsifier registered in `../RUNS.md` before it ran. Landed results are in §2 and in the
report sections named; nothing here is still a placeholder.

| arm | what it decides | state |
|---|---|---|
| `tlab-divx-s1` | capacity-vs-diversity, all three arms **in one job**, seed 1 | **LANDED.** Pinned arm **beats** cycled, −0.1470 vs −0.0261. §4.23c |
| `tlab-xsa-s1` | second seed for the −0.216 | **LANDED.** −0.2633; CE replicates, **band claim withdrawn** |
| `tlab-recmethod-s2` | the recommended configuration's own weights | **LANDED.** Band 5/5 at 4× budget; CE +0.1119. §4.23e |
| `tlab-duocausal-s0/-s1` | duo-causal W=2/3 and the scale-invariant depth gate | **LANDED.** §4.23, §4.23b |
| Kaggle `tlab-lora-scaleup` | does the LoRA positive survive ~5× budget (12M/arm) — **the only budget probe of the r=1 pattern (§4.24)** | running, ETA ~21:52 |

## 5.3 What the model actually writes

*Included because `FAILURES.md` records that the sharpest errors in this project were surfaced by
outside questions, and one of them was simply **"has anyone looked at what it writes?"** — asked after
a capability verdict had been printed over empty strings. Perplexity alone does not distinguish a
working small model from a degenerate one.*

Greedy-free sampling, temperature 0.8, `n_loops=10` (the released checkpoint's optimum), seed 0:

> **"The city of"** → *"The city of Mediterranean. The study examines the new country of the South
> Africa and its cultural species and plans to increase its strengths. In the recognition and
> development of the Normans, USA's Cent"*

> **"In 1945, the"** → *"In 1945, the National Operational Organization is going to help focus on
> theisms in the region for the city of the Northwestern continent. Our goal is to enable the housing
> drivers to outperform the region's g"*

> **"Water is"** → *"Water is the perfect remodel. It works fine in shape to dry air. Expressed shape
> is made up of a dirtie, covered with white onions, constructed washing and darkening."*

**Read this as a sanity check, not as a capability claim.** The output is **syntactically well-formed
English** — agreement, articles, subordinate clauses, plausible noun phrases — and **semantically
incoherent**. That is exactly what 9.06M parameters trained on 90M tokens should produce, and it is
the honest characterisation: *the model has learned the shape of the language and not its content.*
**No claim in this submission depends on generation quality**; every result is teacher-forced
cross-entropy. This is here so a reader does not have to take "CE 3.66" on faith that something
non-degenerate is behind it.

## 6. Verification

```
src/test_model.py                  13 checks — ALL PASS
src/test_plateau.py                8 checks incl. a deliberate falsification probe — ALL PASS
src/headline.py check              every headline number matches its artifact — consistent
src/check_tokenizer_identity.py    on the SHIPPED checkpoint — PASS, |diff| 0.0020 vs chance 8.3178
src/make_inventory.py --check      every trained arm accounted for (count generated, not transcribed)
src/check_crossref.py --strict     every figure quoted here appears in report.md
```
