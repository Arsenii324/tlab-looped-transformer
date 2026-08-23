# A looped transformer on FineWeb — submission

| | |
|---|---|
| **validation perplexity** | **38.86** (CE 3.6599) · **bits/byte 1.5829** |
| **useful band** | **loops 6–17**, on a dense every-integer 1..64 sweep |
| parameters | **9,064,608** (cap 10M) † |
| training tokens | **89,999,360** (cap 100M) |
| architecture | Qwen3-style 3-layer block, weight-tied, applied `r` times. No prelude, no coda, no inter-loop norm, additive re-injection, learned `h₀` |

*† Count with `sum(p.numel() for p in model.parameters())`. A `state_dict` sum returns 10,899,616 and
looks like a cap violation — see "Counting the parameters" below.*

## The result

The trajectory never converges and saturates anyway. The depths a weight-tied loop visits are
near-indistinguishable, and when we gave the architecture the capacity to avoid that, training
collapsed it again. We built the experiment that could have refuted the mechanism, registered its
criterion before the data existed, and it held.

The task asks for low perplexity *by exploiting many loops*. Those two come apart here, and what this
submission adds is the measured reason: a token's 32 depth keys span an effective rank of **~1.6 out
of 32**, present at initialisation and worse after training. A weight-tied loop has one key projection
where an unshared stack has one per layer and gets decorrelation for free, so there is almost nothing
for a depth-mixing or early-exit mechanism to discriminate between.

The last experiment tested whether that is fixable by architecture, and it is not. Given four distinct
key projections, costing **+10.0%** of the parameter budget, rank starts at **8.818/32** and trains
back down to **1.74**, against a tied control's **1.66**. The collapse is what this training objective
drives the representation toward, even where the architecture can avoid it. That predicts no purely
architectural fix in this family will hold, and it accounts for the published positives in the
depth-mixture family, which are measured on unshared stacks.

That experiment was built to decide a different question: whether the rank collapse *causes* the
depth-mixing failure. Its registered gate required the trained rank to exceed ~4. It came back at
1.74, so the causal question is undecided and `LIMITATIONS.md` reports it as an open absence. What is
measured is the collapse under training. One seed, 3.5M tokens.

## The answer to the brief, in five sentences

1. **The brief's own diagnosis does not hold here.** It attributes saturation to DEQ-style
   convergence. This model's state never converges: the unit drifts *logarithmically* (R² **0.986** on
   one parameter against a convergent power law's **0.748**; ρ > 1 at every measured depth; 0.18 rad
   still accumulating between loops 129 and 384). It saturates at loop ~8 regardless. The drift is
   architectural, since an untrained model of the same shape drifts *faster*.

2. **What binds is that the depths are not distinguishable.** A token's 32 depth keys span an
   effective rank of **~1.6**, present at initialisation and worse after training. The cause is
   projection asymmetry: an unshared stack manufactures a near-orthogonal key set from a state stream
   that is *just as collinear* as the tied one (**4.36** vs **1.40** of 33), because each layer owns a
   `W_K` `[RANK-PROJECTION]`. Buying that back architecturally does not work; see *The result* above
   and §4.30.

3. **We built the experiment that would have refuted it, and registered the criterion before the arm
   existed.** A scale-invariant depth gate that mixes for real, **7.58/8, 14.96/16, 29.84/32**
   effective loops with zero tokens above 0.99 top-weight, where the project's earlier gate saturated
   to a hard argmax. The falsifier was written first: *mixes and gains ⇒ the explanation is wrong.* It
   mixes, and it returns **−0.0012 / +0.0023** at two seeds. A working mixture over a collapsed
   representation buys nothing. Mixture-over-depths was tested seven ways; every one is null, an
   instrument failure, or a gain whose own mechanism check failed (`RESULTS.md` §1b).

4. **Twelve interventions. Five lower the loss. Not one widens the useful band**, at any tolerance
   tested. Four of the five put **67–101%** of their gain at a single loop where their own mechanism is
   provably inert: LoRA 67–95% `[POSTHOC-LORA-RANK]` `[CAPACITY-NOT-DIVERSITY]`, XSA 84–91%
   `[XSA-AT-R1]`, duo-causal W = 3 78–101%, the saturating depth gate 96%. They improve the block, not
   the looping. The fifth inverts that: the norm penalty wins perplexity (**37.52** vs **38.86**) by
   damaging loop 1 (`ΔCE@1 = +0.2263`). And the one that replicated across three platforms vanishes at
   5× the budget, **−0.0936** at 2.5M against **+0.0077** at 12M in a config-identical pair. **This
   project has no replicated CE improvement at scale.** (§4.29; `RESULTS.md` §1b for what survives.)

5. **One lever moves depth, costs zero parameters, and survives the two tests the others failed.**
   Supervision annealing widens the band at **5 of 5 seeds**, with the same edge decomposition at 2.5M
   and at 10M. On an every-integer sweep of depths 12–32 the annealed arms hold within tolerance over
   **2.1× and 2.0×** as many depths as their controls. It is the only band claim here that is robust to
   the plateau tolerance, resolved to ±1 loop, and replicated across seeds. **It does not lower the
   loss**; that half of the claim is withdrawn. (`METHOD.md` §2 for the withdrawal and its five seeds;
   §4.25c for the sweep.)

Per-token depth demand is real and unreachable, which is the subject of `EARLY_EXIT.md`. The evidence
for "real" is the split-half reliability, **0.866** against a null of **0.0007**, and not the
0.3084-nat oracle headroom, whose two nulls are mis-specified: they destroy the per-token curves'
smoothness (4.6× rougher) and produce *more* headroom than the real data (**0.3877**, **0.4110**).
Eight rules across five instrument classes capture at most **0.1%** of it, for the reason in
sentence 2.

The brief states that *«отсутствие положительного результата при хорошем анализе всех негативных —
хороший результат»*. This submission is largely that: a negative with a measured mechanism, one lever
that works on the axis it works on, and an account of what was tried and failed.

## The eight documents

| file | answers |
|---|---|
| **`METHOD.md`** | *«описание финальной архитектуры…»* — the architecture, the training recipe, the measurement behind each choice, and how the released artifact differs from the recommendation |
| **`RESULTS.md`** | headline figures, all twelve interventions with their effects, and depth behaviour |
| **`EXPERIMENTS.md`** | *«подробное описание всех экспериментов»* — every trained arm, generated from artifacts, with a mechanical coverage check |
| **`SCALE.md`** | *«почему ваш метод будет работать хорошо и на большем скейле»* — the scale argument and its one weak joint |
| **`NEGATIVE_RESULTS.md`** | every method tested to destruction, with the mechanism for each failure |
| **`EARLY_EXIT.md`** | *«опционально реализовать ранний выход из лупа»* — it was implemented and measured; the demand is real, eight rules across five classes cannot reach it, and the reason is measured |
| **`FAILURES.md`** | criterion 2: every error that reached a number, how it was caught, what it cost |
| **`LIMITATIONS.md`** | **what we do not have** — measurements never made, comparisons never run, choices never screened; plus **§6b: the numbers that will *not* reproduce from the shipped artifacts and why**, and which checkpoint carries which claim |

`../report.md` remains the full evidence base; every document here cites into it by section.

## Reading notes

*Housekeeping, placed here rather than above it.*

### Counting the parameters: the obvious way gives the wrong answer

Summing the checkpoint's `state_dict` returns **10,899,616**, which is over the cap. That is an
artifact of weight tying: `lm_head` and `embed` are the same `nn.Parameter` under two names, so
`state_dict()` counts the tied embedding twice while `.parameters()` de-duplicates. The difference is
exactly `vocab × hidden = 4096 × 448 = 1,835,008`, and `10,899,616 − 1,835,008 = 9,064,608`. Verify
with `sum(p.numel() for p in model.parameters())`, which is what every number here uses. A grader
checking the brief's hardest constraint the obvious way would otherwise conclude the model is
disqualified (`report.md` §6.0 row 27). A bare `Config()` prints **9,065,056**, because its default
`state_renorm=True` is the arm this report rejects; the shipped model sets `state_renorm=False`.

### How interventions are counted, so every document agrees

**Twelve interventions: eleven mechanisms on the model, one lever on the loss schedule.** Three of the
eleven ran at two settings each (LoRA at rank 2 and rank ≥ 4; duo-causal attention at W = 2 and W = 3;
the per-token depth gate unnormalised and scale-invariant), so `RESULTS.md` §2 has more rows than
there are mechanisms. The unnormalised depth gate is reported as an instrument failure rather than a
result, since it saturates to a hard argmax and cannot express a mixture at all, which is why it is a
row but not a claim. Where a count appears in this folder it is this one.

### Three terms, defined once

- **useful band**: the contiguous run of loop counts whose validation CE is within **0.01 nats** of
  that model's best. Computed by `src/plateau.py`; the report also calls the measured object a
  *plateau*. Every band figure is a `tol = 0.01` statement unless marked otherwise, and §4.25 sweeps
  that tolerance.
- **replicate floor**: run-to-run variation between same-config arms, measured per device and per
  configuration. **0.0150** (CUDA, dense supervision), **0.0541** (CUDA, terminal-only) and **0.0527**
  (MPS, n = 3; MPS run-to-run spread is 0.031–0.068). "Inside the floor" means smaller than that. All
  were measured at 2.5M tokens and are applied to 90M claims; `LIMITATIONS.md` §3 says so.
- **loop gain**: `CE@1 − CE_best` for one model, i.e. what the looping is worth to it. Distinct from
  **ΔCE_best**, which compares two models at their own optima.

### Comparability

Both caps are respected with margin. Perplexity is tokenizer-dependent (vocab 4096), so bits/byte is
the only externally comparable figure, and `SCALE.md` explains why even that is not like-for-like
against published numbers, which train on roughly 70× the tokens.

## Verifying this yourself

The released checkpoint ships with **the vocabulary that produced it** — a mismatch would not raise,
it would silently report CE ≈ ln(4096) = 8.32 and look like a broken model. So the gate judges against
*chance*, not a fixed tolerance:

```bash
python src/check_tokenizer_identity.py checkpoints/full_control90_kaggle --expect-ce1 3.9622
# [gate] vocabulary identity : PASS (|diff|=0.0020 vs |CE-chance|/3=1.4512)
```

*(`FAILURES.md` quotes |diff| 0.045 / 0.043 for the same gate — those are the two **remotely-trained
Kaggle** checkpoints it was written to check. 0.0020 is the **shipped** checkpoint. Same gate, three
different artifacts; all three PASS against a chance threshold of ≈1.45.)*

```bash
python src/test_model.py       # 13 correctness checks, incl. the block vs the real Qwen3 reference (2.4e-07)
python src/test_plateau.py     # 8 checks on the depth statistic, incl. a deliberate falsification probe
python src/headline.py check   # verifies every headline number still matches the artifact it came from
python src/make_inventory.py --check   # verifies the experiment inventory against the stored JSON
python src/check_caveats.py --strict   # every deflated claim carries its caveat in every file stating it
python src/check_crossref.py --strict  # every figure quoted in submission/ appears in report.md
```

**Do not run `src/train_tokenizer.py` before evaluating a released checkpoint** — it overwrites
`configs/tokenizer.json`, and every downloaded checkpoint would then evaluate at chance. This trap was
live in this repository until it was caught; see `FAILURES.md`.

## What is not here, and one disclosure

- **`../report.md` §1, the idea narrative, is written by the coding agent from the project's dated
  record, at the author's instruction**, and says so in a banner at its own head. The task grades idea
  generation separately and warns against LLM-sourced ideation, so it is stated here rather than left
  to be inferred: §1 is an account of what happened, not a claim of authorship over the ideas. Where
  an idea came from an external reviewer, §4.18 and §4.22 name it in place.
- Absolute perplexity is not competitive with data-unconstrained work; `SCALE.md` gives the token
  arithmetic rather than leaving it to be discovered.
- Several results are single-seed, and each says so where it appears.
