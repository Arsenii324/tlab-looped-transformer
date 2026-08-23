# A looped transformer on FineWeb — submission

| | |
|---|---|
| **validation perplexity** | **38.86** (CE 3.6599) · **bits/byte 1.5829** |
| **useful band** | **loops 6–17** — the run of loop counts whose validation CE is within 0.01 nats of this model's best, on a dense every-integer 1..64 sweep |
| parameters | **9,064,608** (cap 10M) † |
| training tokens | **89,999,360** (cap 100M) |
| released weights | **[huggingface.co/Arsen4ikVar/tlab-looped-transformer](https://huggingface.co/Arsen4ikVar/tlab-looped-transformer)** — `model.pt`, its tokenizer, and the architecture file |
| architecture | Qwen3-style 3-layer block, weight-tied, applied `r` times. No prelude, no coda, no inter-loop norm, additive re-injection, learned `h₀` |

*† Count with `sum(p.numel() for p in model.parameters())`. A `state_dict` sum returns 10,899,616 and
looks like a cap violation — see "Counting the parameters" below.*

## The result

A looped transformer reuses one block many times, buying compute without buying parameters. The field
runs few loops because the gains saturate early, and attributes that to the loop map converging — past
the fixed point there is nothing left to compute. **Here the map never converges and the gains
saturate anyway, and the reason is something else: a weight-tied loop visits depths it cannot tell
apart.** The 32 key vectors a token produces, one per
loop, span an effective rank of **~1.6 out of 32** — so attention or selection over them is close to
an average with extra steps, and every mechanism that needs to distinguish depths has nothing to work
with. We built the experiment that could have refuted that explanation, registered its criterion
first, and it held at two seeds.

That is the measured reason behind the dissociation the brief's question runs into: **twelve
interventions, five lower the loss, and not one widens the range of loops that stay useful.**

## What worked

Stated before the failures, because the page after this is mostly failures.

- **Removing inter-loop normalisation: ≈ −0.68 nats**, the largest single effect in the project. The
  field's default is wrong for this architecture — normalised, the map contracts to a fixed point by
  loop ~16 and never accrues loop gain at all.
- **Supervision annealing widens the useful band at 6 of 6 arms, for zero added parameters.** The
  best-supported result here, and the only one tested across two budgets *and* a tolerance sweep. **It
  does not lower the loss.**
- **Finishing the token budget: 0.39–0.42 nats** from 46M → 90M, against 0.002–0.26 for every
  architectural intervention here. **The data is worth more than any of them.**
- **Exclusive self attention: −0.2162 / −0.2633** at two seeds, zero parameters. It replicates — but
  84–91% of it lands at a single loop, so it improves the block rather than the looping, and it is
  untested at scale.

## The answer to the brief, in six parts

1. **The brief's own diagnosis does not hold here.** It attributes saturation to DEQ-style
   convergence. This model's state never converges: the unit drifts *logarithmically* (R² **0.986** on
   one parameter against a convergent power law's **0.748**; ρ > 1 at every measured depth; 0.18 rad
   still accumulating between loops 129 and 384). It saturates at loop ~8 regardless. The drift is
   architectural, since an untrained model of the same shape drifts *faster*.

2. **Why the depths are indistinguishable, and why it is not a small-model artifact.** The cause is
   projection asymmetry. An unshared stack builds a near-orthogonal key set out of a state stream that
   is *just as collinear* as the tied one (**4.36** vs **1.40** of 33) purely because each layer owns
   its own `W_K`; one shared projection cannot buy that at any width `[RANK-PROJECTION]`. The collapse
   is present at initialisation, worse after training, and **flat across hidden 224–896** — 2.73M to
   32.58M parameters, spread 0.062, no trend (§4.31). It accounts for the published positives in this
   family, which are measured on unshared stacks, rather than contradicting them. *One arm tried to buy
   the rank back: four distinct projections give **8.818/32** at initialisation and **1.74** trained,
   so training reproduced the collapse. That arm is n = 1 and it **failed its own registered gate**, so
   the causal question is undecided and this is supporting detail, not a load-bearing claim (§4.30).*

3. **The strongest idea we tested was per-token depth mixing, and we built the experiment that could
   have refuted our own explanation of why it fails.** The idea: different tokens need different
   amounts of computation, so a model should learn to weight or select among the states its loops
   produce. Our earlier gate could not test it — it collapsed onto a single loop and mixed nothing.
   The replacement is scale-invariant, and it genuinely spreads its weight: **given 8 loops it
   effectively uses 7.58 of them; given 16, 14.96; given 32, 29.84**, and no token puts more than 0.99
   of its weight on any one loop. **The criterion was registered before the arm existed:** *if it
   mixes and still gains, our explanation is wrong.* It mixed. It gained **−0.0012 / +0.0023** across
   two seeds — nothing. A mixture that works, over depths that are not distinguishable, buys nothing.
   Depth mixing was attacked seven ways in all; every one is null, an instrument failure, or a gain
   whose own mechanism check failed (`RESULTS.md` §1b).

4. **Twelve interventions. Five lower the loss. Not one widens the useful band**, at any tolerance
   tested. Four of the five put **67–101%** of their gain at a single loop where their own mechanism is
   provably inert: LoRA 67–95% `[POSTHOC-LORA-RANK]` `[CAPACITY-NOT-DIVERSITY]`, XSA 84–91%
   `[XSA-AT-R1]`, duo-causal W = 3 78–101%, the saturating depth gate 96%. They improve the block, not
   the looping. The fifth inverts that: the norm penalty wins perplexity (**37.52** vs **38.86**) by
   damaging loop 1 (`ΔCE@1 = +0.2263`). And the one that replicated across three platforms vanishes at
   5× the budget, **−0.0936** at 2.5M against **+0.0077** at 12M in a config-identical pair. **This
   project has no replicated CE improvement at scale.** (§4.29; `RESULTS.md` §1b for what survives.)

5. **One lever moves depth, costs zero parameters, and survives the two tests the others failed.**
   Supervision annealing widens the band at **6 of 6 seeds**, with the same edge decomposition at 2.5M
   and at 10M. On an every-integer sweep of depths 12–32 the annealed arms hold within tolerance over
   **2.1× and 2.0×** as many depths as their controls. It is the only band claim here that is robust to
   the plateau tolerance, resolved to ±1 loop, and replicated across seeds. **It does not lower the
   loss**; that half of the claim is withdrawn. (`METHOD.md` §2 for the withdrawal and its six points;
   §4.25c for the sweep.)

6. **Per-token depth demand is real and unreachable — the subject of `EARLY_EXIT.md`.** The evidence
   for "real" is the split-half reliability, **0.866** against a null of **0.0007** — not the
   0.3084-nat oracle headroom, whose two nulls are mis-specified: they destroy the per-token curves'
   smoothness (4.6× rougher) and produce *more* headroom than the real data (**0.3877**, **0.4110**).
   Eight rules across five instrument classes capture at most **0.1%** of it, for the reason in
   part 2.

A negative result with a measured mechanism, one lever that works on the axis it works on, and an
account of what was tried and failed.

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

**Русская версия:** [`ru/`](ru/) — **машинный перевод** восьми из девяти документов
(`EXPERIMENTS.md` не переведён: его таблица генерируется из артефактов). **Авторитетной версией
являются английские документы в этой папке**; при расхождении верна английская. Все числовые
значения сверены: 1 255 чисел, множества совпадают.

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

### The bracketed tokens

`[POSTHOC-LORA-RANK]`, `[CAPACITY-NOT-DIVERSITY]`, `[XSA-AT-R1]`, `[WITHDRAWN-ANNEAL-CE]` and
`[RANK-PROJECTION]` are not unfinished markup. Each marks a claim that was **deflated or withdrawn**,
and `src/check_caveats.py --strict` fails the build if any document states one of those claims'
numbers without carrying its token — so a caveat cannot survive in one file and be lost in another.
That check exists because exactly that happened three times in one evening (`FAILURES.md`).

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

> **Start here: the weights are not in this repository.** `.pt` files are not tracked — a clone
> carries the code, the tokenizer and the eval JSONs, but no checkpoint. **Download the released
> model from Hugging Face first**, then point the commands below at it:
>
> ```bash
> hf download Arsen4ikVar/tlab-looped-transformer --local-dir /tmp/tlab
> #   -> model.pt, tokenizer.json, model.py, README.md
> python src/check_tokenizer_identity.py /tmp/tlab/model.pt --expect-ce1 3.9622 \
>        --tok /tmp/tlab/tokenizer.json
> ```
>
> **To train rather than evaluate**, no download is needed — the data and tokenizer are built from a
> stream:
>
> ```bash
> python src/data.py            # streams + packs FineWeb shards using the SHIPPED tokenizer
> python src/train.py           # trains the released configuration
> python src/eval.py checkpoints/<name> --max-loops 64
> ```
>
> **Verify against the *downloaded* copy, not this repo's.** A broken upload is invisible to a
> check run against the local files.

The released checkpoint ships with **the vocabulary that produced it** — a mismatch would not raise,
it would silently report CE ≈ ln(4096) = 8.32 and look like a broken model. So the gate judges against
*chance*, not a fixed tolerance:

```
[gate] vocabulary identity : PASS (|diff|=0.0020 vs |CE-chance|/3=1.4512)
```

```bash
python src/test_model.py       # 13 correctness checks, incl. the block vs the real Qwen3 reference (2.4e-07)
python src/test_plateau.py     # 8 checks on the depth statistic, incl. a deliberate falsification probe
python src/headline.py check   # verifies every headline number still matches the artifact it came from
python src/make_inventory.py --check   # verifies the experiment inventory against the stored JSON
python src/check_caveats.py --strict   # every deflated claim carries its caveat in every file stating it
python src/check_crossref.py --strict  # no figure in this folder is absent from ../report.md
```

**Do not run `src/train_tokenizer.py` before evaluating a released checkpoint** — it overwrites
`configs/tokenizer.json`, and every downloaded checkpoint would then evaluate at chance. This trap was
live in this repository until it was caught; see `FAILURES.md`.

## What is not here

- Absolute perplexity is not competitive with data-unconstrained work; `SCALE.md` gives the token
  arithmetic rather than leaving it to be discovered.
- Several results are single-seed, and each says so where it appears.
