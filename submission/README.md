# A looped transformer on FineWeb — submission

**Start here.** `../report.md` is the complete record (6,400+ lines) and is *evidence*, not reading
material. This folder is the readable submission: five documents, each answering one thing the task
asks for, each linking into the report for the measurement behind it.

---

## The artifact, in one table

| | |
|---|---|
| parameters | **9,064,608** (cap 10M) — 79.7% in the reused block, 20.2% in the tied vocabulary |
| training tokens | **89,999,360** (cap 100M) |
| validation perplexity | **38.86** (CE 3.6599) |
| bits/byte | **1.5829** |
| useful-depth band | **loops 6–17** (dense every-integer 1..64 sweep) |
| architecture | Qwen3-style 3-layer block, weight-tied, applied `r` times. No prelude, no coda, no inter-loop norm, additive re-injection, learned `h₀` |

Both caps are respected with margin. **Perplexity is tokenizer-dependent** (vocab 4096), so
bits/byte is the only externally comparable figure — and see `SCALE.md` for why even that is not a
like-for-like comparison against published numbers.

## The answer to the brief, in five sentences

The task asks for low perplexity **by exploiting many loops**. We report a **dissociation**: those are
two goals and this architecture does not deliver them together.

1. **The trajectory never converges** — the unit state drifts *logarithmically* (R² 0.986 on a
   one-parameter model against 0.748 for a convergent power law) — **and it saturates at loop ~8
   anyway.** That is *saturation without convergence*, and it contradicts the premise the brief
   itself advances (that fast convergence is what makes further compute pointless).
2. **Nine interventions on the dynamics; one lowers the loss; none widens the useful band.**
3. The one that lowers it (loop-cycled LoRA, −0.086 replicated across three platforms) delivers
   **~90% of its gain at loop 1**, where its mechanism is provably inert — **it improves the block,
   not the looping.**
4. **One lever does move the useful band** — *where the loss is applied* (supervision annealing),
   at **4/4 seeds** and **zero added parameters** — but its effect on the *ceiling* was withdrawn at
   n=4 by a criterion registered before the data existed.
5. **Per-token depth demand is real** (oracle headroom 0.20 nats, split-half reliability 0.866
   against a null of 0.0007) **and unreachable** — and we can now say why: **a token's 32 depth keys
   span an effective rank of ~1.6.** There is almost nothing for a mixing or selection mechanism to
   discriminate between.

The brief states that *«отсутствие положительного результата при хорошем анализе всех негативных —
хороший результат»*. This submission is largely that: a negative with a measured mechanism, one
lever that works on the axis it works on, and an explicit account of what was tried and failed.

## The six documents

| file | answers |
|---|---|
| **`METHOD.md`** | *«описание финальной архитектуры…»* — **PENDING**: six arms land between 20:00 and 21:45 that bear directly on it (capacity-vs-diversity, the scale-invariant gate, the recommended config's own weights). Writing it now would mean rewriting it. Until it exists, `../report.md` §3.5 is the method of record |
| **`RESULTS.md`** | **PENDING**, same reason. Headline numbers are in the table above and in `../report.md` §.headline |
| **`EXPERIMENTS.md`** | *«подробное описание всех экспериментов»* — **all 113 trained arms**, generated from artifacts, with a mechanical coverage check |
| **`SCALE.md`** | *«почему ваш метод будет работать хорошо и на большем скейле»* — the scale argument and its one weak joint |
| **`NEGATIVE_RESULTS.md`** | every method tested to destruction, with the mechanism for each failure |
| **`FAILURES.md`** | criterion 2: every error that reached a number, how it was caught, what it cost |

`../report.md` remains the full evidence base; every document here cites into it by section.

## Verifying this yourself

The released checkpoint ships with **the vocabulary that produced it** — a mismatch would not raise,
it would silently report CE ≈ ln(4096) = 8.32 and look like a broken model. So the gate judges against
*chance*, not a fixed tolerance:

```bash
python src/check_tokenizer_identity.py checkpoints/full_control90_kaggle --expect-ce1 3.9622
# [gate] vocabulary identity : PASS (|diff|=0.0020 vs |CE-chance|/3=1.4512)
```

```bash
python src/test_model.py       # 13 correctness checks, incl. the block vs the real Qwen3 reference (2.4e-07)
python src/test_plateau.py     # 8 checks on the depth statistic, incl. a deliberate falsification probe
python src/headline.py check   # verifies every headline number still matches the artifact it came from
python src/make_inventory.py --check   # verifies the experiment inventory against the stored JSON
```

**Do not run `src/train_tokenizer.py` before evaluating a released checkpoint** — it overwrites
`configs/tokenizer.json`, and every downloaded checkpoint would then evaluate at chance. This trap was
live in this repository until it was caught; see `FAILURES.md`.

## What is not here

- **Section 1 of `../report.md` — the idea narrative — is the author's and is written separately.**
- Absolute perplexity is not competitive with data-unconstrained work; `SCALE.md` gives the token
  arithmetic rather than leaving it to be discovered.
- Several results are single-seed, and each says so where it appears.
