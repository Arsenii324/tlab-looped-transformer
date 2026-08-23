# A looped transformer on FineWeb — submission

**Start here.** `../report.md` is the complete record (6,600+ lines) and is *evidence*, not reading
material. This folder is the readable submission: eight documents plus this index, each answering one
thing the task asks for, each linking into the report for the measurement behind it.

---

## The artifact, in one table

| | |
|---|---|
| parameters | **9,064,608** (cap 10M) — 79.7% in the reused block, 20.2% in the tied vocabulary. **Counting `state_dict` instead gives 10,899,616 and looks like a cap violation — see the note below.** |
| training tokens | **89,999,360** (cap 100M) |
| validation perplexity | **38.86** (CE 3.6599) |
| bits/byte | **1.5829** |
| useful-depth band | **loops 6–17** (dense every-integer 1..64 sweep) |
| architecture | Qwen3-style 3-layer block, weight-tied, applied `r` times. No prelude, no coda, no inter-loop norm, additive re-injection, learned `h₀` |

> **⚠ How to count the parameters, because the obvious way gives the wrong answer.** Summing the
> checkpoint's `state_dict` returns **10,899,616** — *over the cap*. That is a counting artifact of
> **weight tying, which is this architecture's central feature**: `lm_head` and `embed` are the same
> `nn.Parameter` registered under two names, so `state_dict()` counts the tied embedding twice while
> `.parameters()` de-duplicates. The difference is exactly `vocab × hidden = 4096 × 448 =
> **1,835,008**`, and `10,899,616 − 1,835,008 = 9,064,608`. **Verify with
> `sum(p.numel() for p in model.parameters())`**, which is what every number in this submission uses.
> *This is stated here rather than in the report's failure log alone because a grader checking the
> hardest constraint in the brief the obvious way would conclude the model is disqualified.*
> (`report.md` §6.0 row 27 — it was caught before the model card shipped with the wrong number.)

Both caps are respected with margin. **Perplexity is tokenizer-dependent** (vocab 4096), so
bits/byte is the only externally comparable figure — and see `SCALE.md` for why even that is not a
like-for-like comparison against published numbers.

## How interventions are counted, once, so every document agrees

**Twelve interventions: eleven mechanisms on the model, one lever on the loss schedule.** **Three** of
the eleven ran at two settings each — LoRA at rank 2 and rank ≥ 4; duo-causal attention at W = 2 and
W = 3; the per-token depth gate unnormalised and scale-invariant — so `RESULTS.md` §2 has more *rows*
than there are mechanisms. **The unnormalised depth gate is reported as an instrument failure rather
than a result** (it saturates to a hard argmax and cannot express a mixture at all), which is why it
is a row but not a claim. Where a count appears in this folder it is this one.

## The answer to the brief, in five sentences

The task asks for low perplexity **by exploiting many loops**. We report a **dissociation**: those are
two goals and this architecture does not deliver them together.

1. **The trajectory never converges** — the unit state drifts *logarithmically* (R² 0.986 on a
   one-parameter model against 0.748 for a convergent power law) — **and it saturates at loop ~8
   anyway.** That is *saturation without convergence*, and it contradicts the premise the brief
   itself advances (that fast convergence is what makes further compute pointless).
2. **Twelve interventions. Five lower the loss. Not one widens the useful band** — and none moves it
   *consistently*: at `tol = 0.01` three of the five narrow it, but that direction does not survive
   halving the plateau tolerance (§4.25), so it is reported as an observation rather than a finding.
   **That "five" is not a softening of an earlier "two" — it is the dissociation appearing five times
   independently, in two opposite directions.** Four interventions help at loop 1 and the help does
   not propagate into depth; one *hurts* at loop 1 and the loop-gain statistic rewards it for that.
   **Both say the same thing: the loop's own contribution is untouched by anything that moves the
   loss.**
3. **Four of those five put 67–101% of their gain at a *single* loop, where their own mechanism is
   provably inert or irrelevant** — loop-cycled LoRA 67–95% `[POSTHOC-LORA-RANK]` `[CAPACITY-NOT-DIVERSITY]`, exclusive self attention
   84–91% `[XSA-AT-R1]`, duo-causal attention at W = 3 78–101%, the learned depth gate 96%. **They improve the
   block, not the looping.** The fifth is the mirror image: the norm penalty wins perplexity (37.52 vs
   38.86) by *damaging* loop 1 — 88% of its loop-gain advantage is `ΔCE@1 = +0.2263`.
4. **One lever does move the useful band** — *where the loss is applied* (supervision annealing),
   at **5/5 seeds**, **zero added parameters**, and with the *same* edge decomposition at 2.5M and at
   10M tokens (onset 8 → 8, end 16 → 24) — but its effect on the *ceiling* was withdrawn at n=4 by a
   criterion registered before the data existed, and a fifth point at 4× the budget is the worst yet
   (**+0.1119**) `[WITHDRAWN-ANNEAL-CE]`.
5. **Per-token depth demand is real — and the evidence is the split-half reliability (0.866 against a
   null of 0.0007), not the oracle headroom.** *The 0.3084-nat headroom cannot carry the claim by
   itself: both nulls built to bound it are **mis-specified**, destroying the per-token curves'
   smoothness (4.6× rougher) and so producing **more** headroom than the real data — 0.3877 and
   0.4110. What survives a null is that each token's preferred depth is **reproducible**, and that is
   what an exit rule would need. `EARLY_EXIT.md` §1 and §4.7 state this in place.* The demand is real
   **and unreachable by eight rules across five instrument classes — the best captures 0.1%.** We can now say why: **a token's 32 depth keys span
   an effective rank of ~1.6.** There is almost nothing for a mixing or selection mechanism to
   discriminate between. **The one test that could have refuted that explanation was run and did not:**
   a *scale-invariant* depth gate (a different arm from the saturating one in sentence 3) that
   demonstrably mixes — 7.6/8, 15.0/16, 29.8/32 effective loops — returns **−0.0012 / +0.0023** at two
   seeds. A working mixture over a collapsed representation buys nothing `[RANK-PROJECTION]`.
   **`EARLY_EXIT.md` is the whole case.**

The brief states that *«отсутствие положительного результата при хорошем анализе всех негативных —
хороший результат»*. This submission is largely that: a negative with a measured mechanism, one
lever that works on the axis it works on, and an explicit account of what was tried and failed.

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
| **`LIMITATIONS.md`** | **what we do not have** — measurements never made, comparisons never run, choices never screened, and the apparatus stated plainly enough to price the rest |

`../report.md` remains the full evidence base; every document here cites into it by section.

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

- **`../report.md` §1 — the idea narrative — is written by the coding agent, not by the author**, from
  the project's own dated record, at the author's instruction. It says so in a banner at its own head.
  The task grades idea generation separately and warns against LLM-sourced ideation, so this is stated
  here rather than left to be inferred: **§1 is an account of what happened, not a claim of authorship
  over the ideas.** Where an idea came from an external reviewer, §4.18 and §4.22 name it in place.
- Absolute perplexity is not competitive with data-unconstrained work; `SCALE.md` gives the token
  arithmetic rather than leaving it to be discovered.
- Several results are single-seed, and each says so where it appears.
