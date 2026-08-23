# §1 raw material — the dated reversals, assembled. NOT a draft of §1.

**This is factual record, not idea generation.** §1 is graded on how *you* arrived at the approach and
the task explicitly warns against LLM-sourced ideation, so nothing here is phrased as narrative and
none of it should be pasted. It exists because the reversals are scattered across `LOG.md` and §6.0's
34 rows, and having them in one place is the difference between a 30-minute §1 and a 3-hour one.

Each row is a **belief that was held, acted on, and then measured false** — with what it cost.

| # | what was expected | what was measured | where |
|---|---|---|---|
| 1 | Contraction would help — the field's stability framing, and this project's own design prior | **No contraction at any depth.** Removing the stabiliser was worth **0.744 nats**, the largest single effect in the project | §4.1, §4.3 |
| 2 | The compute-matched untied baseline was untrainable (NaN at every LR, six configs) | **The hardware was manufacturing the NaNs.** On CUDA it trains at all three LRs, including the one that died at step 13 | §4.4, §6.0 row 1 |
| 3 | `argmin` was the right statistic for a depth claim | **134 of 165 stored curves have argmin margins below the noise floor.** The replacement killed one finding before publication and revised another from 2× to 1.50× | §4.15, §6.0 row 3 |
| 4 | Supervision density was a dial to tune | **A threshold at k = 1.** k = 2, 3, 5, 8 all behave alike; gain drops 0.162 from k1→k2 then varies 0.018 | §4.16 |
| 5 | The angular budget measured useful computation | **An untrained model travels 4.5× further with zero capability.** The instrument's own null refuted the interpretation 90 minutes after it was written | §4.16c |
| 6 | Cross-layer collapse to one direction was a real degeneracy | **Largely a shared-residual artifact.** Contributions sit at cos ≈ 0.14–0.18, not 1.0. Two forward passes closed a whole architectural family | §4.20 |
| 7 | Supervision annealing improved the ceiling (n = 2) | **Withdrawn at n = 4** by a pre-registered criterion. Seed 2 reverses it. **The band widens at 4/4 seeds regardless** | §3.5, §4.17 |
| 8 | The `outputs:` fix protected future jobs (23 configs) | **It was a glob, and there is no globbing.** A job that finished all three arms returned `results.json` alone | §6.0 row 34 |
| 9 | Three targeted propagation passes had cleared the retractions | **12 defects on the first end-to-end read**, three serious, none reachable by grep | §6.0 row 33 |
| 10 | The learned depth gate would test per-token depth mixing | **It saturates to a hard argmax** — it cannot express a mixture at all | §4.22 |

**The shape that is already in the record, stated once so you can accept or reject it:** *the field's
framing was assumed and did not apply; several instruments did not measure what they were named after;
what survived is what was measured against a null.* Rows 3, 5, 6 and 10 are all the same failure —
**a statistic that was never asked what it samples** — and it is the failure this project is most
prone to.

**Two things worth knowing while writing it.** (a) The task rates a well-analysed negative as a good
result (*«отсутствие положительного результата при хорошем анализе всех негативных — хороший
результат»*), and §0's abstract now states the report's answer in that form. (b) Criterion 1 has no
other carrier — every other section is implementation and verification.
