# Reply — 2026-08-23 18:05 · a correction to reply 16, a first replicated positive, and a fifth instrument that doesn't work

Self-contained. Leading with the correction, because it lands on the sentence I told you was this
report's spine.

---

## 1. Reply 16 §2(c) is now false. "Seven interventions; zero raise the ceiling" has an exception.

I wrote: *"Seven interventions on the dynamics; zero raise the ceiling."* **One does.**

**Operator diversity** — the shared block given `B = 4` low-rank branch adapters cycled by loop index
(`branch = t mod 4`), so consecutive loops apply genuinely different operators while base weights stay
tied. Five in-job pairs, every one arm-and-control in the same job on the same shard and tokenizer:

| platform | seed | rank | ΔCE_best | control band → arm band |
|---|---|---|---|---|
| local MPS | 0 | **2** | **+0.0941** | [8,16] mid 11.3 → **[8,16] mid 11.3** |
| local MPS | 0 | 4 | −0.0514 | [8,16] mid 11.3 → **[8,16] mid 11.3** |
| DataSphere T4 | 0 | 4 | **−0.1011** | [8,20] mid 12.6 → **[8,20] mid 12.6** |
| Kaggle | 1 | 8 | **−0.0733** | [8,20] mid 12.6 → **[8,20] mid 12.6** |
| Kaggle, `sw90` | 0 | 4 | **−0.1172** | [8,24] mid 13.9 → **[8,24] mid 13.9** |

At rank ≥ 4: mean **−0.0857**, sd 0.0292, **95% t-interval [−0.1322, −0.0393] — excludes zero.** That
is the same paired t-test that *withdrew* the annealing CE claim at n=4, applied here and passing.
Two ranks, three platforms, two seeds, one of them under a different supervision schedule.
**It is the only CE claim in this project that survives multi-platform replication.**

**And the useful band is identical to its own control in all five pairs — not one grid point.**
Including in the two pairs where CE improves by more than 0.10 nats. So the corrected sentence is
better than the one it replaces, and it is the one I would now put in the spine:

> **Eight interventions. One lowers the loss, replicated. None widens the useful band.**

Against the brief — *low perplexity **by exploiting many loops*** — that is the first clause delivered
and the second untouched, which is this report's central dissociation appearing on the one
intervention that reliably works.

**Three things that keep it honest and that I would not want you to hear from anyone else first:**

- **Rank 2 reverses the sign**, at +0.0941, which is above the MPS floor. The effect has a threshold
  between rank 2 and 4, and the cheapest version is actively harmful.
- **It costs parameters**, and §3.5's strongest asset was that its recommendation costs none:
  **+408,576 at rank 4 (+4.51%)**, **+817,152 at rank 8 (+9.01%)** — the rank-8 model is 9,881,760,
  **118,240 under the 10M cap**. It passes §3.4's letter (`t mod 4` is a function of t, defined for
  every t, extrapolates cleanly to loop 64) and arguably fails its spirit (the benefit is bounded by a
  fixed four-branch table however wide the block gets). Both readings are in §4.21; nothing here tests
  which is right at scale.
- **2.5M tokens, no full-budget replication.** This project's own regularity says a 2.5M effect can
  shrink or reverse by 90M — the norm penalty shrank 12× and flipped character over exactly that span.

**This also corrects an inference I made three hours earlier and told you about.** When `od_lora_r2`
came back at +0.0941 I wrote that it *"empirically confirms §4.20's retraction — the arm was built to
fix what turned out to be a shared-residual artifact."* **Wrong, and rank is why.** §4.20's retraction
stands exactly as stated (the cross-layer cosine *is* a shared-residual artifact, and operator
diversity genuinely does not move it — cos@64 = 1.0000, unchanged). What does not follow is that
branch diversity is therefore useless. **A retracted statistic was allowed to retire a whole design
axis on one under-powered arm.** They were never the same claim.

## 2. The learned depth gate — the fifth instrument class — does not test its own hypothesis

You asked for this one twice, and §4.7c's static-mixture null was explicitly its *lower* bound. The
pre-registered read (written into `RUNS.md` before any number existed) was: *do the gate weights
concentrate, or stay near uniform?*

**They concentrate completely.** On the local checkpoint, through the model's own loop:

| r | logit range per token | mean top weight (uniform) | **effective loops mixed** | argmax loop |
|---|---|---|---|---|
| 8 | 1,872.6 | 0.9975 (0.1250) | **1.01 / 8** | 7.40 |
| 16 | 3,445.6 | 0.9940 (0.0625) | **1.02 / 16** | 14.56 |
| 32 | 6,312.7 | 0.9873 (0.0312) | **1.05 / 32** | 28.59 |

A softmax over logits spanning thousands is a hard argmax. **It mixes 1.0 loops, not r**, and it
selects the deepest one — so the "mixture" evaluates to `readout(h_r)`, which is what the control
already computes.

**Why: the gate's logits are `w·h_t`, unnormalised, and ‖h‖ grows 1.8–4.0× within a forward pass and
~10³ over training.** The softmax temperature is effectively zero. **The readout is deliberately
scale-invariant (RMSNorm before the tied head); this gate is not.** A gate reading `norm1(h_t)` is the
two-line version that would test the hypothesis, and it was not run.

**So the honest status of your question is: the per-token depth headroom (0.2008–0.2032 nats,
split-half 0.866) is unreached by four instrument classes and UNTESTED by the fifth — not refuted by
it.** That is weaker than what I would have been entitled to claim, and it is the true statement.

**The decisive number needs no weights at all.** On the T4 the arm is −0.2950 against its in-job
control. **At `r = 1` the gate is provably inert** — a softmax over one state returns that state — and
**96% of the −0.2950 is already there** (ΔCE@1 = −0.2830, Δgain = +0.0121, inside every floor). The
mixing is not what produced the improvement. The same holds for every LoRA arm at 88–95%.

**Two further things I would rather state than have found:** the two replicates of this arm disagree
by **0.32 nats** (T4 −0.2950; MPS +0.0241 at matched step, wrong sign) on configs verified identical
field-by-field with byte-identical gate code, and I cannot settle it because the DataSphere weights
were lost. And the gate retains all `r` states with gradient — **O(r) activation memory, which breaks
the constant-memory property of a weight-tied loop** and OOM'd the local arm at a deep loop draw. For
a +448-parameter mechanism, the parameter count is not where the cost lives.

## 3. `report.md` has now been read end to end — the first time — and it had twelve defects

This was the largest outstanding risk I named to you, and it was worth the two hours. **None of the
twelve was reachable by grep**, which is the transferable part: after each of today's three
retractions I propagated by searching for the withdrawn *number*. That misses claims restated in
*words*. Three were serious:

- **§3.5 restated *both* withdrawn claims** — "better CE than its control" and "band from ~23 loops to
  32–64" — **four lines below their own withdrawal blocks.**
- **§8 still carried `+0.022 nats`**, the *cross-job* control that §4.17 had explicitly replaced with
  an in-job one **reversing the sign to −0.0264**.
- **§4.2's summary still said loop gain was "flat"**, 120 lines after its own paired re-measurement
  overturned exactly that (+0.0130, CI excludes zero).

Nine more, including a false `iff` on `σ_max` that contradicted both §2 and its own correction block,
the retracted unseeded ρ = 1.70 quoted as live ten lines after being retracted, and a plateau quoted
with no grid in the §3.5 table — violating this report's own rule. All fixed, all recorded as §6.0
row 33. **A retraction needs a prose pass, not a number pass.**

## 4. The fix for the lost-checkpoints defect never worked, and I diagnosed that wrong too

~20 DataSphere jobs lost their weights because `outputs:` listed only `results.json`. The remedy was
applied to 23 configs — **as a glob, `"*_last.pt"`.** A job then completed all three arms, wrote all
three files, and returned **`results.json` alone, 11.5 KB.**

I first reported the mechanism as submit-time path resolution. **That was wrong**, and an outside
check found the answer in `log.txt` — a file I never opened, having read its two siblings:

    [ERROR] - Some output files were not uploaded due to errors:
    [ERROR] -   * *_last.pt (Error while processing file)

Outputs skip the existence check entirely, so a non-resolving path passes submission silently and
fails server-side at upload. There is no globbing. **22 of 26 configs carry that glob**, so the
protection believed to cover every future job covered none — and `outputs: [results/**]`, the form
inherited from the workspace's own DataSphere notes, has failed every time it was used here.

**A second correction, and it inverts a rule I gave you:** I said a DataSphere `ERROR` status is
usually cosmetic and to read stdout before believing it. **This ERROR was not cosmetic — it was caused
by the failed upload.** Two different failures were sharing one rule. The discriminator is
`grep "Error while processing file" log.txt`.

**Both of these are the same shape and it is now a named pattern: the dangerous state is not
"unfixed", it is "recorded as fixed".** Each remedy was applied, written into two documents, and never
checked against the artifact it was supposed to produce.

## 5. What I am *not* claiming, from a results table I was handed

A collaborating agent supplied a cross-platform table and four headline conclusions. **I re-derived
all eight completed arms from the raw `val_curve` and its numbers are accurate** — I reproduce every
CE@1, best CE, best loop and Δ. Its conclusions I do not adopt:

- *"Operator diversity consistently outperforms control across all ranks"* — **false by its own
  table**; rank 2 is +0.0941, worse.
- *"State depth gating completely eliminates late-loop degradation (0.0070 nats from r=8 to r=64)"* —
  arithmetically true and mechanistically empty: a hard selector's curve is flat across `r` by
  construction. Its plateau **[8,64] mid 22.6** is over *mixture-window size, not depth* — pre-
  registered as excluded from the band tables before the run, and this conclusion puts it in them.
- The table **omits the local depth-gate arm entirely** — the one arm that contradicts it.

## 6. Status

| stream | state |
|---|---|
| DS `tlab-operator-diversity` (`bt1sglqurmj6frrmsfrk`) | **DONE, harvested.** All 3 arms. ERROR status was *real* but confined to output upload; the experiment completed |
| local `od_depth_gate` | **INCOMPLETE at step 760/1219**, OOM on resume (O(r) state retention). Compared at matched steps instead; no new compute needed |
| Kaggle `kg_rank8`, `kg_sw90` | **DONE, harvested, verified** |
| everything else | harvested |

**Still the author's, not mine:** §1 (empty), D3 (which checkpoint ships — my recommendation remains
the control), rotating the wandb key, and any push or upload. **No git remote is configured and no HF
upload has ever run.**
