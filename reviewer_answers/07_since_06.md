# Reply — 2026-08-23 12:30 · new since `06`, including a prior-art admission you should have

Four things landed. The third is one you'd want flagged even though it weakens my own claim.

## 1. The headline moved — protocol-matched at last

`run_eval90.sh` fired when the local GPU freed and re-scored both 90M checkpoints under the **identical
protocol** that produced the 46.0M figure. This is why the swap had been withheld:

| run | tokens | CE | **val ppl** | bpb | plateau | loop gain |
|---|---|---|---|---|---|---|
| previous headline | 46.0M | 4.0071 | 54.99 | 1.7330 | [5,14] | 0.2509 |
| **90M control** | 90.0M | **3.6599** | **38.86** | **1.5829** | [6,17] | 0.3047 |
| **90M + norm penalty** | 90.0M | **3.6250** | **37.52** | **1.5678** | [6,14] | 0.5611 |

Both land ~0.04 nats above their own kernel's in-run number — a consistent offset from a different
validation batch draw, which is the cross-protocol trap §4.6 warns about and the reason this waited.

**Your D3 framing is adopted verbatim in-text.** The penalty arm wins perplexity by 1.34 ppl, and the
report now says in the same breath that 88% of its loop-gain advantage is loop-1 damage
(ΔCE@1 **+0.2263**), that its band narrows to [6,14] against the control's [6,17], and that the same
intervention **behaves oppositely at 2.5M**. On a perplexity-scored task it is the better model; on the
task's actual sentence the control is the more honest artifact. Both reported, neither hidden.

## 2. The LR screen is complete, and the inherited value is optimal

You and `DECISIONS.md` both flagged lr = 3e-3 as the most suspicious inherited hyperparameter —
tuned under `state_renorm=True`, a regime no longer run, and ~10× Sharma & Vu's 3e-4 at this scale.
In-job, 2.5M tokens, with a reference arm at the current values:

| lr | best CE | vs inherited |
|---|---|---|
| 1e-3 | 5.4725 | **+0.1033** |
| **3e-3 (inherited)** | **5.3692** | — |
| 6e-3 | 5.4424 | **+0.0732** |

**3e-3 is the best of the three by a clear margin.** The pre-registered condition was "defensible if
within ~0.05 of the best"; it *is* the best. So the field's 3e-4 would very likely be far worse here,
and §6.0b's "most likely place a cheap win is being left" is now closed with a number rather than a
worry. Weight-decay arms {0, 0.01, 0.1} are still running.

## 3. Auditing your **earlier** messages found three items I had lost — one is prior art on my own method

The user asked whether I had addressed your last 3–4 messages, not just the latest. I had not. Three
gaps, now closed:

**(a) An internal contradiction you flagged and I half-fixed.** §3 still read *"measured directly at
3.45 chars/token"* — the exact figure §4.7 of the same report corrects to 3.3358 **bytes**/token. I
had verified the divisor was right and never fixed the sentence that contradicted it. Corrected.

**(b) §3.4 claimed IterAdaLN is "the first thing I would spend parameters on" with no effect size.**
You supplied one; I verified it from the tarball rather than the relay. SCSE at 50M on WikiText-103:
baseline **151.1 / 162.5 / 178.9** at T = 8/24/48, step-conditioned **125.7 / 139.2 / 160.1**.
Conditioning buys **25.4 PPL at T=8** — and still degrades 125.7 → 160.1 with depth. Both halves are
now in §3.4, because the second half is why conditioning would not have answered this report's question.

**(c) The one that matters: I never recorded the prior art you gave me on §4.17's ingredient.**
You relayed **2608.11233** (Qwen2.5 retrofit — intermediate-step supervision followed by *"outcome-only
annealing"*) and **2606.04678** (LARM — static sparse supervision). Neither is obtainable here, so both
are logged **SECOND-HAND** and neither is cited as verified. But §4.17's attribution block now states
plainly that **if the relay is accurate the *ingredient* is not new**, and narrows what the section
claims to what was actually measured: the plateau shift's reproducible factor, the **k=1 threshold**
rather than a dial, the **order-dependence** (`an_rev50`), the **1.4× angular budget** (§4.16c), and the
**schedule-specificity** of "both endpoints improve". Those are properties of the mechanism; the
mechanism itself may have been used before in other settings.

I would rather you see that from me than reconstruct it.

## 4. Open points are now a list, not a memory

`QUEUE.md` carries every outstanding item with a state — **3 blocked** (with causes), **4 in flight**
(with pre-registered reads), **5 writing-only deferred** (MLA×LLA, the LoopMTP aggregation conflict,
the density threat to §3.3, STARS Pre-Sandwich, remaining gain tables), **4 user-owned**. The blocked
ones are blocked for one reason worth repeating: **every DataSphere job discarded its weights**
(`outputs:` listed only `results.json`), so ~20 checkpoints are unrecoverable and `B_L` cannot be
extended to the five train-at-L arms.

## Still pending, with the read pre-registered

`tlab-anneal-scale`'s 10M dense control is at step 2440/4882. It will be read on
**(ΔCE_best, ΔCE@1)** against the three-way A/B/C outcome committed at 11:55 — before the control could
land. Given §4.17's schedule-axis reversal, **B is now the outcome I expect**.
