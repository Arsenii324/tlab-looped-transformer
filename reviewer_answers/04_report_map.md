# Report map — one line per claim, with status

Written 2026-08-23 11:25 to fix an asymmetry you named: you hold `report.md` at 1,521 lines
(22 Aug 21:40); it is now **3,397**. Roughly half is invisible to you, including everything that
became the method. **NEW** marks sections written after your copy.

Status: **HOLDS** (replicated or wide margin) · **NARROWED** (weaker than first stated) ·
**RETRACTED** · **PROVISIONAL** (a running job can still change it).

---

## §1 The idea — *empty, and staying empty*
Reserved for the author. Not written by me, and I decline to draft candidate voices for it; see the
note at the end.

## §2 Task and constraints
| claim | status |
|---|---|
| 2.0 Every clause of the task mapped to where it is answered | HOLDS |
| The three exemplars' loop counts are **engineering choices, not measured ceilings** — Ouro's R=4 reduced *from* 8 for stability, Huginn's "8–12" is the zero-shot figure (20 with 1 example, 32 with 25–50), Loopie's R=2 is FLOP allocation | HOLDS — dismantles the task's own premise from the citation side |
| **NEW** The DEQ premise is refuted on its own metric: σ_max = 1.7019 → 1.0015 over loops 2→64, **never contracting**, and the model saturates at 8 anyway — *saturation without convergence* | HOLDS |
| 2.1 Two comparability constraints (vocab 4096 ⇒ token-PPL incomparable; chunked eval used throughout) | HOLDS |

## §3 Architecture
| claim | status |
|---|---|
| 3.1 Qwen3 block verified against the reference to **2.38e-07** before spending compute | HOLDS |
| 3.2 Five ablation axes, `‖e‖/‖h‖ = 7e-5` explains the injection null | HOLDS |
| 3.3 Scale argument: no fixed-size table anywhere | HOLDS — but the live threat is *density*, not tables (Sparse Layers 2605.09165); named in `DECISIONS.md` |
| 3.4 Function-vs-table selection rule; IterAdaLN the only survivor | **NARROWED/CORRECTED** — an earlier draft said this architecture passes "trivially and uninterestingly". False since §4.17: **annealing IS loop conditioning via the loss**, and passes with *zero parameters* |
| **NEW** 3.5 **The final method, named**, with each load-bearing choice tied to the measurement that decided it, and the zero-parameter scale argument aimed at the task's value-embedding counter-example | **PROVISIONAL** — banner names the two runs that can falsify it |

## §4 Experiments — *opens with the spine* (**NEW**)
> *Where useful depth sits is set by the supervision schedule, not the dynamics. Three traversal interventions relocate the optimum without raising the ceiling; the map never contracts.*

| claim | status |
|---|---|
| 4.1 Screening sweep — **wall-clock budgeting was a methodological error**, flips 2 of 5 axes | HOLDS (self-retracted in-text) |
| 4.2 Full-budget run; sliding-window vs chunked bpb 1.6436 vs 1.6938 | HOLDS |
| 4.3 **Not contraction — geometric dilution.** ‖h‖ linear, tangential step ~constant, angular step ~1/t. σ_max > 1 at every depth | HOLDS |
| 4.4 Compute-matched non-looped baseline | **RETRACTED IN FULL** — the "untrainable" result was an MPS artifact; CUDA trains it at every LR |
| 4.5 Prelude/coda at fixed budget | **SHARPENED** — the prelude arm is depth-**inert** (plateau [1,96]); it wins CE by removing the reason to iterate |
| 4.6 Radial clamp: scale relocates the optimum without raising the ceiling | HOLDS. **NEW**: the 90M norm-penalty prediction **resolved ambiguously** — gain +0.2564 and plateau narrowing unambiguous, CE −0.0301 lands *between* the measured and conservative floors; both readings kept live |
| 4.7 Per-token depth demand: real, reliable (split-half +0.866 vs null +0.0007), **unreachable** by four signal families | HOLDS. **NEW**: now states the head is **PALBERT's own best row** (`[h_t,h_{t−1}]` + tanh), not Ouro's weakest |
| 4.8 Cross-depth KV: a ragged cache costs almost nothing | HOLDS |
| 4.9 Train-at-L; t/L collapse; optimum ≈ half trained L | **NARROWED** — the seed-1 test **FAILED its pre-registered rule** (0.0294 vs 0.0148). Now *a reproducible average relationship, not a law each arm obeys*. The half-of-L rule **did** replicate argmin-for-argmin at all five arms. **NEW**: the mechanism is **2024 prior art** (2311.12424), verified verbatim; §4.9 supplies the constants |
| 4.10 Convex gate + fixed-`g` sweep: no movement | HOLDS. **NEW**: withdrew a sentence that contradicted the section's own noise estimate |
| 4.11 Schedule shape sets the optimum | HOLDS. **NEW**: re-derived on the plateau (its argmins were 0.0019–0.0092) — claim survives, ratio 0.44–0.67 |
| 4.12 Loop gain **emerges** with tokens, then saturates | HOLDS — and supplies the mechanism for §4.17 |
| 4.13 Exploration noise during loops **hurts** monotonically | HOLDS |
| 4.14 Terminal-only supervision shifts the useful band | **NARROWED then STRENGTHENED** — "optimum doubles" (2×) was argmin at 0.003-nat margins; real effect **1.50×** on the plateau. **NEW**: replicated across **2 devices**, plateaus identical to the digit; CE cost is *not* stable (+0.017/+0.046 MPS vs **+0.191** CUDA) |
| **NEW** 4.15 **The noise floor, measured** from accidental replicates: MPS 0.031/0.068, CUDA dense **0.0150**, CUDA terminal **0.0541** — config-dependent, not just device-dependent. CPU bit-identical, MPS 9.5e-07/step. **63 of 82 stored curves have an unresolvable argmin** | HOLDS — the statistics everything else is judged against |
| **NEW** 4.16 Supervision density is a **threshold at k=1**, not a dial (gain drops 0.162 from k1→k2, then varies 0.018 across k=2,3,5,8) while CE improves monotonically 0.296 nats | HOLDS |
| **NEW** 4.16b Terminal-only's useful depth **tracks μ_rec** at 1.09/1.00/0.98 across three schedules; at μ_rec=40 the band is loops **32–48**. My prediction that the CE penalty shrinks with depth was **FALSIFIED** (non-monotone) and is retracted in-text | HOLDS (depth) / RETRACTED (price) |
| **NEW** 4.17 **Supervision annealing** — the *last* phase sets the band. `rev50` control: same k=1 exposure placed **first** gives no effect and the worst CE. In-job, 2 seeds: **sw90 beats dense at both (−0.081, −0.061)**; sw75's CE flips sign. At μ_rec=40: plateau **[32,64]**, an **interior maximum** exceeding both endpoints. Useful at **64 loops, 1.33× beyond max trained depth** | **PROVISIONAL** — budget test running |

## §5 What didn't work
| claim | status |
|---|---|
| **NEW** 5.0 `ε=λ/(N√L)` residual scaling: **no measurable benefit**, and its apparent optimum shift was a **0.0001-nat argmin artifact** | HOLDS — killed before publication |

## §6 Honest scope
| claim | status |
|---|---|
| **NEW** 6.0 **21 rows**: every substantive thing that went wrong and what caught it | the criterion-2 deliverable |
| **NEW** 6.0b Hyperparameters **inherited, not chosen** — LR across a regime change, unscreened weight decay, no gradient checkpointing, 80 multi-digit tokens, 92.0M of 100M packed, fp32/upcast verified | HOLDS |

## §7 Reproducing — full instrument set, incl. **NEW** statistics/audit tools
## §8 Where this points
| claim | status |
|---|---|
| 8.0/8.0b/8.0c Field position; seq_len is first-order; the t/L screen | 8.0c inherits §4.9's narrowing |
| **NEW** §8 opening: the 90M runs beat the prediction (**0.39–0.42 nats**, ppl 54.99 → **37.14/36.03**); annealing is the first config where CE and loop gain move **together** | PROVISIONAL |
| 8.1 The objection I would raise against my own architecture | HOLDS |
| 8.2 Per-token rate control as the differentiable dual of early exit | proposal, untested |

---

## On the three §1 drafts

**I'm not doing that one, and the reason is not squeamishness.** §1 is graded as the author's own idea
generation, and the task explicitly cautions about LLM involvement there. Three drafted voices would
not be neutral stimulus — whichever the author picked, the framing, emphasis and vocabulary would be
mine, and the one artifact the task reserves for them would be a selection from my options rather
than their account. That is a worse failure than an unwritten §1, and it is not recoverable once read.

What I have done instead is make the raw material complete: `LOG.md` carries the full chronology with
every retraction dated, §6.0 lists all 21 failures with causes, and this map gives claim-level status.
An author writing "I expected X, the arms said Y, so I built Z" has X, Y and Z all in the record with
timestamps. That is the input to §1; the sentence itself should be theirs.
