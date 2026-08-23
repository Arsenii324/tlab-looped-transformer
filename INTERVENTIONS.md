> **Dated working record — not a maintained document.** Last committed 2026-08-23; the text itself may be older. Kept intact rather than rewritten, because this project's rule is that superseded statements stay visible with their corrections. **Current numbers are in [`submission/`](submission/) and [`report.md`](report.md); where this file disagrees with them, they win.** See the repository README's *How to read this repository* table.

# INTERVENTION LEDGER — every method proposed in this project, with status

WORKING doc. Built 2026-08-23 02:10 by auditing QUEUE.md, REVIEW_NOTES.md, report.md and the
conversation. Status: TESTED / RUNNING / QUEUED / BUILT-NOT-RUN / WONTFIX(reason) / **GAP**.

## Normalisation & scale control
| method | status | result |
|---|---|---|
| inter-loop RMSNorm (`state_renorm`) | **TESTED** | largest effect in the project, −0.68 nats token-corrected. Contracts (σ_max 0.80–0.82), kills depth: optimum ~4, loop gain never emerges (§4.3, §4.12) |
| no scale control (the winner) | **TESTED** | σ_max 1.0015–1.70, never contracts, dies by 1/t dilution instead (§4.3) |
| radial clamp at inference | **TESTED** | relocates the optimum (5/15/24) without improving it — scale is a rate, not a ceiling (§4.6) |
| norm penalty λ‖h‖²_rms (Sharma&Vu) | **RUNNING** | Kaggle 90M, paired against control. Pre-registered prediction in §4.6 |
| raw readout (scale-visible CE) | **QUEUED** | `run_scale_control.py` |
| final-only-norm readout (hybrid) | **QUEUED** | same |
| ε = λ/(N√L) residual scaling | **QUEUED** | `run_residual_scale.py`, n_loop_eff corrected 24→18 |

## Mixing / topology
| method | status | result |
|---|---|---|
| prelude/coda sandwich | **TESTED** | double dissociation; prelude buys 0.355 CE and costs 86% of loop gain via 14× faster norm growth (§4.5) |
| **convex gate** (2-term mixing, mine) | **TESTED** | **negative on every axis** (+0.0203 CE, −0.0080 gain) despite more params (§4.10) |
| fixed-g sweep (g=1.0 ≡ ungated) | **RUNNING** | monotone-trend version of the above |
| softmax depth-mixing over past loop states | **BUILT-NOT-RUN → downgraded** | §8.2. Its pre-registered cheap test (the convex gate) failed, so expected value is now low |
| hyper-connections (Hyperloop) | **WONTFIX** | coarser version of depth-mixing; same cheap test failed |
| MixerLoop (loop the mixer, FFN once) | **WONTFIX** | verified from source: FullLoop has the LOWEST NLL at both 15M and 110M. Wrong direction for a perplexity-scored task (§8.1) |
| MLA / low-rank KV | **WONTFIX** | at seq_len 256 the KV cache is trivial; only a parameter-efficiency argument, which loses to fixing the MLP split |
| MoE / Looped-MoE | **WONTFIX** | spends stored params to buy sparse FLOPs — the opposite of a parameter-capped, FLOP-free regime |

## Conditioning / per-iteration differentiation
| method | status | result |
|---|---|---|
| input injection (none / additive / concat) | **TESTED** | `inject_none` worst; mechanism corrected — loops need a *k-varying* signal, not new information (§5) |
| IterAdaLN `v_k = MLP(PE(k))` | **BUILT-NOT-RUN** | §3.4: the **only** published loop-conditioning that survives the no-lookup-table rule. ~344k params. Never trained here |
| per-loop LayerNorm tables, iteration embeddings, depth-wise LoRA | **WONTFIX** | tables over t — undefined outside the trained range (§3.4) |
| soft-MTP per-loop targets | **WONTFIX** | manufactures depth-demand but its difficulty decays in t, so it buys a better optimum not a further one (§8.0) |
| SCSE anchor + zero-deviation mask | **BUILT-NOT-RUN** | names the forcing-bias quantity §4.3 measures; not implemented |

## Depth schedule & supervision
| method | status | result |
|---|---|---|
| randomized loop count U[4,32] | **TESTED** | the default |
| fixed loop count | **TESTED (screening)** | verdict flipped three times under token correction — unresolvable at that scale (§4.1) |
| **train-at-L sweep** L∈{2,4,8,16,32} | **RUNNING** | the task's literal question; §4.9 |
| schedule shape (shallow/uniform/concentrated) | **TESTED** | optimum tracks μ_rec 4/8/16, loop gain scales 3.8×, replicated on 2 seeds (§4.11) |
| dense per-loop supervision (k=5) | **TESTED** | the default |
| **terminal-only loss (k=1)** | **TESTED — shape-changer, but priced** | plateau [8,16]→[12,24] (bit-identical, 2 seeds), midpoint 11.3→17.0 (**1.50×**, not the 2× argmin suggested), loop gain 2.5× and non-overlapping across seeds. **Only intervention that breaks the t/L rule**: mid 0.94·μ_rec vs dense's 0.63 (§4.14). Plateau shift replicated across 6 arms / 2 devices, identical to the digit. **CE cost is NOT stable**: +0.017/+0.046 (MPS) but **+0.191 (CUDA)**, where 16 loops lose to the dense control at 1 loop. Shape-changer, not yet a usable mechanism |
| deep terminal-only (μ_rec 18/32) | **TESTED** | useful depth SCALES with trained depth (dense mid 11.3→22.6, terminal 19.6→32.0). Unpredicted bonus: the terminal CE penalty **shrinks** with depth, +0.191 (μ18) → +0.070 (μ32). μ=56 and μ=44 pairs OOM'd at batch 8; μ=40 running |
| supervision-density ladder k∈{1,2,3,5,8} | **TESTED — §4.16** | **THRESHOLD at k=1, not a dial.** gain drops 0.162 from k1→k2 then varies 0.018 across k=2..8; CE improves monotonically 0.296 nats across the ladder. No intermediate dose exists |
| supervision ANNEALING (dense→terminal partway) | **RUNNING** | MY OWN proposal, from §4.16: if density can't be dosed, try TIME at k=1. Arms switch 5→1 at 50/75/90% + a reverse control |

## Optimizer
| method | status | result |
|---|---|---|
| AdamW | **TESTED** | the default. Matched to Sharma&Vu (3e-4), the paper this replicates |
| Muon | **WONTFIX** | needs its own LR sweep, unaffordable. But the diagnostic ran: **the low-rank conjecture is refuted** — tied gradient stable rank 6.73 vs untied 4.40, i.e. tied is *more* spread. Unprompted finding: ‖G‖_F is **31× smaller** under tying, a candidate mechanism for §4.4's baseline NaN |

## Randomness / exploration
| method | status | result |
|---|---|---|
| random `h0` init (Huginn) | **WONTFIX** | Done Right's controlled transformation reports it mixed-to-negative (2 gains, 4 losses >1pt) — the one uniquely-Huginn ingredient that loses |
| loop-count randomization | **TESTED** | see schedule above |
| **stochastic exploration during loops** | **TESTED** | HURTS monotonically (σ 0.05/0.15/0.4 → CE −0.006/+0.183/+0.790, gain +0.004/−0.036/−0.125); optimum never moves. Resolves the pre-registered fork: the ray's coherence is **load-bearing**, matching Huginn's "sliders" reading (§4.13) |

## Architecture shape
| method | status | result |
|---|---|---|
| MLP ratio 3.0 vs 8/3 | **NOT TESTED** | self-assessment corrected: worth ~600k params = H 448→468, 4.5%. The position-local *split* matters more than the ratio (§8.1) |
| seq_len 256 → 512 | **NOT TESTED** | promoted to a first-order limitation by Huginn's context-scaling result (§8.0b) |
| ALBERT-factorised embeddings | **NOT TESTED** | would free budget under the total-param convention |
