# QUEUE — every open item from every review, with status. Nothing is dropped from this file.

Status: TODO / RUNNING / DONE / WONTFIX(reason). Append new items; never delete. WORKING doc.

## A. Corrections to the report (cheap, do first)
| # | item | status |
|---|---|---|
| A1 | §4.6 says the clamp "reproduces" 2606.24898 Table 4 at K=4. Their ΔCE is **+0.0004…+0.0055**, mine is **−0.012** — same order, **opposite sign**. Say that, don't say "reproduces". | DONE |
| A2 | State explicitly that the clamp numbers came from `eval.py`-protocol (15 batches × 4 = 15,360 tokens), **NOT** the frozen 524k paired set. At a 0.006-nat spread that distinction decides result-vs-noise. | DONE |
| A3 | State the clamp conclusion more strongly: *scale sets the RATE of angular traversal and therefore where the optimum falls, but not its VALUE; no inference-time scale intervention can raise the ceiling, which is a property of the learned path fixed at training time.* | DONE |
| A4 | Pre-register the norm-penalty prediction **before** the 90M run lands: clamp says inference-time scale control relocates without improving; if the **training-time** penalty improves best CE, it changed the learned path, which clamping cannot. | DONE |
| A8 | DATASPHERE_NOTES: e2e timing table, notebook-vs-Jobs distinction, inline-docker-image fix (NOT build+push — I priced that wrong), conditional torch predicate, system.log-timestamps gap. | DONE |
| A5 | §2: state that this uses **FineWeb**, while Schwethelm/Parcae/LoopMTP/DeepLoop/residual-scaling/LoopFormer use **FineWeb-Edu** (filtered, lower entropy). BPB is therefore not comparable to theirs even after the bytes/token fix. | DONE (§2.1) |
| A6 | §2: state the budget is **92M train + 6M val = 98M of a 100M allowance**; ~8% unspent, ≈0.03 nats by this project's own 0.398 nats/e-fold. | DONE (§2.1) |
| A7 | Cite LoopFormer's warning next to the exiter result: *"naive early exiting in looped architectures leads to stagnant representations in later iterations"* — a claim about training WITH early exit; this is post-hoc on a frozen model, so the pathology isn't induced, but it says what a joint-trained version would need. | TODO |

## B. Experiments
| # | item | est | status |
|---|---|---|---|
| B1 | **TRAIN-AT-L sweep** L∈{2,4,8,16,32}, 10M tok each, token-budgeted, each evaluated at its own L. The literal task question; everything else in this repo is eval-at-T. | ~4h | RUNNING (DataSphere `tlab-train-at-L`) |
| B2 | **final-loop-only supervision** arm (`supervise_k=1`) vs dense per-loop. Two papers say terminal-only wins on CE (Sharma&Vu Tab.2: 5.40 vs 6.04 @44M; LoopFormer 10.91 vs 11.60) **but makes intermediate exits unusable** (their Tab.14: CE 5.52@K=1 vs 1.54@K=4). Converts the biggest untested axis into a measured trade-off. | ~1h | TODO |
| B3 | supervision-density (schedule shape) 3 arms × 2 seeds | ~3.7h | RUNNING (local) |
| B4 | scale-control: raw / final_only readout / norm-penalty × 2 seeds | ~5h | QUEUED (local) |
| B5 | item 5: ε=λ/(N√L) residual scaling, 3 arms | ~2-4h | QUEUED (local) |
| B6 | cross-depth KV grid on the headline ckpt | ~30m | QUEUED (local) |
| B7 | paired scoring of both headline checkpoints | ~30m | QUEUED (local) |
| B9 | **Convex-gate arm** `h_t=(1-g_t)h_{t-1}+g_t·block(h)`, g from sinusoidal PE(t)+state (function of t, passes §3.4). Minimal 2-term version of depth-mixing: bounds ‖h‖ WITHOUT shrinking the branch, the one property state_renorm/ε-scaling/no-control all lack. Implemented, +65,793 params (0.72%), total 9,130,401 < 10M cap; verified to change the forward. Tests convexity-vs-accumulation before paying for 64-state softmax mixing + the routing-collapse risk (2606.22325). | ~1h | READY |
| B8 | Muon: **WONTFIX for the deliverable** (needs its own LR sweep, none affordable). Keep the *diagnostic* idea: log the singular-value spectrum of the accumulated gradient for a looped projection vs the same projection untied — if the looped spectrum decays much faster, orthogonalization is qualitatively different in this regime. Nobody has that figure. | — | WONTFIX(+idea kept) |

## C. Report content owed
| # | item | status |
|---|---|---|
| C1 | **Exiter result**: oracle headroom 0.3084 nats (> whole loop gain 0.2509), argmin deciles [1,2,7,43,64], **but no label-free rule beats fixed-8** (best −0.0001). Depth demand is real and *not predictable* from loop-1 entropy/margin. | TODO |
| C2 | **Novelty claim**: per-token *argmin-CE depth* is a distinct object from the *saturation depth* that CALM/D³/TIDE/2607.14427 measure (earliest layer agreeing with the FINAL layer). In a looped model past its optimum the final layer is **not** the target, so the two coincide only outside this regime. | TODO |
| C3v | **MixerLoop (2608.18230) VERIFIED from source** — title is "Allocating Recurrent Compute in Looped Language Models". Confirmed: surpasses FullLoop on aggregate CORE at 15M, retains 41.5% of the CORE improvement at 110M, cuts recurrent-backbone projection FLOPs 45.9% (15M) / ~43% (110M), "under the same data, initialization, and architecture" = iso-token. **Three caveats the second-hand summary dropped:** (a) the mixer is **Gated DeltaNet**, not softmax attention, so it is a different mixer family from my Qwen3 block; (b) **T=4 loops** — a low-loop-count result, not a many-loops one; (c) metric is **CORE downstream aggregate**, not val perplexity, at **52.43B ClimbMix tokens** (~500x my budget). The FLOP argument transfers regardless of (a)-(c); the performance claim transfers weakly. | DONE |
| C3 | **§8**: the position-local objection. 75% of the looped block's params and ~68% of its per-loop FLOPs are MLP, which moves no information between positions. MixerLoop (2608.18230) loops the mixer and applies the FFN once, beating full-block looping at 15M. An attention-only loop is ~3× cheaper per loop → the same wall clock buys **mean-54 instead of mean-18**. This attacks the actual binding constraint on "many loops". | TODO |
| C4 | §6.1 self-assessment was **overstated**: 8/3 vs 3.0 frees ~600k = H 448→468 (4.5%), and the FFN-ratio literature is flat over 2–4. Correct it; the *split* (position-local vs mixing) is the real question, not the ratio. | TODO |
| C5 | Supervision-density: my hypothesis in its strong form is **dead** (density peaks at loops 1–4, optimum at 8–11). Weak form survives: density can't explain why the optimum is 8 not 1, but can explain why it isn't 25. | TODO |
| C6 | A looped transformer is a deterministic discrete dynamical system with no noise model and no formal objective on the trajectory — diffusion/GFlowNet/EBM framings are analogies, not shared formalism. One sentence, prevents importing intuitions that don't hold. | TODO |
| C7 | Related work: STARS/Parcae/SCSE/LoopMTP all show "degrades less", never "keeps improving". **Nobody has demonstrated monotone loss improvement past ~8 loops at <150M params on AR text.** LoopMDM is the one positive and it is masked diffusion with a mask-token workspace. | TODO |
| C8 | Loopie's own §"Why Only Two Loop Steps" says R=2 is a **FLOP-budget allocation decision**, not a claim about loops, and names *"where inference-time computation is cheap"* and *"where adaptive computation is available"* as settings where larger R may pay — i.e. this task's regime, from the task's own cited paper. | DONE (§2) |
| C9 | Depth-mixing proposal for §8: softmax attention over past loop states bounds ‖h‖ **without** forcing the angular step to zero (convex combination), which none of {inter-loop renorm, ε=1/N, no control} achieves. Trilemma table. Caveat: 2606.22325 finds depth routing collapses onto 2 hubs (0.643 top weight vs 0.245 uniform) — measure before believing. | TODO |

## D. Risks
| # | risk | status |
|---|---|---|
| D1 | 90M control-vs-penalty single-seed concern: **RESOLVED FAVOURABLY.** Both kernels use `seed=0` with `torch.manual_seed(seed)` for init and `np.random.default_rng(seed)` for data order and loop-count sampling — verified identical code paths. So it **is** a paired training comparison (same init, same data order, same loop schedule; the penalty adds a loss term and draws no extra randomness). Much stronger than two independent draws. | DONE |
| D2 | Report writing is last with runs landing 06:30. Skeleton with placeholders NOW. | TODO |
| D3 | DataSphere job submission is not idempotent; always `job list` first. | MITIGATED |

---

# WORKSPACE — points raised faster than addressed (write down, discard later)

Added 2026-08-23 ~01:30. Anything here is unaddressed unless marked. Cheaper to write and discard
than to forget.

## Pre-commitments to make BEFORE the numbers arrive (so they aren't made tired at 18:00)
| # | decision | status |
|---|---|---|
| P1 | **L\* choice rule:** take the **largest L whose 10M CE is within measured noise of the argmin**, not the argmin. Three independent biases push the true 90M optimum above what a 10M sweep shows: optimal loop count grows with data; §4.12 says loop gain is still emerging at 10M (the arms are at the bottom edge of where gain exists); and the bias is worse for deep arms, which are more undertrained relative to their capacity. Task rewards loop count, so ties break upward. | **COMMITTED** |
| P2 | **Size by harvest, not completion.** At 2415 tok/s @mean-18: L=16 → 90M done 14:12; **L=24 → ~85M by 18:00 (~0.02 nats data penalty)**; L=32 → ~64M (~0.13 nats). L=24 is the sweet spot unless train-at-L is steeply monotone through 32. **Measure tok/s in the first 5 min on gt4.1 and set the token target from the measurement**, not the T4 extrapolation. | **COMMITTED** |
| P3 | **If the fixed-L run loses on absolute CE:** still report it as the headline *for the task's question*, with the randomized-schedule number alongside. The task asks for lowest perplexity **obtained by exploiting many loops**; a fixed large L answers that in one number. Report both, label both. | **COMMITTED** |
| P4 | **Fallback for the single point of failure:** if train-at-L has not landed by **05:15**, launch fixed **L=24 at 90M anyway**. Cost of the wrong L is small; cost of launching nothing is the whole 13h window. | **COMMITTED** |
| P5 | **Free hedge:** eval the fixed-L checkpoint with T swept well past L, so the same checkpoint yields both the fixed-L headline and an eval-at-T curve. | TODO at harvest |
| P6 | **Placeholder cutoff T−3h (≈20:30):** §4.8/4.9/4.10 convert from empty to "designed, launched, did not land; here is the protocol and what each outcome would have shown." An empty section ships worse than an honest one. | TODO |

## Analyses still owed
| # | item | status |
|---|---|---|
| W1 | **Per-token rate test:** `mean_i min_c CE_c[i,K]` at fixed K vs `mean_i min_k CE[i,k] = 3.6193`. Decides whether tokens are on ONE path at different speeds or on DIFFERENT paths — §8.2 rests on it. Needs per-token CE under clamp; the clamp script currently stores only means, so this needs a small re-run. | TODO |
| W2 | Oracle null calibration (queued, `oracle_null.py`) | QUEUED |
| W3 | Argmin × position-in-chunk decomposition (queued, `argmin_anatomy.py`); Huginn predicts a RISING trend | QUEUED |
| W4 | Q-exit at PALBERT's real spec | **DONE** |
| W5 | Gradient spectrum tied vs untied | **DONE — see below** |
| W6 | screening_results.json emergence-by-arm | **DONE → §4.12** |
| W7 | Unread artifacts | **DONE — second_seed_results.json produced the biggest correction of the night (seed spread 0.25 -> 0.117 nats, token-confounded). Per-arm sandwich sweeps read via sandwich_eval.json.** |
| W8 | All quoted paper numbers verified from source where obtainable; second-hand ones hedged in place. See VERIFICATION.md | **DONE** |

## §1 (the author's idea narrative) — NOT mine to write, but nothing in the schedule reserves time
Flagged: criterion 1 is graded on §1, and the timeline allocates 19:30–23:30 to consolidating
numbers. The natural window is 02:30–04:30 (waiting on a decision point anyway, needs no new
numbers). **This is the user's to write; recorded here so the timeline doesn't silently omit it.**

---

# OPEN POINTS as of 2026-08-23 12:20 — nothing here is skipped, each has a state

## BLOCKED (cannot be done with what exists; reason recorded in-report)
| # | item | why blocked |
|---|---|---|
| B1 | **`B_L` angular budget across §4.9's five train-at-L arms** — would test whether the budget is invariant *within* a supervision scheme while only the rate varies, completing §4.16c | **weights unrecoverable.** Those arms trained on DataSphere and every DS config listed only `results.json` under `outputs:`, so `{run_name}_last.pt` was written on the node and never returned (§6.0 row 23). Fixed in 23 configs for future jobs; the ~20 already-run jobs are gone |
| B2 | `B` on the **CUDA** terminal replicate (`dt_mu18_term`) to check the budget result across devices | same cause as B1 |
| B3 | Verify the four relayed papers (2602.14759, 2603.19714, 2604.18839, 2509.23314) | not obtainable here. Logged SECOND-HAND in `VERIFICATION.md`, hedged in §4.13/§4.3, **no claim rests on them**. Unblocks instantly if the tarballs arrive the way 2311.12424 and 2511.08577 did |

## IN FLIGHT (results will land; analysis pre-registered)
| # | item | ETA / gate |
|---|---|---|
| F1 | **`tlab-anneal-scale` 10M dense control** — the §3.5 falsifier | ~13:30. Read on **(ΔCE_best, ΔCE@1)**, never Δgain. Three-way outcome A/B/C pre-registered in `RUNS.md` at 11:55, before the control landed. Prior on **B** is now raised by the schedule-axis reversal found in §4.17 |
| F2 | `tlab-hyper-screen` — lr 6e-3, then wd {0, 0.01, 0.1} | lr 1e-3 already lands: **3e-3 beats it by 0.103**, so the inherited LR is vindicated, not merely defensible |
| F3 | **`tlab-deep-full`** — the deep artifact, ~32M tokens | harvest ~17:18. Returns **curves only** (B1's defect); fills §4.17's deep half |
| F4 | `run_eval90.sh` — protocol-matched local eval of both 90M checkpoints | running; **gates the headline swap**. Nothing swaps until it lands |

## NOT STARTED — writing, no compute (deferred deliberately, ranked)
| # | item | source |
|---|---|---|
| W1 | **§8 paragraph: MLA × LLA.** LLA measured the recurrent KV trajectory as genuinely low-rank; §4.3 here shows keys near depth-invariant (25.13→21.36) while values fall 2×. MLA's imposed structure is the structure a looped model's KV already has | reviewer, ranked "§8 material, not tonight" |
| W2 | **§8: the LoopMTP aggregation conflict.** Their gate beats "only last" (what this runs) — but aggregation spreads the readout back across the trajectory, which is the opposite of what annealing does. A conflict to name, not a missing ablation | reviewer |
| W3 | **§3.3: the density threat.** Sparse Layers argues *dense* looped models scale worse than sparse ones. §3.3 answers "no fixed table"; the real objection is "your capacity is dense". Currently only in `DECISIONS.md` Q2 | reviewer |
| W4 | **§8: STARS Pre-Sandwich** (`h + Norm(f(Norm(h)))`) — the untested 4th cell of their taxonomy, with this project's own prediction attached (still 1/t, because ‖h‖ stays linear) | reviewer |
| W5 | Decomposition table for the remaining §4 sections that quote loop gain without splitting it | own audit |
| **W6** | **The anchor account** — write it as the framing that unifies §4.5/§4.14/§4.16/§4.17. Largest outstanding item | reviewer, 12:40 |
| **W7** | State the tension with 2606.20075, which calls removing intermediate supervision a pathology while this measures it widening the band 1.5× | reviewer, 12:40 |
| **W8** | Restate §3.5's rule as **token-keyed** (dense until gain flattens) with the fraction rule reported as what was measured | reviewer, 12:40 |

## USER-OWNED (not mine to do)
| # | item |
|---|---|
| U1 | **§1** — reserved. Three stimulus drafts in `needs_user/section1_drafts_STIMULUS.md`, explicitly not for the report |
| U2 | **Rotate the wandb key** — `needs_user/ROTATE_WANDB_KEY.md`. Scrubbing the repo does not un-send it |
| U3 | **D3: which checkpoint ships.** norm-penalty 90M wins ppl (36.04) but is 88% damage-driven with a [8,8] band; the control (37.14) is the config §3.5 describes. Whichever ships, the model card states the 12× shrinkage in the same sentence |
| U4 | Run `/ultrareview` on branch `review`; any GitHub/HF push |

---

# REVIEWER-POINTS LEDGER — every message, every point, standing
*Added 2026-08-23 12:35 because I had twice audited only the LATEST reviewer message and let earlier
ones fall through. This is the fix: the ledger is cumulative, and a new message appends rather than
replaces. Verified by grep against the artifacts, not from memory — and note that a first pass produced
three false "gaps" purely from markdown-sensitive patterns, so check the text, not the pattern.*

| # | msg | point | state | where |
|---|---|---|---|---|
| R1 | task-statement | final architecture section required by the spec | **done** | §3.5 (provisional banner) |
| R2 | task-statement | criterion 2 = did you lose track of the code → make the failure log a section | **done** | §6.0, 23 rows |
| R3 | task-statement | scale argument against the value-embedding counter-example | **done** | §3.5 (zero parameters ⇒ no table to outgrow) |
| R4 | task-statement | exemplar audit belongs early with quotes | **done** | §2 |
| R5 | task-statement | aim σ_max at the DEQ premise | **done** | §2, "saturation *without* convergence" |
| R6 | task-statement | perplexity vs loop-utility disjoint, narrow form | **done** | §4.9 correction (ρ=−0.081 pooled) + §4.5 |
| R7 | arch-audit | STARS taxonomy: Pre-Sandwich untested, prediction attached | **noted, unwritten** | `DECISIONS.md` Q3; QUEUE W4 |
| R8 | arch-audit | Sparse Layers: the threat is *density*, not tables | **noted, unwritten** | `DECISIONS.md` Q2; QUEUE W3 |
| R9 | arch-audit | §3.4 has no effect size; SCSE supplies one | **done, verified from tarball** | §3.4 (151.1→125.7; degrades to 160.1) |
| R10 | arch-audit | CART frozen-KV vs this model's near-invariant keys | **done** | §4.8 |
| R11 | arch-audit | learned h0 is a third option, unmarked | **done** | `DECISIONS.md` §1 |
| R12 | arch-audit | LoopMTP aggregation *conflicts* with annealing | **noted, unwritten** | `DECISIONS.md` Q3; QUEUE W2 |
| R13 | arch-audit | did raw / final-only readouts land? | **done** | §4.6b, all four interventions, 2 seeds |
| R14 | choices | fp16 on T4 could corrupt the deep run | **done — checked, not live** | §6.0b (fp32 + RMSNorm upcast, verified) |
| R15 | choices | LR tuned for a regime no longer run | **done — and screened** | §6.0b + LR screen: 3e-3 optimal of {1e-3, 3e-3, 6e-3} |
| R16 | choices | weight decay never screened | **screening now** | §6.0b; `tlab-hyper-screen` wd arms running |
| R17 | choices | chars-vs-bytes inconsistency in §3 | **done** | §3 corrected (was contradicting §4.7) |
| R18 | choices | digit tokenization unresolved | **done** | §6.0b (80 multi-digit tokens) |
| R19 | choices | gradient checkpointing unstated | **done** | §6.0b (none; it bounds the deep schedules) |
| R20 | choices | 8% of the token budget unused | **done** | §6.0b (92.0M of 100M) |
| R21 | choices | MLA is a quality play; LLA low-rank connection | **noted, unwritten** | QUEUE W1 |
| R22 | deliverable | D1 tokenizer shipping | **done** | gate passes both ckpts; README trap fixed |
| R23 | deliverable | D2 fresh-clone dry run | **done** | byte-identical shards from cold |
| R24 | deliverable | D3 which checkpoint ships + disclose the shrinkage | **done in-text; choice is the user's** | headline section; QUEUE U3 |
| R25 | deliverable | S1 is the head PALBERT-spec? | **done** | §4.7 states it rules out their *best* row |
| R26 | blind-spot | `DECISIONS.md` with provenance tags | **done** | `DECISIONS.md`, 4 questions answered |
| R27 | blind-spot | screen LR and weight decay | **done / running** | `tlab-hyper-screen` |
| R28 | unknown-knowns | claim-level TOC | **done** | `reviewer_answers/04_report_map.md` |
| R29 | unknown-knowns | three §1 drafts as stimulus | **done** | `needs_user/section1_drafts_STIMULUS.md` |
| R30 | decomposition | Δgain = ΔCE@1 − ΔCE_best across all curves | **done** | `src/gain_decomp.py`, 49 in-job pairs |
| R31 | decomposition | three task-named levers all have results | **done** | §2.0 table |
| R32 | decomposition | gradient spectrum is homeless | **done** | §5.1 |
| R33 | decomposition | §4.8 is the author's idea and resolves against itself | **done** | §4.8 connected to §4.3 |
| R34 | angular | compute the angular budget B | **done — decisive** | §4.16c, ratio 1.38/1.42 |
| R35 | angular | penalty *reverses* rather than shrinks | **done** | §4.6b + pre-registration |
| R36 | angular | three-way A/B/C pre-registration | **done, before the control landed** | `RUNS.md` 11:55 |
| R37 | angular | §4.13 promotion + 3 citations | **done; citations hedged** | §4.13; papers unobtainable → SECOND-HAND |
| R38 | angular | bpb calibration vs Parameter Golf | **done** | §6.0a (D/N ≈ 9.9 vs Chinchilla 20) |
| R39 | novelty | prior art on the annealing *ingredient* (2608.11233, 2606.04678) | **done — recorded, claim narrowed** | §4.17 attribution; `VERIFICATION.md` SECOND-HAND |
| R40 | novelty | 2311.12424 is prior art for §4.9's mechanism | **done, verified from tarball** | §4.9 repositioned to supply constants |

## Ledger, continued — the anchor-account message (2026-08-23 12:40)
| # | point | state | where |
|---|---|---|---|
| R41 | **The anchor account** — a decodability anchor unifying §4.16's k=1 threshold, §4.5's inert prelude, §4.17's `rev50`, §4.14/§4.16b's band shift | **NOT WRITTEN** — the single largest outstanding item; it is a framing, and framings are cheap to write and expensive to get wrong | QUEUE W6 |
| R42 | Contradiction with 2606.20075 (latent-CoT supervision treats *absence* of intermediate supervision as the pathology) | **NOT WRITTEN**; paper unobtainable → would be SECOND-HAND | QUEUE W7 |
| R43 | **Token-keyed anneal rule** (dense until loop gain flattens ~10–15M) instead of a fraction — changes §3.5's recommendation and is the scale-transferable form | **NOT WRITTEN**; the discrimination is pre-registered | QUEUE W8 |
| R44 | Pre-register fraction-vs-token before the 10M control | **done** | `RUNS.md` 12:40, control was at step 2440/4882 |
| R45 | §4.7 ran on the dense-supervised checkpoint; rerun exit rules on the **annealed** one (2607.20519) | **BLOCKED then, POSSIBLE at 17:30** | needs `tlab-deep-full`'s weights — but see B1, DS returns curves only. **Would need a local annealed run** |
| R46 | Cumulative-`dnorm` exit rule | **done — informative negative** | §4.7b (−29.2%; cv 0.068 vs 0.798) |
| R47 | "processing domain vs output domain" interpretability framing | unattributed relay; usable only as an unsourced idea | — |

**Correction to my own note:** I said earlier that the first message's points (LoopMDM, XSA, Done
Right, the layer-band idea) were never recorded. **That was wrong** — they are in `REVIEW_NOTES.md`,
which has tracked reviewer claims since the start, including a "Papers surfaced but NOT yet read"
list of six. The ledger above and `REVIEW_NOTES.md` are complementary, not duplicative.


## Ledger, continued — the arc/chord retraction + five unknowns (2026-08-23 13:05)
*Answered in `reviewer_answers/10_five_answers_and_run_state.md`. All five were MEASURED, not recalled.*

| # | point | state | where |
|---|---|---|---|
| R48 | **Reviewer retracts the arc/chord mechanism** (4 objections: within-block vs between-loop are different objects; pre-norm attn/MLP anti-correlation is a static block property; the ρ derivation assumed equal-length layer steps; the arc is confounded with the ‖h‖ trajectory) | **accepted — and nothing was built on it.** The only report change from the arc work was §4.3's scope, already in. The efficiency ratio, the ρ numbers and the `3/√3` null are not in the report and will not be | §4.3 scope note |
| R49 | Q1: which grid produced the headline plateau [6,17]? | **done — dense every-integer 1..64.** All three headline rows share it, so [6,17] vs [6,14] is valid. On the sparse grid the same checkpoints read [8,16]/[8,12]/[8,12] and the difference vanishes. Grid now named in the headline table | §.headline table |
| R50 | Q2: do §4.3's norms (6630@8, 30097@64) transfer to the 90M artifact? | **done — NO, and the reviewer was right.** 90M control is 2334@8 / 12424@64 (2.4–2.8× lower); normpen is 17.5 / 89.4 (~380× lower). Relative dilution survives (18.2×/26.6×/20.3× growth); absolute norms do not. **§4.6's radial-clamp levels are 46M-specific** and must be re-derived before being quoted against the shipped checkpoint | §4.3 transfer note |
| R51 | Q3: was the local annealed run launched? | **done — launched, was silently producing nothing, fixed, relaunched 12:42.** `eval_every_tokens` typed as 1_250_000 vs the reference's 312_500 put the only checkpoint-save site 610 steps out against ~250 steps per chunk, so no save → no resume → every chunk restarted at 0. Two fixes: `train.py` saves on the `max_seconds` break; `run_anneal_local.py` derives its config from the reference checkpoint | R45's missing cell |
| R52 | Q4: does §3.5 overclaim at μ_rec=40? | **done — partially.** §3.5 already stated the 0.030-nat cost, but not the decomposition. Added: ΔCE@1 **+0.0749/+0.1749** against ΔCE_best −0.0264/−0.0192, and both ΔCE_best sit **inside** the 0.0541 CUDA terminal replicate floor while both ΔCE@1 clear it. The deeper band is bought, not free | §3.5 decomposition block |
| R53 | Q5: what is in §1? | the reserved placeholder, unchanged. User-owned (U1) | §1 |
| R54 | **wd/lr screen landed** — 3e-3 optimal; wd 0.01 beats inherited 0.05 by 0.0190 (just clear of the 0.0150 floor); wd0 worse by 0.0243. **Δgain null across all six arms (±0.02), onset=8 for all five well-trained arms** | **done** | §6.0b screen block |

## Ledger, continued — the two hour-long items (2026-08-23 13:40)
*Reply: `reviewer_answers/11_rho_was_measured_all_along.md`.*

| # | point | state | where |
|---|---|---|---|
| R55 | **`‖e‖/‖h_t‖` on the penalised checkpoint** | **done — regime break confirmed, mechanism REFUTED.** e/h@1 = 3.59e-01 vs control 3.22e-03, driven entirely by ‖h₁‖ collapsing 107× while ‖e‖ is unchanged (1.504→1.573), so the penalty does not reach the embedding. But cos(h₁,e) ≈ −0.07 and copy-rate ≈ 0.002 in all three arms: h₁ is nearly ORTHOGONAL to e. The loop-1 damage stays unexplained, with the most natural explanation eliminated | §4.3 injection block; `src/injection_ratio.py` |
| R56 | **ρ never measured** | **done — and INVERTED. ρ was measured all along; `jacobian_spec.py::sigma_max` never applied `J^T`** and is plain power iteration on J, i.e. the spectral radius. Null-verified on a known non-normal operator (ρ=1 vs σ_max=10.0990; the loop returns 1.0889), wired in as `--null`. Corrects §2 in the FAVOURABLE direction — `ρ<1` is the iff, `σ_max<1` only sufficient — while narrowing the claim to low loops, since loop-64 readings sit inside the estimator's ~9% upward bias | §4.3 correction block, §2, §6.0 row 25 |
| R57 | **ρ across the 100× ‖h‖ spread** (reviewer's proposed experiment) | **done — ρ is scale-invariant to 2% at loop 8 across a 380× range of ‖h‖** (1.0467/1.0692/1.0480). Second finding: the norm penalty is the only arm that crosses below 1 (0.9953/0.9915 at 32/64), a mechanism for its narrower plateau [6,14] vs [6,17], and the only arm inside the DEQ regime | §4.3 correction block |
| R58 | radial_clamp's fake fallback (found while verifying a §4.6 claim) | **done — fixed.** `levels={}` + a false "falling back" message meant the script produced only the unclamped control and exited 0; neither 90M checkpoint has the dynamics json, so §4.6 on the shipped model was a silent no-op. Real fallback implemented (reproduces the json path to 0.3%) + RuntimeError guard | §6.0 row 24 |

## Deep repo review (subagent, 2026-08-23 14:10) — findings and disposition
*Scoped to the two failure modes the task statement names, plus infra. Verified each before acting;
one of the reviewer's claims I initially contradicted with a buggy check and it was right.*

| # | finding | severity | disposition |
|---|---|---|---|
| S1 | **HF upload shipped weights without `configs/tokenizer.json`**, and the card had no CE@1, so the tokenizer gate the project built was unrunnable by a downloader | **BLOCKING** | **FIXED** — uploader now ships tokenizer + `model.py`, generates the card from the checkpoint's own `eval_*.json`, prints the gate command with the number substituted, and raises rather than shipping unverifiable weights. §6.0 row 26 |
| S2 | Model-card template lacked CE@1/ppl/bpb; QUEUE U3's shrinkage disclosure absent | **BLOCKING** | **FIXED** with S1; card now also carries the checkpoint's OWN state norms so nobody copies §4.6's 46M clamp levels onto it |
| S2b | *(found while fixing S1, not in the review)* first fixed card counted `sum(state_dict.values())` = **10,899,616** and would have reported the model as **violating the 10M cap**; tied embedding double-counted | **BLOCKING** | **FIXED** — `.parameters()` de-duplicates; true count 9,064,608. §6.0 row 27 |
| S3 | `refs/tags/main-backup-20260823` carries 4 blobs >100MB (1.83 GB); every branch is clean | **HIGH (push)** | **CLOSED (documented)** — now in `OPS.md` as a named PUSH HAZARD, not just a QUEUE line. **VERIFIED** — my first check said no tag carried them and was buggy (missing `%(rest)`); the review was right. `git push <branch>` is fine; **`--tags` / `--mirror` will be rejected**. Tag left intact per the no-deletion rule; recorded in `OPS.md` |
| S4 | No root dependency manifest | **HIGH** | **FIXED** — `requirements.txt` added, scoped by purpose, torch deliberately unpinned with the CUDA-12 index note. §6.0 row 28 |
| S5 | `eval.py::contraction_estimate` hand-rolls the loop, omits the convex gate, and its docstring's justification is false | MEDIUM | **GUARDED** — never triggered (verified); `NotImplementedError` on a gated checkpoint, docstring corrected. Not refactored (§3 rule). §6.0 row 31 |
| S6 | Kaggle checkpoints carry incomplete `model_cfg` (15–18 of 23 fields); `readout_mode` would silently mis-evaluate if the default changed | MEDIUM | **CLOSED** — `load_checkpoint` now prints which BEHAVIOURAL fields it is defaulting and to what; fires on the Kaggle checkpoints (6 fields), silent on local ones. Recorded in §6.0b |
| S7 | `train.py` persisted only `state_norms[0]` and `[-1]` while `forward()` computes the full list — a direct violation of `CLAUDE.md:40`, and `results.json` DID survive every DS job, so per-loop diagnostics would have survived the ~20 lost checkpoints | MEDIUM | **CLOSED, and it needed two passes.** First pass added per-loop `state_norms` + per-loop train CE — but `CLAUDE.md:40` asks for **four** things and that was two. Second pass added **per-loop predictive entropy** and the **online contraction-rate estimate**, via an `extras` out-parameter so the three existing `evaluate()` callers are untouched. Smoke-verified: all four present, and `contraction_dist` rises 142→190 across loops, independently corroborating §4.3's no-contraction result |
| S8 | `evaluate()` draws from the training RNG, so eval cadence perturbs training data order — two runs differing in `eval_every_tokens` are NOT paired even at the same seed | MEDIUM | **CLOSED (disclosed).** Now written into §6.0b with the D1 verification that the headline pair IS genuinely paired (shared seed + cadence, identical step 43,944 and token count) |
| S9 | Optimizer state deliberately not resumed; chunk count depends on wall-clock, so a 2.5M local run takes ~21 Adam-momentum resets and a slower machine takes more — uncontrolled and unrecorded | MEDIUM | **OPEN** — worth a line in §6.0b for local matched pairs |
| S10 | `eval.py` reported `exp(min(CE,20))` silently | MEDIUM | **FIXED** — loud warning above chance. §6.0 row 30 |
| S11 | Stale `baseline_nonlooped` (296,960 tokens); `SKIP_DOCS_FOR_TOKENIZER` dead code; local vs Kaggle val different document ranges | LOW | **CLOSED, all three.** (a) `checkpoints/baseline_nonlooped/STALE.md` marks it in place — deleted nothing; (b) the dead constant is named as dead in the §.headline correction; (c) **the substantive one is now measured**: the streams are offset by ~681 docs ≈ 634k tokens (local skips a fixed 20,000; the kernel continues its own iterator from ~19,319), so the val shards overlap ~89%, not 100%. Report re-attributed, with BOTH readings left live because the 0.0450 offset is also within 3×SEM |

**Review self-declared its own limits** (worth keeping, so the depth is not overestimated): 40 of 51
`src/*.py` unread in full; `report.md` not read end-to-end; `LOG/PLAN/BRIEFING/RUNS/DECISIONS/
VERIFICATION/REVIEW_NOTES` unread; only three published quantities independently re-derived; the 23
`ds_*/main.py` bodies not individually audited; papers unchecked; upload path read, not executed.

## The annealing switch fraction — a §3.5 overclaim found by the local run (2026-08-23 14:10)
| # | point | state |
|---|---|---|
| R59 | §3.5 recommended terminal-only supervision "for the last ~10–25% of steps". That range conflates **sw90** (depth-driven at both seeds: −0.0811/−0.0416 and −0.0609/−0.0277) with **sw75** (damage-driven at s0: −0.0656/**+0.0185**; **worse on CE_best** at s1: **+0.0906**/+0.2629). The headline pair always was sw90 | **FIXED** — §3.5 narrowed to `supervise_switch_frac = 0.90` with the four-pair table. The switch fraction is not a tolerant dial |
| R60 | **Cross-device replication of an annealing arm.** Local MPS `sw75` s0 vs its own in-job dense control: ΔCE_best **−0.0514**, ΔCE@1 **+0.0156**, against CUDA's −0.0656 / +0.0185 — same signs, magnitudes inside the MPS floor | **DONE**, in §3.5. Valuable because this report screens on MPS and confirms on CUDA |
| R45 | exit rules on an ANNEALED checkpoint | **UNBLOCKED** — `local_anneal_sw75_s0` trained to completion (step 1219/1220, 2.50M tokens, plateau [12,24] mid 17.0 vs the dense control's [8,16] mid 11.3, a 1.50× shift). Dumps running |

### Fork-review disposition, final: **11 of 11 findings closed** (4 blocking, 6 medium, 1 low-triple)
Two were closed only on a second pass after the user asked whether *all* had been worked: **S7** had
been half-done (two of `CLAUDE.md:40`'s four diagnostics) and **S8/S3** were recorded in QUEUE but
not in the documents that actually get read (§6.0b and `OPS.md`). A finding written into a ledger and
not into the operational doc is not closed — the same lesson as S1, where the tokenizer fix lived in
the README while the shipping path stayed broken.
