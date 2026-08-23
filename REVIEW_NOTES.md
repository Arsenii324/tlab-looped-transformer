> **Dated working record — not a maintained document.** Last committed 2026-08-23; the text itself may be older. Kept intact rather than rewritten, because this project's rule is that superseded statements stay visible with their corrections. **Current numbers are in [`submission/`](submission/) and [`report.md`](report.md); where this file disagrees with them, they win.** See the repository README's *How to read this repository* table.

# REVIEW_NOTES — external reviewer's substantive claims, and my position on each

STATIC-ish doc. Written because these arrived as pasted-in messages that live only in conversation
context; a compaction would lose the reasoning and leave only one-line queue items. Each entry:
**what was claimed**, **whether I verified it**, **what I did**. Where I disagreed, the disagreement
and its grounds are recorded too — an unrecorded disagreement is indistinguishable from an oversight.

## Verified from source (I read the paper's own LaTeX)

**2606.24898 (Readout Blind Spot, Sharma & Vu) — CONFIRMED, and it costs us a novelty claim.**
Lemma 2 is exactly this report's §4.3 dilution account: for pre-norm `F(H)=H+B(Norm(H))`, `H=su`,
decomposing `b(u)=a_rad(u)·u+b_⊥(u)` gives scale update `‖F(su)‖ = s + a_rad + O(s⁻¹)` and direction
update `u + b_⊥/s + O(s⁻²)`. Lemma 1: a scale-invariant readout has `⟨∇_H L, H⟩ = 0`. §4.3 reframed
as **replication at 9M with an independent readout-space instrument**. Three things remain ours:
the contraction *refutation* with an instrument that passes a null; the *persistence* of direction
(`cos(du_t,du_{t−1}) → 0.9999` — Lemma 2 bounds step SIZE, says nothing about step AGREEMENT); and
the depth range (their entire study is K ≤ 4).
Their Table 2 answers my §6.9: **terminal-only loss beats per-loop** (44M: 5.40 vs 6.04; 129M: 4.88
vs 5.35) **but makes intermediate exits unusable** (Table 14: CE 5.52 at K=1 vs 1.54 at K=4). So
best-perplexity and a working exiter are not simultaneously achievable — that is a trade to state,
not a gap to close. Their Table 4 clamp is §4.6's experiment at one level and K=4, ΔCE
+0.0004…+0.0055; mine is **−0.012**, same magnitude **opposite sign** — corrected in §4.6.
Their depth table: ΔPPL K=1→4 is +0.01 (RMSNorm readout), −0.20 raw, −0.20 final-only, −0.22 norm
penalty; dynamic halting averages 1.00/2.16/1.78/2.60 loops. **That establishes scale control makes
loops 2–4 useful; it does NOT establish an extended depth RANGE** — nothing there runs past K=4.

**2607.27656 (SCSE) — CONFIRMED.** Names the quantity behind §4.3's injection tension: the
**zero-deviation forcing bias** `b_t(e) := T_t(0;e)`, the shared transition's response at an
input-conditioned anchor, generically nonzero under additive injection and able to be "contracted,
cancelled, exploited, or coherently accumulated" over depth. Our `inject_mode="none"` rollout is a
crude bias-subtraction counterfactual of that. It also claims Parcae/STARS/Loopus/Sharma&Vu do not
define or control this quantity.

**2606.04438 (IterMoE) — CONFIRMED, and it settles the §3.4 rule's one open case.** IterAdaLN is
`v_k = MLP_iter(PE(k))` — sinusoidal encoding of the iteration index through a learnable MLP, fused
with a projection of the token state. A **function of t, not a table over t**, so it is the only
published loop-conditioning mechanism that survives §3.4. Caveat: their own loop layer shows
monotonically increasing adjacent-pair cosine → fixed-point convergence, the regime §4.3 measures as
fatal to depth here.

**2607.15456 (LLA) — CONFIRMED, does NOT preempt §4.8.** It is a post-training cache *codec*
(compress the loop-indexed KV at matched budget), not a measurement of what a depth-k cache costs a
depth-t query. It contributes one decisive cell of our grid: **final-loop reuse buys 4× for free and
"collapses GSM8K generation to zero."** Also reports the value cache has higher cross-loop variance
than the key cache.

**2608.18230 (MixerLoop) — CONFIRMED with three caveats the summary dropped.** Real title
*"Allocating Recurrent Compute in Looped Language Models."* Beats FullLoop on aggregate CORE at 15M,
retains 41.5% at 110M, cuts recurrent-backbone projection FLOPs 45.9%, **iso-token** ("same data,
initialization, and architecture"). Caveats: mixer is **Gated DeltaNet** not softmax attention;
**T=4** loops; metric is **CORE downstream at 52.43B tokens** (~500× our budget). FLOP argument
transfers; performance claim transfers weakly. → §8.1.

**2605.23872 (Training-Free Looped Transformers) — read abstract.** Frames a pre-norm block as a
**forward Euler step**, so looping more with a smaller step is a finer integration of the SAME
trajectory. That unifies §4.6 (clamping changes step size → optimum moves, endpoint doesn't) with
item 5 (`ε=λ/(N√L)` holds total integration time constant by construction → predicts a null). Their
setting is training-free retrofit onto frozen 4B+ checkpoints; the framing transfers, the numbers
don't.

## Accepted and acted on, not independently verified

- **eval-at-T vs train-at-L.** The reviewer's sharpest structural point: everything in this repo
  until 2026-08-22 was *train once, sweep inference depth*. The task's "чем больше, тем лучше" reads
  as train-at-L. LoopFormer reportedly monotone over L∈{8,12,24}; SCSE degrades over T=8..48. They
  are not disagreeing — different curves. → §4.9, sweep running.
- **Loopie's own §"Why Only Two Loop Steps?"** quote: R=2 is a FLOP-allocation decision, and it
  explicitly names "where inference-time computation is cheap" and "where adaptive computation is
  available" as settings where larger R may pay — i.e. this task. → §2.
- **FineWeb vs FineWeb-Edu** incomparability, and the 92M+6M=98M-of-100M budget gap. → §2.1.
- **Nobody has shown monotone improvement past ~8 loops at <150M on AR text** (Ouro ~4, STARS 4,
  LoopMTP turns at 7, SCSE degrades from T=8). LoopMDM is the one positive and it is masked diffusion
  with a mask-token workspace. → §8.0.
- **AttnRes (2603.15031)** independently diagnoses PreNorm dilution in unshared stacks and reports
  softmax depth-mixing bounds output magnitudes. **2606.22325** finds that routing collapses onto ~2
  hubs (top weight 0.643 vs 0.245 uniform). → §8.2 trilemma + the caveat.
- **Muon**: strong prior in the recurrent setting (McLeish et al. report AdamW NaN spikes removed),
  but needs its own LR sweep — WONTFIX under deadline. The kept idea: log the **singular-value
  spectrum of the accumulated gradient** for a looped projection vs the same projection untied. If
  the looped spectrum decays much faster, orthogonalization is qualitatively different here. Nobody
  has that figure.
- **Per-token argmin-depth novelty**: existing work (CALM, D³ 2503.08524, TIDE 2603.21365,
  2607.14427) measures *saturation* depth = earliest layer agreeing with the FINAL layer. Past the
  optimum the final layer is not the target, so argmin-CE depth is a distinct object. Priors that it
  is NOT a null: HELIOS finds 74%/5%/21% dispersion in OPT-1.3B; AdaPonderLM's learned gates
  allocate more compute to high-NLL tokens. → §4.7, flagged as a claim to check.

## Where I disagreed, and why

- **"Cut the compute-matched non-looped baseline."** Declined. It is a documented negative result;
  §6 already states the MPS caveat plainly. Deleting an inconvenient finding to reduce attack
  surface is the wrong instinct — the honest fix is the caveat, already written.
- **"You will not build a sandwich envelope."** Moot — §4.5 already built and measured it
  (parameter-, token- and compute-matched), which is how the prelude/coda double dissociation exists.
- **Custom Docker image priced at 1–2h.** I was WRONG initially (assumed build-and-push); ccm-intro
  §6 shows it is an inline public image tag. But the follow-up AI's "80% wall-time saving" generalizes
  my *trivial probe* to *4h jobs* where the saving is 1.7%, and its claim that "container pull is
  already cached" is false for a *custom* image on first pull.
- **Supervision-density hypothesis (strong form).** Killed by measurement: density is monotonically
  decreasing (d(1)=.337 → d(32)=.035), peaking at 1–4 while the optimum is 8–11. The *schedule*
  handle survives and is much stronger — §4.11 shows optimum tracks μ_rec (6/18/28 → 4/8/16) with
  loop gain scaling 4.6×.

## My own considerations, recorded so they are not re-derived
- The recurring pattern, now seen twice independently (§4.5 prelude, §4.11 schedule): **the config
  that wins absolute CE is not the config that makes loops matter.** If this holds at full budget the
  report must say which the task rewards rather than pretending one config wins both.
- §4.6 + §4.11 together: saturation is **demand-side**. Supply-side (scale) interventions relocate
  the optimum without raising the ceiling; the training schedule moves both the optimum and the gain.
- Therefore the deliverable-optimal config is plausibly **full budget at a fixed large L**, not the
  randomized 4–32 schedule the current headline uses. That is the 04:30 decision.

## Papers surfaced but NOT yet read (recorded so they are not lost)

Ranked by bearing on open questions. None of these has been read; do not cite from this list.

1. **Attractor Models / "Solve the Loop" (2605.12466)** — fixed-point solver with implicit
   differentiation, O(1) training memory in effective depth, convergence-adaptive iteration count.
   Claims PPL up to 46.6% better and a 770M beating a 1.3B trained on 2x tokens. **Directly
   challenges this task's premise** that convergence makes further computation pointless — §4.3
   measures contraction as fatal to depth, and this claims fixed-point convergence as the win.
   **Critically: FPRM (2606.18206) states they could not reproduce the Attractor results on
   Maze-Hard.** A reproducibility flag from a credentialed adjacent group belongs in related work.
2. **Recirculation (2608.17981, Mozer et al.)** — inference-time recurrence for belief-state
   tracking, explicitly positioned *against* depth-recurrence looping, claiming perplexity gains at
   ~no generation latency.
3. **"Chain-of-Thought and Compressed Looped Transformers: A Memory-Budget Separation"** — argues a
   compressed looped transformer is bounded by its persistent recurrent-state budget *regardless of
   loop count*. That is a **capacity-theoretic ceiling** distinct from every dynamics-based failure
   mode measured here, and it would be a competing explanation for §4.6's invariant ceiling.
4. **"Looped Transformers with LayerNorm Provably Learn the Power Method"** (Wu, Zhang, Cao) — a
   proof about which dynamics LN-in-loop selects, i.e. convergent-by-construction. Bears directly on
   §4.3's finding that `state_renorm=True` contracts to a fixed point by loop ~16.
5. **Asymmetric Input Recurrence** (Wang, Li, Zhang, Huang, Yan, Li) — input-injection asymmetry in
   a two-state shared-weight recurrent transformer. Sits on §4.3's unresolved injection territory.
6. **Sapunov, "Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive
   Reasoning" (2604.21999)**.

## Optimizer choice — settled, keep AdamW

Audited across tarballs rather than by impression. **Muon+AdamW:** Parcae (Muon LR fixed 8e-3,
5 polar-express iters, AdamW swept 2e-3..8e-3), LoopMTP (peak 1.9e-3), Schwethelm, McLeish (Muon
specifically because AdamW spiked to NaN on recurrent-depth training). **AdamW only:** Sharma & Vu
(3e-4 at 44M/129M/1.4B), residual scaling 2606.18524, SCSE (3e-4 at 22M), STARS, **Loopie (3e-4 at
20B-A2B)**. So it splits by *culture* (nanochat/speedrun lineage vs academic mechanism studies), not
by scale. **Decision: keep AdamW.** The paper this project most directly replicates — Sharma & Vu —
used AdamW 3e-4, so our optimizer is matched to the comparison that carries the mechanism claim.
The one looped-specific failure Muon is documented to fix (AdamW NaN spikes at ~1B) is not a failure
we have. Kept idea, unrun: the singular-value spectrum of the accumulated gradient for a looped
projection vs the same projection untied.

## Eval protocol — audited 2026-08-23

No off-by-one: position 0 predicts token 1 from token 0, i.e. 1 token of context, not zero. Average
left context is **128.5 tokens (~429 bytes)**; **25.7% of frozen-set windows span a document
boundary** (standard GPT-2-style packing). Within-project comparisons are unaffected (protocol fixed
and identical across arms). The ABSOLUTE bpb is deflated by context starvation, so
`src/sliding_eval.py` reports a stride-64 sliding-window number alongside the chunked one, both
labelled. Parameter Golf measured this protocol effect at ~0.032 BPB.

## Prediction on the cross-depth (k,t) grid, recorded BEFORE it lands

From measurements already in hand: total rotation from loop 8 to 64 is ~0.41 rad; `‖norm1(h)‖` is
flat (25.13 → 21.36) and RMSNorm is direction-only, so **keys barely change with depth**; `‖v‖`
*falls* 82.9 → 39.6. Therefore: **the matrix should be near-flat for k, t ≥ ~8, with damage
concentrated in the small-k rows.** The smoke test on `center` already shows that shape (k=1 row:
7.62, 7.71 vs ~6.75 elsewhere). If it holds, the same 1/t dilution that kills depth utility is also
what makes a mixed-depth cache safe — one mechanism, two consequences. Note the asymmetry: since
‖v‖ falls with depth, a *shallow*-exited token contributes the LARGER value vector, the opposite of
the "deep tokens dominate" story already refuted on pre-norm grounds in §4.3.

## Huginn (2502.05171) verified from source — and it supplies the context-scaling argument

The report's "~10 in Huginn" citation is **accurate**: verbatim, *"without few-shot examples to
consider, the model saturates in compute around 8-12 iterations."* But the sentence continues and
the continuation is the valuable part: *"saturating around 20 iterations if 1 example is provided,
and 32 iterations, if 25-50 examples are provided"*, and *"saturation is highly task-dependent."*

**Saturation depth scales ~3x with available context.** Consequences recorded in report §8.0b:
(a) our seq_len 256 / ~128.5-token average left context sits in their "saturates early" regime, and
our optimum at 8-11 matches their zero-shot 8-12 almost exactly; (b) seq_len is therefore a *cap on
what depth can be for*, not a minor hyperparameter, which promotes §6.5 from a footnote to a
first-order limitation; (c) it makes a free, sharp prediction for `argmin_anatomy.py` — argmin depth
bucketed by position-in-chunk should RISE if depth demand is context-driven. Pre-registered.
Also confirmed from their text: input injection at every step plus random latent init is described
as stabilizing recurrence and promoting "convergence to a steady state independent of
initialization" — i.e. Huginn deliberately targets the contracting regime that §4.3 measures as
fatal to depth here. Worth one line in related work.
