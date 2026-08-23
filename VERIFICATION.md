# VERIFICATION LEDGER — every external claim, its status, and where it is used

STATIC doc. Rule: **nothing quoted in `report.md` may sit at UNVERIFIED without a hedge in the text.**
Status: VERIFIED (read in the paper's own source) / SECOND-HAND (relayed, not checked) /
REFUTED / N-A (not used in the report).

## VERIFIED from the paper's own LaTeX

| claim | source | check |
|---|---|---|
| Dilution mechanism = Lemma 2: scale `s + a_rad + O(s⁻¹)`, direction `u + b_⊥/s + O(s⁻²)` | 2606.24898 | quoted verbatim; §4.3 reframed to replication |
| Lemma 1: scale-invariant readout ⇒ `⟨∇_H L, H⟩ = 0` | 2606.24898 | verbatim |
| Clamp ΔCE +0.0004…+0.0055 at K=4, one level | 2606.24898 Tab.4 | verbatim; my −0.012 is **opposite sign**, stated in §4.6 |
| ΔPPL K=1→4: +0.01 RMSNorm / −0.20 raw / −0.20 final-only / −0.22 penalty; halting 1.00/2.16/1.78/2.60 | 2606.24898 | verbatim; used to bound "recovers loops 2–4, not the range" |
| Terminal-only beats per-loop: 5.40 vs 6.04 (44M), 4.88 vs 5.35 (129M) | 2606.24898 Tab.2 | verbatim; motivates the queued `supervision_depth` arm |
| Huginn "sliders": trajectory "drifts in a single direction… count how many iterations have occurred" | 2502.05171 | verbatim; §4.3 |
| Huginn saturation 8–12 zero-shot → 20 (1 example) → 32 (25–50 examples) | 2502.05171 | verbatim; §2, §8.0b |
| Huginn k=8 compared to **randomized k, never full BPTT** | 2502.05171 | verbatim; §3.2 |
| Huginn init `σ²_out = 1/(5hl)`, `l = l_P + r̄·l_R + l_C` (MEAN recurrence, layer applications) | 2502.05171 | verbatim; resolves §6.4 |
| Ouro R=4 is a **stability** decision taken down **from 8**; "monotonic improvement from 1 to 4 rounds" | 2510.25741 | verbatim; §2 |
| Loopie: R=2 is a FLOP-allocation decision; "larger R may be useful where inference-time computation is cheap" | 2607.16051 | verbatim; §2 |
| MixerLoop NLL: NoLoop/Mixer/**Full** = 2.995/2.946/**2.936** (15M), 2.401/2.377/**2.342** (110M) | 2608.18230 | verbatim; **REFUTED my §8.1 draft**, corrected |
| MixerLoop iso-token ("same data, initialization, and architecture"), T=4, GDN mixer, 52.43B tokens | 2608.18230 | verbatim |
| IterAdaLN = `v_k = MLP_iter(PE(k))` — a function of k, not a table | 2606.04438 | verbatim; §3.4 |
| LoopMoE removes re-injection: "a constant h_pre… suppressing iteration-to-iteration differentiation" | 2606.04438 | verbatim; §5 |
| LLA is a cache **codec**; final-loop reuse "collapses GSM8K generation to zero" | 2607.15456 | verbatim; §4.8 framing |
| SCSE 50M: baseline 151.1→178.9; step-cond 125.7→160.1; SCSE 123.1→135.5→156.4 (T=8/24/48) | 2607.27656 | **all five figures found in source**; §8.0 |
| SCSE zero-deviation forcing bias `b_t(e) := T_t(0;e)` | 2607.27656 | verbatim; §4.3 |
| STARS peaks at 4 recurrents (74.18), 8 gives 65.55; "performance does not always improve monotonically with more steps" | 2605.26733 | **both numbers + quote found**; §8.0 |
| LoopMTP BPB math 0.6575(T=7)→0.6586(T=9), code 0.7822→0.8003, QA 0.9541/0.9366; "We leave the analysis of looping limits and upper bounds to future work" | 2608.03624 | **all found + quote**; §8.0 |
| Residual scaling: LM transfer validated at `N∈{1,2,4,8}` on FineWeb-Edu 10B, L=12/24/48 | 2606.18524 | verbatim. *Refinement:* a separate scaling comparison sweeps `N∈{1,2,4,8,16,32,64}`; the **LM** claim is N≤8 |
| PALBERT: `λ_i = Λ([h_i, h_{i−1}])`, explicitly replacing PonderNet's single-state MLP; q=0.5; weights shared across layers | 2204.03276 | verbatim; **caught my qexit in PonderNet's weaker configuration**, fixed |
| PALBERT: "Q-exit stands for Quantile"; rationale "no ability to estimate mean or argmax… without running all layers" | 2204.03276 | verbatim |
| LTO: latent classifier predicts correctness "even for partial trajectories with just the first few thinking steps" | 2509.26314 | verbatim; §4.7 |
| LTO: without regularization "may exploit the errors of the LRM" | 2509.26314 | verbatim; the selection-on-noise caveat |
| Training-Free Looped: pre-norm block as a **forward Euler step**, damped sub-steps | 2605.23872 | abstract read; §4.6/item-5 synthesis |

## MY OWN numbers: what resolution they have (added 2026-08-23)

This ledger tracks *external* claims. The symmetric question for my own measurements is answered in
report §4.15 and belongs here as a pointer, because it gates how any row below may be compared
against a number I measured:

- **Measured floor (MPS runs only):** two runs of an identical config at an identical seed end
  **0.031–0.068 nats** apart. Fixed seed is not a replicate. Covers §4.1, §4.5, §4.11, §4.14,
  `residual_scale`.
- **Inferred floor (CUDA/DataSphere runs):** ~0.06 nats, from the non-monotonicity of §4.10's
  fixed-`g` sweep. **Not** a replicate measurement. Covers §4.9, §4.10, §4.13, §4.4.
- **Working rule:** a single-arm difference under ~0.05 nats is not a result on either device. Any
  comparison between a paper's reported delta and mine must clear that bar before the signs are
  called agreeing or disagreeing.
- **argmin is retired** as a statistic: 52 of 71 stored loop curves have argmin margins under 0.005
  nats (`src/argmin_audit.py`). Depth claims are stated as plateaus (`src/plateau.py`).

## VERIFIED 2026-08-23 — prior art on my own headline mechanism, checked from source

Surfaced by an external reviewer and **verified against the papers' own text before use**, because
it bears directly on what §4.9 and §4.14 may claim.

| claim | source | check |
|---|---|---|
| *"the looped transformer consistently discovers a fixed-point solution that saturates prior to the trained iteration b"*, and this occurs *"due to the loss objective, which requires the looped transformer to match the target within b steps"* | 2311.12424, *Looped Transformers are Better at Learning Learning Algorithms* (ICLR 2024) | **verbatim, found in source.** Setting is **in-context data-fitting/regression**, not LM pretraining. Their loss is windowed over iterations `t ∈ [b₀, b]` with `b₀ = max(b−T, 0)`, i.e. a **truncated loss window T** — structurally the same knob as this report's `supervise_k`. |
| *"deeper iterations serve a different objective: they refine the first iteration's prediction rather than predicting further ahead to the next-next token"* | 2511.08577, *Think-at-Hard* | verbatim |
| *"recurrent transformers must accommodate both objectives with shared weights, potentially limiting performance"* | 2511.08577 | verbatim |
| *"we apply a LoRA adapter to the shared LLM backbone only for iterations d>1"* | 2511.08577 | verbatim |
| *"over 73\% of next-tokens are correctly predicted at the first iteration"* | 2511.08577 | **VERIFIED verbatim in the v3 LaTeX** (`3_method.tex` line 206). **RETRACTION: I previously marked this "REFUTED — the paper says 85%". That was my error, not the relay's.** I had checked only the arXiv *HTML* through a summarising fetch, which returned 85% — a figure that appears in the source only as unrelated table cells in the experiments section. The relayed 73% was correct all along. |

**What this does to this report's claims, stated rather than buried.** The *mechanism* — that the
objective determines where the optimum lands, below trained depth — is **prior art (2024)**, and
§4.9 must not present it as new. What this report adds on top of it, and what I have not found
reported anywhere: (i) the **ratio is measurable and stable** at LM-pretraining scale — dense
supervision puts the useful-depth midpoint at 0.50–0.71 of trained depth, terminal-only at 0.98–1.09,
across three schedules and two devices; (ii) supervision density is a **threshold at k=1**, not a
dial (§4.16); (iii) the location can be **annealed in time**, recovering the depth shift at near-zero
CE cost (§4.17). Think-at-Hard reaches the adjacent diagnosis (two objectives, shared weights) and
fixes it with **depth-specific parameters** (a LoRA adapter for d>1); §4.17's answer is a
**zero-parameter schedule** on the same problem, which is a different point in the design space and
is worth stating beside theirs rather than instead of it.

## SECOND-HAND — relayed, NOT checked against source. Must be hedged wherever used.

| claim | where used | risk |
|---|---|---|
| **Qwen2.5 retrofit (2608.11233): intermediate-step supervision followed by "outcome-only annealing"** | **§4.17 — prior art on THIS REPORT'S OWN METHOD** | **relayed 2026-08-23, source not obtainable.** Reported as a fine-tuning recipe on ARC, not LM pretraining. **This is the closest thing to precedent for the annealing ingredient and it is recorded here rather than omitted** |
| **LARM (2606.04678): static sparse supervision with a checkpoint interval** | §4.17, same | relayed; reported as an ASR encoder. Sparse-but-static, i.e. §4.16's k>1 regime rather than an anneal |
| Inner Loop Inference (2602.14759): noise ablation "consistently underperforms the structured looping variants" | §4.13, hedged | **relayed 2026-08-23, source not obtainable.** Same direction as my result |
| LoopRPT (2603.19714): "disabling Gaussian noise reduces accuracy… stochastic latent rollouts facilitate exploration" | §4.13, hedged | relayed; **opposite** direction to mine — RL post-training, different regime |
| SPRM (2604.18839): noise helps "in small-data regimes" but not "in data-rich settings" | §4.13, hedged | relayed; this is the *predictive* account that reconciles the other two with §4.13 |
| Two-Scale Latent Dynamics (2509.23314): looped updates become "small and increasingly orthogonal, tracing a stable-curvature spiral" | §4.3 contrast, hedged | relayed. **Apparently contradicts §4.3's cos → 0.9999 (near-parallel).** Most likely a different regime (their setting presumably retains inter-loop normalisation, mine does not) — but unverified, so stated as a possible contradiction rather than a resolved one |

**All four were relayed on 2026-08-23 and none is in `papers/sources/`.** After §6.0 row 22 — where I
asserted a "correction" to a relayed number from a summarising web fetch and was wrong — the standing
rule is that a citation is VERIFIED only against the paper's own LaTeX. These are used as hedged
attributions, never as verbatim quotes, and no claim in this report rests on them.


| claim | where used | risk |
|---|---|---|
| **AttnRes (2603.15031)**: "mitigates PreNorm dilution: output magnitudes remain bounded across depth" | §8.2, quoted | **tarball not obtained**; quote is relayed. Hedge or drop. |
| **2606.22325**: depth routing collapses, top source 0.643 vs 0.245 uniform, two hubs | §8.2, quoted | **not obtained**; load-bearing caveat for the depth-mixing proposal |
| D³ (2503.08524), TIDE (2603.21365), 2607.14427 define per-token depth as **saturation depth** | §4.7 novelty argument | not obtained. The **structural argument** is what carries §4.7, not the citation list |
| HELIOS: OPT-1.3B, 74% exit after layer 6, 5% after 12, 21% need all 24 | REVIEW_NOTES only | not in report; fine |
| Done Right: MATH500 +12.00, DROP +2.61; random init 2 gains / 4 losses | REVIEW_NOTES only | not in report; fine |
| LoopFormer: L∈{8,12,24} monotone, iso-token 25B Pile; "naive early exiting… stagnant representations" | §4.7 caveat, §4.9 framing | tarball extracted but numbers not verified |
| Parcae: Muon LR 8e-3 fixed, μ_rec∈{2..12}, smallest 140M | REVIEW_NOTES | not in report |

## Action taken
Every SECOND-HAND row that appears in `report.md` is being hedged in place (next commit): AttnRes and
2606.22325 are marked as relayed-not-verified at the point of use, and §4.7's novelty argument is
restated so it rests on the structural claim (past the optimum the final layer is not the target)
rather than on an exhaustive literature sweep.

## 2606.20075 — VERIFIED FROM SOURCE 2026-08-23 (was blocked/unobtainable)
Tarball arrived in ~/Downloads mid-session; extracted to `papers/sources/2606.20075/`.
Title (from `icml_latex.tex`): *"What Makes Effective Supervision in Latent Chain-of-Thought: An
Information-Theoretic Analysis"*. Numbers read from `Tables/1_table_sec4.tex`, quotes from
`Sections/1_Intro.tex` and `Sections/4.tex`:
- `OS-No-CoT` 18.7 / `OS-Latent` **9.8 / 18.3** / `OS-GC` 13.1 / `OS-GR` 18.2 / `Explicit CoT` 43.1
- *"outcome supervision alone is insufficient to induce meaningful latent CoT steps"* (Intro)
- `OS-GR` = *"a specialized decoder to recover the original discrete token from the continuous hidden
  states... preserving semantic information without enforcing strict geometric conformity"* (§4)
Used in §4.18. **No number is quoted that was not read in the source** — the standing rule after the
73%/85% retraction (§6.0 row 22).

## 2604.11791 — VERIFIED FROM SOURCE 2026-08-23 (arrived mid-session)
*A Mechanistic Analysis of Looped Language Models.* Extracted to `papers/sources/2604.11791/`.
Quote from `sections/appendices/additional_soi_results.tex` line 104; macros resolved from
`looped_llms.tex` lines 553-555 (`\ouro`→Ouro, `\raven`→Huginn-0125, `\rllama`→Retrofitted Llama).
- **Causal ablation**: zeroing the layer-2 MLP output responsible for massive activations in the
  Retrofitted Llama removes its stages of inference. Used in §4.1 as a mechanism for −0.744 nats.
- **Claimed contradiction CHECKED AND REJECTED.** A reviewer flagged line 49 (post-block residual
  normalisation attributed to Ouro) as contradicting line 104 (Huginn's lack of stages attributed to
  normalisation). It does not: line 49's contrast is with the *retrofitted* models — "This is not the
  case for the retrofitted series of models, which lack this norm" — not with Huginn. Both Huginn and
  Ouro normalise. Verified by reading both lines with macros resolved rather than by relay.

## 2607.27656 (SCSE) — anchor-response construction verified 2026-08-23
`lvr.tex` §131-137: anchor `h*(e)`, deviation `Δ_t = h_t − h*`, zero-deviation forcing bias
`b_t(e) := 𝒯_t(0;e)`; "harmful, neutral, or beneficial depending on the readout and loss" — so they
do NOT claim the bias causes saturation. Implemented as `src/anchor_response.py`, §4.3.

## 2607.10681 — obtained, NOT yet used
*LayerNorm as Implicit Gain Control in Looped Transformers.* Extracted; relayed as claiming LayerNorm
scales the Jacobian inversely with activation norm. **The specific claim has NOT been located in the
source yet**, so nothing in the report cites it. Logged as obtained-but-unverified.

## Citation audit, 2026-08-23 ~18:00 — every arXiv id in report.md vs papers/sources/
`agy` job C (independent citation cross-check) never ran; this is the cheap substitute, done directly.

- **28 arXiv-shaped strings cited**; 2 are regex false positives (`2567.9355`, `3675.1917` are not
  arXiv ids — they appear in other contexts).
- **15 verifiable from tarball on disk**: 2311.12424, 2502.05171, 2509.26314, 2510.25741, 2511.08577,
  2604.11791, 2605.23872, 2606.04438, 2606.18524, 2606.20075, 2606.24898, 2607.15456, 2607.16051,
  2607.27656, 2608.18230.
- **11 NOT on disk**: 2503.08524, 2602.14759, 2603.15031, 2603.19714, 2603.21365, 2604.18839,
  2605.09165, 2606.04678, 2606.22325, 2607.14427, 2608.11233. **All 11 are already flagged
  second-hand / relayed / unverified within ±6 lines of their citation** — checked mechanically, not
  by memory. No claim in this report rests on an unflagged unverifiable source.

**2511.08577 (Think-at-Hard) re-verified from source**, since §6.0 row 22's retraction was about this
exact paper: `3_method.tex:206` reads *"over 73\% of next-tokens are correctly predicted at the first
iteration"* — the reviewer's figure, not the 85% a summarising web fetch produced. The report's other
quote from it is also verbatim at `3_method.tex:207`.

**2608.09444** (relayed claim that heterogeneous exit depths create OOD attention) is **not cited
anywhere in report.md**, so there is nothing to flag. Had it been used, the reconciliation with
§4.8/§4.8b would be: our ragged cache is safe *because* the per-depth states are nearly identical in
the directions attention reads — which is the same dilution (§4.3) that kills depth utility. That
unifies §4.3 and §4.8 rather than threatening either.
