# Reply — 2026-08-23 19:50 · the negative generalises, and it is weight tying

Three results since the last update, one of which is the strongest thing in the project. Every number
re-derived from artifacts; every confound stated in the same paragraph as the claim.

---

## 1. The depth-key rank collapse is WEIGHT TYING, not smallness — 11.7×

**The objection this answers was the right one to raise**, and it came from an independent pass over
today's work, not from me: 448 hidden units with 4 heads and 3 distinct layers is not much to
differentiate with, so perhaps §4.7e's rank ~1.6/32 is a *small-model* artifact rather than a
statement about looping. That distinction decides how far the whole negative reaches.

**Both models untrained. Identical hidden size, heads, head_dim, initialisation. 33 depths each.**
Training quality therefore cannot explain the difference and the **only** variable is tied-vs-untied:

| | effective rank | mean pairwise cos |
|---|---|---|
| **weight-tied** — one block applied 33 times | **2.73 / 33** | **0.8022** |
| **untied** — 33 distinct layers, same width | **31.80 / 33** | **−0.0029** |

**An untied stack at identical scale has essentially full-rank, near-orthogonal depth keys. The tied
loop has 2.7 of 33.**

**Three consequences, and the second is the one I did not expect.**

1. **§4.7e's negative generalises.** It is not a 9M-parameter artifact; it is a property of the
   mechanism this entire report is about. The depth-mixing family is closed for weight-tied looped
   models *at any size*, not just this one.
2. **MoD-Attention's published positive is EXPLAINED rather than contradicted.** They run 24 and 48
   **unshared** layers — precisely the near-orthogonal key set measured above — so same-position depth
   attention has something real to attend over. **Their gain is a property of distinct layers, and
   that is now measured rather than argued.** Two results that looked like they were in tension are
   one mechanism seen from two sides.
3. **It sharpens the trade the report keeps circling.** Weight tying buys parameter efficiency and
   **pays for it in depth distinguishability**. The same block applied twice cannot produce two
   different views, and every depth-mixing mechanism needs exactly that.

*Cost: one forward pass each, no training. Scope: one width, one depth count — the ratio at other
widths is untested, though the mechanism (identical weights produce identical maps) does not obviously
depend on width. `src/depth_key_rank.py::tied_vs_untied` reproduces it.*

## 2. Capacity vs diversity: resolved, ~65% capacity, from two independent lines

You were right that pinning to branch **0** could not answer this — branch 0 is the only branch
trained at `r = 1`, exactly where 88–95% of the effect lives. `tlab-pin2-control-s0` pins to branch
**2**, which never trains at `r = 1`: identical parameter count (9,473,184), zero diversity.

**Final, step 1219, complete:**

| arm | best CE | ΔCE_best | band |
|---|---|---|---|
| `pin_control_s0` | 5.3052 | — | [8,20] mid 12.6 |
| `pin_lora_b2_s0` | 5.2237 | **−0.0815** | **[8,16] mid 11.3** |
| *(cycled `dv_lora_r4_s0`, other job)* | 5.2514 | *−0.1251* | *[8,20] mid 12.6* |

**A single fixed branch with zero diversity recovers ~65% of the cycled arm's CE gain.** Diversity's
own contribution is **~0.044**, which sits only just above the measured cross-job drift band
(0.0074–0.0334) — and this is a **cross-job difference-of-differences at one seed**, so that residual
is at the edge of what the design can resolve.

**It agrees with a completely independent argument**, which is what makes it worth stating: §4.21b
found 88–95% of every LoRA arm's gain already present at `r = 1`, where cycling is *logically inert*
(verified: pinned and cycled give max|diff| = 0.000e+00 at r=1, 1.05 at r=4). One line from where the
gain sits on the depth curve, one from an explicit zero-diversity control. **§4.21 is a capacity
result.**

**One asymmetry I am flagging rather than smoothing:** the cycled arm keeps its control's band
([8,20] → [8,20]) while the pinned arm **narrows** it ([8,20] → [8,16]). If that survives replication,
capacity buys CE *at the cost of band* while diversity preserves it. Single seed, cross-job, one grid
point — an observation, not a result.

**Also, the LoRA r ≥ 4 set is now n = 5** (`dv_lora_r4` adds −0.1251, a third platform): mean
**−0.0936**, 95% CI **[−0.1319, −0.0554]**, excluding zero. **Including rank 2 (n=6) the interval
still covers zero** ([−0.1478, +0.0231]) — so the post-hoc restriction still decides significance,
exactly as before.

## 3. Duo-causal W=2: complete at both seeds, clean null — and it corrects an interim I gave you

| seed | ΔCE_best | ΔCE@1 | onset | end | mid |
|---|---|---|---|---|---|
| 0 | **+0.0093** | +0.0226 | 8 → 8 | 20 → 20 | 12.6 → 12.6 |
| 1 | **−0.0115** | −0.0221 | 8 → 8 | 20 → 20 | 12.6 → 12.6 |

Sign reverses, both inside the 0.0150 floor, **band identical to the digit**. The registered falsifier
fires.

**Against myself:** at 18:52 I told you this arm was "tracking negative at both seeds" (+0.1632,
+0.0360) and used it to argue against V100 spend. That was the **first eval at 500k tokens**, where
§4.12 says loop gain barely exists. **The arms converged to a null.** The recommendation survives, for
a different reason than I gave.

**Still unsettled per the 19:02 gate:** `cos(du_t, du_{t−1})` needs the returned checkpoints. A CE
null without the cosine is a null on *a mechanism that may not have engaged*.

## 4. Two process results worth more than they look

**§6.0 row 34's fix is validated on a completed job.** `pin2` returned **3 files, 70.8 MB — including
both `.pt` checkpoints.** The explicit per-filename `outputs:` works where this morning's glob
returned `results.json` alone and cost the depth-gate weights.

**An independent read of today's ~2,000 new report lines found 11 defects, 2 of which would have
changed a reader's conclusion.** Both were mine, both are fixed, and one is the 17:50 pattern
repeating: §4.7e's mechanism paragraph correctly labelled itself post hoc **and then claimed the
running arms supported it** — citing the interim above, which the completed arms contradict. The
other: **§8 carried neither deflation** that §4.21 and the abstract both carry, so a reader stopping
at the synthesis got the unqualified version. Also caught: a broken blockquote *my own earlier edit*
created, stranding three supporting sentences inside the caveat that undercuts them.

**And a trap now seen twice in one hour, which makes it a rule:** partial arms in a live log read
against a completed control produce spectacular fake numbers — `dc_w3` showed **+1.117**,
`dv_lora_fixed0` showed **+0.8985**. Both are arms with 1 eval against controls with 5–7. Excluded by
checking eval counts, not by noticing the number looked wrong.

## 5. Running now

| job | state |
|---|---|
| `tlab-duocausal-s0/-s1` | W=3 at 850/1708 both seeds; `dg_norm` still to come (~20:30) |
| `tlab-diversity-control-s0` | pin-0 arm at 450/1220 (~20:05) |
| `tlab-xsa-s0` | XSA arm at 200/1220 (~20:10) |
| `tlab-recmethod-s2` | 2150/4882 (~20:50) |
| Kaggle `tlab-lora-scaleup` | RUNNING, 12M/arm (~21:45) |

**`dg_norm` remains the highest-stakes cell**, and §4.7e now makes it a *joint* test registered at
19:22: mixing engaging (≥1.5) **with no gain** confirms the rank collapse as the binding constraint; a
real gain means **§4.7e is wrong and the per-token headroom is reachable.**

## 6. Also new: `submission/`

`report.md` is 6,400+ lines and is **evidence, not reading material**. A separate jury-readable folder
now exists — `README`, `EXPERIMENTS` (all **113** trained arms, generated from artifacts by
`src/make_inventory.py`), `NEGATIVE_RESULTS`, `SCALE`, `FAILURES`. `METHOD.md` and `RESULTS.md` are
deliberately **pending** until the six arms land, rather than written twice.

A coverage check fell out of it: of 113 arms, **7 appear in `report.md` by neither identifier nor
value** — the five `trainL*_s1` seed-1 arms (§4.9 reports only the re-zeroed mean curve), plus
`as_10M_sw90` and `sc_final_only_s1` (reported by delta only). **No experiment went unreported;
several numbers did.**

## 7. What I would still attack

- **§4.7e's causal direction.** Depths could collapse *because* the loss never rewarded distinguishing
  them, rather than mixing failing *because* they collapsed. The tied-vs-untied result narrows this —
  both models are untrained, so the loss cannot be the cause of the *tied* collapse — but it does not
  eliminate it for the trained case.
- **Diversity's ~0.044 is at the edge of the design's resolution**, cross-job, one seed.
- **Everything tonight is 2.5–3.5M tokens** against this project's own 12×-shrinkage regularity. The
  Kaggle 12M arm is the only scale check and it is one arm, one seed.
