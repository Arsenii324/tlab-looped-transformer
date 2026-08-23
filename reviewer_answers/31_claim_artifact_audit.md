# Reply — 2026-08-23 23:35 · does the submission describe the model it ships?

*Audited by loading the artifacts, not by reading the prose. **No critical disagreement found**: every
field of the released checkpoint's own config matches the described architecture, and every headline
number traces to that checkpoint's own eval file. Three gaps are below, none of them a wrong number.*

---

## Verified clean — do not re-chase these

**The architecture the checkpoint carries is the architecture `METHOD.md` §1 describes.** All 17
fields in the checkpoint's saved `model_cfg`, checked one by one:

| field | checkpoint | described |
|---|---|---|
| hidden_size | 448 | 448 ✓ |
| n_heads / n_kv_heads | 4 / 2 | 4 / 2 ✓ |
| head_dim | 112 | 112 ✓ |
| intermediate_size | 1344 | 1344 ✓ |
| layers_per_loop | 3 | 3 ✓ |
| vocab_size | 4096 | 4096 ✓ |
| n_prelude / n_coda | 0 / 0 | "no prelude, no coda" ✓ |
| **state_renorm** | **False** | "no inter-loop normalisation" ✓ |
| inject_mode | `additive` | "additive re-injection" ✓ |
| depth_init | True | "`1/√(2·n_loop_eff)` output scaling at init" ✓ |
| n_loop_eff | 24 | 24, and **already flagged as a known defect** in §1 ✓ |
| rope_theta / rms_norm_eps / max_pos | 10000.0 / 1e-06 / 512 | ✓ |

**Every headline figure traces to `checkpoints/full_control90_kaggle/`**, recomputed from its own eval
JSON: CE@1 **3.9622**, best CE **3.6599 @ r = 10**, ppl **38.86**, bpb **1.5829**, loop gain
**0.3023**, band **[6,17]** at tol 0.01 on a dense 1..64 grid, tokens **89,999,360**, step 43,944.
Parameters **9,064,608**; `state_dict` sum **10,899,616**; difference exactly **1,835,008 = 4096 ×
448**. Reused block **7,228,704 = 79.7%**, matching §1 to the digit.

**The checkpoint is the dense control**: its `train_cfg` carries `supervise_k = 5`,
`min/max_train_loops = 4/32`, and **no `supervise_k_final` and no `supervise_switch_frac`** — so it
does not implement supervision annealing, exactly as `METHOD.md` §4 states.

**Not a defect, checked because it looked like one:** the saved `model_cfg` predates 16 `Config`
fields added later (`cond_mode`, `kv_window`, `depth_gate_mode`, `xsa`, `kv_untie_buckets`, …). They
load at their defaults, and **every one of those defaults is the "off" value** that matches the
description. The model card prints the full effective config, which is why it has more fields than
the checkpoint's own dict. Both describe the same model.

---

## 1. MED-HIGH — the model card never says these weights are *not* the recommended recipe

`METHOD.md` §2 recommends **supervision annealing** (`supervise_k_final = 1`,
`supervise_switch_frac = 0.90`). The shipped weights **do not implement it**, and `METHOD.md` §4 and
`RESULTS.md` both say so.

**The Hugging Face model card does not.** It explains why the control ships **instead of the
norm-penalty arm** (37.52 vs 38.86 perplexity, 88% loop-1 damage) — but it **never mentions
supervision annealing, and never describes the training recipe at all**. Grepping the card for
`supervis|schedule|recipe|terminal-only|U[4,32]` returns nothing.

**Why it costs something:** the card is the surface attached to the download. A grader who pulls the
weights, then reads `METHOD.md` §2's recommendation, has nothing at the artifact telling them the two
differ. Everything needed is already written in `METHOD.md` §4 — this is a propagation gap, not a new
claim. **Two sentences on the card closes it.**

## 2. MED — a checkpoint-name collision, and the obvious directory is the wrong model

Three checkpoints, and `train_cfg['run_name']` does not disambiguate them:

| directory | `run_name` | tokens |
|---|---|---|
| `full_control90_kaggle` | `no_state_renorm` | **89,999,360** — the shipped artifact |
| `full_no_state_renorm_kaggle` | `no_state_renorm` | **45,975,552** — the "46M" arm |
| `full_no_state_renorm` | `full_no_state_renorm` | **14,600,192** |

**Two different checkpoints share `run_name = 'no_state_renorm'`**, including the one that ships. And
`LIMITATIONS.md` §6b attributes the geometry and exit-rule claims to "**46M** `no_state_renorm`" —
where the directory literally named `full_no_state_renorm` holds a **14.6M** model, not the 46M one.

A reader following that label to the obvious path gets the wrong checkpoint. **The fix is to name the
directory** in §6b's table: `full_no_state_renorm_kaggle` for the 46M rows. *(The claims themselves
are correctly assigned — `EARLY_EXIT.md`'s scope line says "46M-token checkpoint", which matches
`full_no_state_renorm_kaggle` at 45,975,552. This is a labelling ambiguity, not a misattribution.)*

## 3. LOW-MED — §6b's annealing row is one arm behind

`LIMITATIONS.md` §6b: *"annealing's band result | **2.5M** (seeds 0–3) and **10M** (seed 2)"*.

The band series moved to **6 of 6** and there are now **two** 10M arms — `rec_sw90_s2` (§4.23e) and
`as_10M_sw90` (§4.17). The row reads as five. One clause.

---

## What a grader checking the artifact will find

They will find that the config matches, the numbers match, the parameter arithmetic is explained at
the artifact, and the identity gate passes against the download. **The one thing they will not find at
the artifact is that the shipped weights are the dense control rather than the recipe the method
section recommends** — and that is finding 1 above.
