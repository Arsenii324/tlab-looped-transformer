# Reply — 2026-08-23 22:30 · everything in this submission that can be questioned

*An anticipatory critique of our own work, written so a reviewer does not have to find these. Ordered
by how much it would cost us if it landed. §6 is the one I would read first if I were reviewing: the
numbers where **recomputing from the shipped artifacts gives a different answer than the report**, and
why each is legitimate.*

---

## 1. The attacks that would land hardest

**(a) "Your central finding is measured only where it is cheapest to be true."**
The spine is *every loss-lowering intervention delivers 67–101% of its gain at a single loop*. **Every
arm showing it sits at 2.5–3.5M tokens** — 3–4% of the headline budget — and loop gain roughly
**triples** by 90M (median 0.1084 at ≤3M over 104 arms, 0.3023 for the 90M control). So the share is
measured where the denominator is smallest. **We had one probe. It landed tonight and it cannot answer
the question**: at 12M there is no gain left to decompose (§4.29). *The budget-invariance of the
central pattern is untested and now has no probe at all.* **This is the strongest attack on the
submission and we cannot answer it.**

**(b) "Your one positive that replicated across platforms died at 5× the budget."**
Correct, and we say so. Loop-cycled LoRA: **−0.0936** across five arms and three platforms at 2.5M,
**+0.0077** at 12M in a **config-identical** pair. **This project has no replicated CE improvement at
scale.** *Bounded honestly: rank 8 and LoRA × annealing were never tested at 12M in their own form.*

**(c) "Your instrument had an unswept free parameter for the entire project."**
`plateau(curve, tol)` replaced `argmin` for being decided within noise — and `tol = 0.01` was set once
and never varied. **It is tighter than the measured 0.0150 replicate floor**, and **65 of 135 arms
change a band edge** at the floor. Sweeping it (§4.25) **withdrew three narrowing claims made the same
evening**. What survived 3/3: annealing widens, LoRA unmoved.

**(d) "Nothing outside your repository has ever scored this model."**
True. No public benchmark, no third-party harness, **no downstream task of any kind**. Every claim is
next-token CE from `src/eval.py` on one validation shard. `src/eval.py` is a single point of failure
with no independent implementation.

**(e) "You are 70× under-trained relative to what you're compared against."**
At 9.06M params and 90M tokens, **D/N ≈ 9.9** against a Chinchilla-optimal ~20. Published entries are
data-unconstrained at ~7B tokens. **Perplexity 38.86 is not competitive and we do not claim it is**;
`SCALE.md` §4 gives the arithmetic rather than leaving it to be discovered.

## 2. Attacks we think we can answer

| attack | our answer |
|---|---|
| *"`state_dict` sums to 10,899,616 — you blew the 10M cap"* | The tied embedding appears under two keys. **9,064,608 + 4096×448 = 10,899,616**, verified exactly. The real count is 9,064,608 |
| *"A fresh `Config()` gives 9,065,056, not 9,064,608"* | `Config()`'s default is `state_renorm=True` — **the arm this report rejects at +0.744 nats.** `Config(state_renorm=False)` is the shipped model |
| *"Your depth gate's band [12,24] is your best depth result and you hid it"* | It is **excluded by a decision recorded at 17:40, before the arm ran**: the gate mixes over loops `1..r`, so its plateau measures *mixture-window size*, not depth |
| *"The annealing band result is one grid interval on a coarse ruler"* | Re-run on **every integer depth 12–32**: controls hold within 0.01 over **9 and 8** depths, annealed arms over **19 and 16** — **2.1× and 2.0×, at two seeds.** The sparse grid was *understating* it (§4.25c) |
| *"Your `dg_norm` null just means the gate didn't work, like the last one"* | GATE 1 was registered first and **passed by a wide margin**: effective loops mixed **7.58/8, 14.96/16, 29.84/32**, zero tokens above 0.99. It is a genuinely working mixture and it gains nothing |
| *"Duo-causal W=3 lowers CE — that contradicts your negative"* | Its **registered mechanism check failed**: `cos(Δu_t, Δu_{t−1})` is indistinguishable from control, and 78–101% of the gain is at `r = 1` where the mechanism is *provably* inert |

## 3. Where the evidence is thinner than the prose

- **§4.7e's mechanism is claimed width-independent and measured at one width** (448). The argument is
  structural — one `W_K` cannot decorrelate a collinear stream — but *the measurement is n = 1 in
  width.* §4.28's dose–response uses **random** projections, so its ranks are plausibly an **upper
  bound** on what trained projections would give.
- **The causal test of §4.7e is still running** (`tlab-untie-s0`). Until it lands, the rank
  explanation rests on `dg_norm`'s null, **which is a correlation between low rank and no gain.**
- **XSA is not evidence about looping.** 84–91% at `r = 1`, a generic attention operator, band claim
  dead at the second seed, untested at scale. We report it; it should not be read as a loop result.
- **Several results are n = 1 or n = 2** and say so in place — the token-keyed annealing lead
  (−0.2208, n = 1) most prominently.
- **Cross-job drift is 0.0914 and unexplained.** *Not* a tokenizer artifact — every DataSphere
  kernel's `train_tokenizer` is byte-identical (md5 `1dab774d…`) — so it is real variation at 6.1× the
  in-job floor, cause unestablished. **An explained spread would be benign; this one bounds every
  cross-job statement.**
- **The noise floors (0.0150 / 0.0541) are measured at 2.5M and applied to 90M claims throughout.** No
  same-config replicate at full budget has ever been run.

## 4. Process criticisms that are fair

- **One agent wrote the analysis, the instruments, and the critique of both.** Three mechanical gates
  now exist because self-review kept failing — **but they check consistency, not correctness. A number
  that is wrong everywhere passes all three.**
- **`report.md` §1 is agent-written**, from the dated record, at the author's instruction. It says so
  in a banner at its own head and in `submission/README.md`. **It is a reconstruction from artifacts,
  not the author's account of their own reasoning.** The task grades ideation separately and we would
  rather state this plainly than have it inferred.
- **Twelve claims were retracted**, three on the final day by their own pre-registered falsifiers. We
  read that as process evidence. **A reviewer may read it as fragile measurement instead, and nothing
  in this repository distinguishes the two.** Both are true: the checking was aggressive *and* many
  effects here sit near the floor.
- **`report.md` has not had a full end-to-end read since ~18:45**, and ~2,000 lines have landed since.
  Targeted reads of the new sections were done and found three defects, all ours, all fixed.
- **The wandb API key has not been rotated** and the repositories go public. Scrubbed and
  secret-scanned; that does not un-send it. Outstanding, and the author's action.
- **Pre-registration covers roughly the final day's arms, not the project.**

## 5. Things in the repository that look wrong and are not

| what you'll find | why it is there |
|---|---|
| `checkpoints/BROKEN_a1_dense_grid_10M__VOCAB_MISMATCH.json` with CE **9.27** | A local eval of DataSphere weights against the shipped vocabulary. **Quarantined with a README, kept rather than deleted** per this project's retraction rule (§6.0 row 35) |
| A DataSphere job with status **ERROR** that we harvested anyway | Cosmetic. `log.txt` has no `Error while processing file`; stdout ends `ALL DONE`; all three checkpoints returned |
| ~20 root-level documents contradicting `report.md` | **Dated working records**, each bannered, kept because the audit trail is what makes the retractions meaningful. The repository README's *How to read this repository* table says which surfaces are current |
| `submission/` has **eight** documents where earlier text says six or seven | Documents were added tonight (`EARLY_EXIT.md`, `LIMITATIONS.md`). The count is now derived from the file list, not typed |

## 6. Numbers that will not reproduce, and why — read this before recomputing

**This is the section we would most want a reviewer to have**, because every entry is a trap we walked
into ourselves today.

| if you recompute… | you will get… | why |
|---|---|---|
| **any DataSphere checkpoint, locally** | CE ≈ **9.3**, above chance (8.3178) | DS kernels train their **own** BPE (NFKC + `unk_token` + 5,000 docs) and **return no `tokenizer.json`**. `src/train_tokenizer.py` uses no normalizer. **Those checkpoints cannot be evaluated locally at all.** Kaggle's tokenizer **is** byte-identical to the shipped one, so Kaggle checkpoints can |
| **ρ from `checkpoints/jacobian_spec_results.json`** | **1.2273 / 1.0801**, not the report's **1.6227** | The report's figure is the **90M control**; that JSON holds the **2.5M donors**. All > 1; only the magnitude is checkpoint-specific |
| **log-drift vs power-law R² from `angular_convergence.json`** | **0.9885 / 0.8341**, not **0.986 / 0.748** | Same split: the report quotes the 90M control, the JSON holds 2.5M donors. Log-drift wins on every checkpoint |
| **"oracle depth's cv = 0.798"** | cv of oracle **depth** is **0.95–1.18** | 0.798 is the **angular budget at each token's oracle depth**, not the cv of the depth. **The report mislabelled this until 21:20**; corrected, and the true figure is *larger*, so the argument strengthens |
| **oracle headroom** | **0.3084**, **0.3083**, or **0.2008/0.2032** | Three legitimate quantities: 46M test-split, 46M full-set, and the 2.5M annealed pair. Each is labelled where used |
| **the tail fraction on a 32-loop dump** | **~30.9%**, not 27.9% | 27.9% is *oracle depth > 32* on a **64**-loop sweep. On a 32-loop dump "past 32" is 0% by construction and 30.9% is *fraction at the cap* — a different quantity |
| **any band, at a different `plateau` tolerance** | different edges for **65 of 135 arms** | `tol = 0.01` is tighter than the 0.0150 floor. §4.25 sweeps it; four of eleven paired verdicts are tolerance-dependent and are marked |
| **absolute CE across two jobs** | up to **0.0914** apart at identical config | Real, unexplained cross-job drift (§4.27). **In-job Δ always; absolute CE only within a tokenizer family** |

## 7. What we would ask for, if the reviewer has budget

1. **One same-config replicate at 90M** — settles the floor every "×the floor" statement uses.
2. **Any downstream task, however small** — the capability claim is currently "CE went down".
3. **A second width** — tests §4.7e's width-independence, asserted from a mechanism and measured once.
4. **An independent re-implementation of `src/eval.py`** — the single point of failure behind every
   number here.

*Full absence list: `submission/LIMITATIONS.md`. Full error log: `submission/FAILURES.md` and
`report.md` §6.0 (35 rows).*
