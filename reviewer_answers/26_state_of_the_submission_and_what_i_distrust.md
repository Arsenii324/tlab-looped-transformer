# Reply — 2026-08-23 21:20 · state of the submission, two audits verified, and what I still distrust

This is a state report rather than a result report. §4 is the part I would read first — it is what I
think is *wrong or unaudited* in what we are about to submit, written before anyone asks.

---

## 1. Where the submission actually is

**Complete and internally consistent, for the first time today.** `submission/` is seven documents;
`report.md` is the evidence base at ~6,800 lines. Four gates run green and two of them are new:

| gate | what it enforces | state |
|---|---|---|
| `src/test_model.py` | 13 correctness checks incl. the block vs the real Qwen3 reference (2.4e-07) | PASS |
| `src/test_plateau.py` | 8 checks on the depth statistic incl. a deliberate falsification probe | PASS |
| `src/headline.py check` | every headline number still matches the artifact it came from | consistent |
| `src/check_caveats.py --strict` | **no file states a deflated claim without its caveat token** | 0 missing |
| `src/check_crossref.py --strict` | **no figure in `submission/` is absent from `report.md`** | 0 orphans |
| `src/make_inventory.py` | the inventory and its **coverage block** are generated from artifacts | 135 arms, 8 absences, each annotated |

The last two did not exist at 20:00. Both were built because a defect got through that the existing
gates structurally could not catch.

## 2. What landed tonight, and one of it cuts against us

**Four DataSphere jobs, all harvested and written up** (`report.md` §4.23a–e).

**`dg_norm` — §4.7e survived the experiment built to kill it.** A *scale-invariant* depth gate that
demonstrably mixes (**7.58/8, 14.96/16, 29.84/32** effective loops, zero tokens above 0.99 top-weight)
returns **−0.0012 / +0.0023** at two seeds. The joint falsifier was registered at 19:22 with both
branches named, before the arm existed, and has not been touched since. *A working per-token soft
mixture over depths — the first this project has built — buys nothing over a representation spanning
~1.6 of 32 dimensions.*

**XSA replicates and is now the project's largest positive** — −0.2162 / −0.2633, mean −0.2398, ~16×
the floor, **zero parameters**. But the "band unmoved" half of the 19:15 prediction **failed at seed
1** ([8,20] → [8,16]) and is withdrawn. I had shipped that sentence into `submission/README.md`
twenty minutes before the replicate contradicted it.

**`divx-s1` cuts against our own positive, and this is the one I would lead with.** At seed 1 the
branch **pinned to a single index** — identical parameters, *zero* diversity — beats the cycled arm
**5.6×**: −0.1470 against −0.0261. With seed 0's 82%, the two-seed average makes **pinning better than
cycling** (−0.1251 vs −0.0756). **Operator diversity, the mechanism loop-cycled LoRA is named for, has
no measurable benefit here.** And the cycled arm ranges −0.0261…−0.1251 across two in-job pairs at one
budget — a **4.8× seed spread** that weakens the LoRA positive further.

**`recmethod-s2` at 10M** reproduces annealing's band effect *to the grid point* (onset 8→8, end
16→24, midpoint 11.3→13.9 — identical to 2.5M; **5/5 seeds across a 4× budget range**) while giving
the CE claim its worst point yet, **+0.1119**. Shipping the **dense** control is now evidenced at the
recipe's own budget rather than inherited from launch order.

**Still running:** Kaggle `tlab-lora-scaleup`, 12M/arm, ETA ~21:52. It is the *entire* budget probe of
§4.24 below.

## 3. Two automated adversarial audits, verified item by item rather than adopted

**Ten flagged items. Two real.** I checked each against the artifact, which is the only defensible
response to an automated pass over one's own work.

**Real defect 1, and it is this project's characteristic error in a sixth costume.** The report says
*"oracle depth's cv is 0.798."* **It is not the cv of oracle depth.** `src/cumulative_exit.py`
computes it as `cum[i, k_oracle(i)]` and prints it as *"budget AT each token's oracle depth"* — an
**angular-budget** quantity. The cv of the oracle *depth* is **0.9514** (2.5M, 32-grid) and **1.18**
(90M, 64-grid) — both **larger**, so the diagnostic strengthens once stated correctly. Fixed in §0,
§4.7b, `NEGATIVE_RESULTS.md` §3, and the `depth_mixture.py` docstring that had propagated it.
*A statistic about one space, labelled as a claim about another. That is the third `FAILURES.md`
pattern, found by an outside pass rather than by me.*

**Real defect 2.** A fresh `LoopedTransformer(Config())` prints **9,065,056**, not 9,064,608. The 448
is `loop_norm.weight`, allocated unconditionally and **unused** when `state_renorm=False`. Both are
under the cap and the released model uses 9,064,608 — but someone who clones and instantiates sees
the other number, so it is now stated. *(The audit blamed `depth_gate_head`; that is created
conditionally and contributes zero. Right defect, wrong cause.)*

**The audits' own headline finding was a scope error, and I made a worse one checking it.** Both
flagged "oracle CV 0.9514 vs 0.798, tail 30.9% vs 27.9%" as a subsample failure. Those numbers are
**exactly right** — for the **2.5M model's 32-loop dump**, where "fraction past 32" is 0.00% by
construction. The report's 27.9% is the **90M model's 64-loop dump**, and I recomputed it
independently at **27.87%**. Different model, different grid, different quantity.

> **My own error, which was worse than the audit's.** I said the cited artifact *"does not exist in
> the repo."* It does exist — in this session's **scratchpad**, which I had not searched. I asserted
> absence from a search whose scope I had not stated. That is the same failure as the audit's, and I
> made it while diagnosing theirs.

**Also checked because it was the sharpest remaining attack surface:** `depth_key_rank.py` computes
effective rank over a **stride-subsample of ~64 tokens**. Re-run at **16× the tokens with no
subsample**, layer 0 gives **1.586** against the reported **1.614**, per-layer delta **≤0.011**. **The
rank collapse is not a sampling artifact.**

**Two shipped JSONs will mislead the next reader exactly as they misled the audit**, so §4.3's table
now says so: `ρ = 1.6227` and log-drift/power R² `0.986/0.748` are the **90M control**, while
`jacobian_spec_results.json` and `angular_convergence.json` hold the **2.5M donors** (1.2273/1.0801;
0.9885/0.8341). All ρ > 1, log-drift wins everywhere; only the magnitude is checkpoint-specific.

**The process finding is worth more than either defect.** Automated adversarial passes have now run
three times over this project (two tonight, one `agy` job earlier). **Roughly 20–30% of flagged items
were real, and every real one was a scope or labelling error rather than a wrong measurement.** The
science has been audited far harder than the prose. What is fragile is the **consistency surface** —
which grew from one file to thirty in five hours.

## 4. What I distrust in our own submission — the part I would read first

**(a) The central finding's scope condition is unfavourable and now stated (§4.24).** *Every* paired
loss-lowering arm in the project sits at **2.5–3.5M tokens** — a 1.4× range against the 36× that
separates screening from the headline. So "78–101% of the gain sits at r = 1" **has no budget
leverage behind it**. Meanwhile loop gain itself roughly **triples** with budget (median 0.1084 at
≤3M over 104 arms; **0.3023** for the 90M control). *The pattern is measured exactly where the
denominator is smallest.* Both readings are live and I have not picked one. One arm probes it, and
seed noise on the share is already ±7–12 points at fixed budget, which bounds what it can settle.

**(b) The LoRA positive is now the weakest thing we call a positive.** Post-hoc rank restriction, an
all-arms interval covering zero, no dose–response, ~90% of the gain at r=1 where cycling is inert, a
zero-diversity pin that beats it, and a 4.8× seed spread. **I would not be surprised to see it
withdrawn**, and the Kaggle arm may do it tonight.

**(c) Nobody has read the current `submission/` end to end.** It changed substantially between 20:06
and 21:20 — count convention, XSA at n=2, `divx` seed 1, §4.24, `METHOD.md` §2's qualification. The
two previous end-to-end reads of same-shape documents found **12 defects each, three serious, none
grep-reachable.** This is the highest-value unspent action.

**(d) Unknown unknowns I can name the shape of but not the content.** Every number here comes from
*our own* eval harness; nothing external has ever scored this model. The `plateau` tolerance (0.01)
was chosen once and never swept — every band claim in the report inherits it. The 4096-vocab
tokenizer was trained once and never varied, so "bits/byte" carries an unexamined dependence. And
§4.24's regularity is the only place we have checked whether a screening result survives scale, on
one arm.

## 5. What I would still do with the time

1. **Kaggle `tlab-lora-scaleup` (~21:52)** → the LoRA scale verdict, whichever way it falls.
2. **An end-to-end adversarial read of `submission/` as it now stands** — (c) above.
3. **Sweep the plateau tolerance** (0.005 / 0.01 / 0.02) over the stored curves and report whether any
   band claim flips. Pure post-hoc, no compute, and it closes the largest unexamined instrument
   choice in the report.

**Not doing, with reasons:** a 90M annealed run (dequeued — the 10M in-job result makes the case
against it); a V100 speedup investigation (no time to spend the result); a second full read of
`report.md` (not affordable — a targeted read of tonight's new sections was done instead and found
three defects, all mine, all fixed).
