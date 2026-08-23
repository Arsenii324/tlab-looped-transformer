# Reply — 2026-08-23 21:50 · the subagent concerns, audited honestly, and the causal test now running

Two subagent passes surfaced a long list. **The short answer to "have they all been addressed" is
no.** §3 is the list of what has *not* been, which is the part worth reading.

---

## 1. What landed since reply 26

**Cross-job drift is 0.0914, not 0.0074–0.0334 (§4.27).** Three of tonight's jobs each carry a control
with a **verified-identical** config — same seed, tokens, `supervise_k`, lr, 1,219 steps: **5.3765 /
5.3052 / 5.2851.** That is **6.1× the in-job dense floor** and 2.7× the top of the band the report
quotes in five places. The old figure came from accidental replicate pairs found while auditing
something else; this is the first deliberate three-way measurement. **It does not touch §4.23** —
every arm there is differenced in-job, which this vindicates — but it does touch any pooled cross-job
statistic, including the LoRA `n = 5` mean.

**A trap disarmed, no result affected.** `dv_lora_r4_s0`'s stored config carries
`cond_fixed_branch=False`. In the **frozen kernel that ran it**, that field is a `bool` and the logic
is `0 if flag else t`, so `False` means **cycled** — §4.23c is correct. But `src/model.py` declares the
same field `int | None` and branches on `is not None`, where `False` selects **branch 0** and silently
turns a cycled arm into a pinned one. **Two implementations of one field with incompatible
semantics.** `src/model.py` now raises rather than guessing; guard verified to fire, 13/13 still pass.

**§4.25b — the grid objection, and it answers differently for the two claims.** A reviewer noted the
paired sweep grid resolves any edge past 16 to one interval. Measured on the one checkpoint evaluable
both ways, **a sparse grid systematically understates the band**: dense `[6,17]` w=11 against sparse
`[8,16]` w=8 at the same tolerance — onset later, end earlier, ~27% of width lost. All paired arms
share the grid, so comparisons are fair, but **the paired tables' absolute bands are not the same
object as the headline's `[6,17]`**, which the report had left implicit. *And annealing's `16 → 24` is
**not** a one-step move:* at depth **20 and 24** the annealed arm is inside tolerance (0.0043, 0.0095)
while the control is outside (0.0142, 0.0237), margins separating 3–4× at each.

**§4.28 — the mechanism priced as a dose–response, and my own pricing was wrong.** Depth-key rank is
`≈ 1.6 × (number of distinct projections)`: 1.603 tied → 3.097 → 5.742 → 10.646 → 30.863. **The
states are rank 1.690 before any projection**, which confirms the 20:01 projection correction as a
curve. I first priced a bucket at +100,352 parameters; **the block has three layers each with its own
`W_K`**, so it is **3 × that**. nb=4 is **+10.0% (9,967,776)**, not +3.3%, and **nb=8 does not fit
under the cap at all.** *Caught by building the arm, not by re-reading the table.*

## 2. The causal test is running — `tlab-untie-s0`, registered before submission

§4.23's `dg_norm` result is a **correlation**: low rank, no gain. This raises the rank and keeps the
gate. Three arms, in one job so §4.27's drift cannot touch it: tied control · `W_K` in 4 loop-index
buckets · 4 buckets **+ the scale-invariant gate**.

**Pre-launch gates, all verified before submitting:** `kv_untie_buckets=1` is **bit-identical** to the
unpatched frozen kernel (max\|diff\| **0.000e+00**); buckets 0 and 1 produce different keys; parameter
counts measured by instantiation; `outputs:` names every file explicitly.

**Falsifiers, in `RUNS.md` before any data exists.** **GATE A:** the trained arm's key rank must
exceed ~4, else the buckets did not take and **nothing is decided**. **GATE B:** if the gate's
contribution at high rank is materially negative, **§4.7e is confirmed causally**; if it is null like
its contribution at rank 1.6, **§4.7e is incomplete and the explanation the report leans on is at best
partial.** **GATE C:** on this report's own regularity I predict **78–101% of any gain at `r = 1`** —
capacity, not depth-mixing. *If instead it concentrates past `r = 8` and the band widens, that is the
first genuine depth mechanism here and the central negative is wrong.*

---

## 3. What has NOT been addressed — the honest list

**(a) The noise floors are measured at 2.5M and applied at 90M.** Every "×the floor" statement in the
report — and there are many — inherits 0.0150 / 0.0541, both measured at screening scale. **A
same-config replicate at full budget has never been run**, and §4.27 has now shown that the *related*
drift figure was understated by 2.7× when someone finally measured it deliberately. **I would expect
the same direction here, and it would weaken several "N× the floor" claims.** Not run: a 90M
replicate is ~9 h of T4.

**(b) A dense integer sweep on the annealing pair.** §4.25b did the check on the 90M control only.
The DataSphere checkpoints **cannot be evaluated locally at all** — those jobs train their own BPE and
do not return `tokenizer.json`, so a local eval reports CE ≈ 9.3 against chance 8.3178. *A fork's
attempt produced exactly that and was quarantined (`checkpoints/BROKEN_a1_dense_grid_10M__README.md`)
after the number was read against chance rather than taken at face value.* **The fix is one line in a
job's `outputs:`, and it was not applied to any job that has already run.**

**(c) `report.md` has not been read end-to-end since ~18:45**, and roughly 1,200 lines have landed
since — §4.23–§4.28. A *targeted* read of the new sections was done and found three defects, all mine,
all fixed. **A full read was judged unaffordable and that is a judgement, not a certainty.**

**(d) The queued ablations A2, A4–A7 are costed and unrun.** Only A3 (this job) was launched.

**(e) The budget-invariance of the central pattern is still open.** Every arm showing "78–101% at
`r = 1`" sits at 2.5–3.5M tokens (§4.24). The Kaggle 12M arm is the only probe and it is still
running; seed noise on the share is ±7–12 points at fixed budget, which bounds what it can settle.

**(f) One thing I got wrong about the audits themselves, recorded because it is the same error they
made.** I wrote that a cited artifact *"does not exist in the repo"*. It existed — in this session's
scratchpad, which I had not searched. **I asserted an absence from a search whose scope I never
stated.**

## 4. What I would do with the remaining time, in order

1. **Harvest `tlab-untie-s0`** (~22:50) and resolve GATES A–C.
2. **Harvest Kaggle `tlab-lora-scaleup`** — the LoRA scale verdict.
3. **(b) above is one line** — if any further job is launched, add `tokenizer.json` to its `outputs:`
   so this class of failure ends.

*What I would not do: start anything whose result could not be written up honestly before 23:59.*
