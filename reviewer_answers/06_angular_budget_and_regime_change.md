# Reply — 2026-08-23 12:15 · the angular budget ran, and it discriminates

Your two headline items were both right, and one of them changed what the report claims. Taking them
in the order you ranked them.

## 1. The angular budget: **B rises ~1.4×. Terminal-only is a budget intervention, not a rate one.**

You called this "the last thing with a chance of changing what the report *is*." It ran — zero
training, the §4.3 state hooks, four batches — on the §4.14 dense/terminal pair at **both** seeds:

`B = Σ_{t≤k*} ‖u_t − u_{t−1}‖`, `u = h/‖h‖`, `k*` = each model's own plateau midpoint.

| seed | B dense (k*=11.3) | B terminal (k*=17.0) | **ratio** | step₁ | step₃₁ |
|---|---|---|---|---|---|
| 0 | 0.3749 | 0.5188 | **1.384** | 0.1053 → 0.1143 | 0.00420 → 0.00551 |
| 1 | 0.3700 | 0.5245 | **1.417** | 0.1072 → 0.1145 | 0.00385 → 0.00569 |

**The budget rises and the step sizes barely move** — if anything terminal-only's steps are marginally
*larger* at both ends. So it is not spending a fixed allowance more slowly. Your two-way
discrimination resolves to the first branch: **terminal-only buys more useful angular computation.**

That gives §3.5 a mechanism instead of a correlation, and the spine you proposed is now the one in the
report: **three interventions change the rate; one changes the budget.** Written up as **§4.16c**.

**One part I could not run, and the reason is a defect worth naming.** Extending `B` across §4.9's five
train-at-L arms was impossible: those trained on DataSphere, and **every DS job config listed only
`results.json` under `outputs:`** — so a file the kernel wrote (`{run_name}_last.pt`) was never
returned. **~20 jobs' weights are unrecoverable.** The local checkpoints all come from Kaggle or MPS,
which is why the dense/terminal pair *was* available. Logged as §6.0 row 23 and fixed in 23 configs
for future jobs. The currently-running artifact will also return curves only.

## 2. The reversal — you were right, and it is already visible on a *second* axis

I had written §4.6b as "the penalty shrinks 12× and flips character". Your decomposition is sharper
and it is now the framing: **ΔCE@1 goes −0.2196 → +0.2263, a sign flip of 0.45 nats**, while ΔCE_best
merely shrinks. A regime change, not a magnitude change, with §4.12 supplying the mechanism.

**Your pre-registration is committed, verbatim, and it went in while the 10M control was still at step
1220/4882** — before it could land (`RUNS.md`, "PRE-REGISTRATION … written 11:55 — BEFORE the control
landed"). Outcome **B** is named there explicitly, and you were right that my previous falsifier
("does CE shrink toward zero") did not cover it.

**And then the sweep found B already happening, one axis over.** Running the decomposition across all
**49 in-job arm-vs-control pairs**:

| annealed arm | ΔCE_best | ΔCE@1 | class |
|---|---|---|---|
| μ_rec = 18, sw90, seed 0 | −0.0811 | **−0.0416** | both-improve |
| μ_rec = 18, sw90, seed 1 | −0.0609 | **−0.0277** | both-improve |
| μ_rec = 40, sw90 | −0.0264 | **+0.0749** | **damage-driven** |
| μ_rec = 40, sw75 | −0.0192 | **+0.1749** | **damage-driven** |

**Annealing is both-improve at μ_rec = 18 and damage-driven at μ_rec = 40.** Same reversal as the
penalty, on the *schedule* axis instead of the *token* axis, and consistent with the same §4.12
mechanism: an intervention helps everywhere while depth utility is scarce (μ18 control gain 0.0992)
and becomes depth-specialisation once there is some to specialise (μ40 control gain 0.1855).

So the "better on both axes" property is **specific to μ_rec = 18**, not a property of annealing. That
is now stated in §4.17, and it materially raises the prior on your outcome B for the 10M test.

## 3. §4.13 — promoted, but I will not cite your three papers

Promoted in-text to what it is: **a negative on one of the three levers the task names by name**, in
the strongest available form — from scratch, **monotone in noise magnitude** across three levels
rather than a single ablation point. §2.0 now carries a table mapping all three task-named levers
(looping schemes → the method; normalisations → a family of nulls; exploration → this negative).

**I did not cite 2602.14759, 2603.19714, 2604.18839 or 2509.23314, and I am telling you why.** None is
obtainable here — they are not in `papers/sources/`. This morning I asserted a "correction" to you
from a summarising web fetch and was wrong (`05_RETRACTION_73_percent.md`), so the standing rule is
now that a citation is VERIFIED only against the paper's own LaTeX. All four are logged **SECOND-HAND**
in `VERIFICATION.md` with your relayed content, hedged in §4.13, and **no claim rests on them**.

That said, the SPRM account you relayed is a better explanation than any I derived — data-poor helps,
data-rich hurts, and FineWeb next-token is the maximally data-rich end — so it is in the text as an
attributed relay. **If you can get me those tarballs the way the last two arrived, I will verify and
promote them.** The Two-Scale spiral-vs-parallel contrast is logged the same way, flagged as a
*possible* contradiction with §4.3's cos → 0.9999 rather than a resolved one.

## 4. bpb calibration — done, with the regime

**§6.0a**, new. Control **1.5633**, penalty **1.5503**, against 1.7330 at 46M — and Parameter Golf's
1.058 stated beside them rather than left to be discovered. The regime that explains most of the gap:
**D/N ≈ 9.9** (12.5 on non-embedding params) against Chinchilla-optimal ~20, because the task caps
tokens at 100M. Measured, not asserted: 46M → 90M bought 0.39–0.42 nats = 0.17 bpb of the 0.49 bpb
gap, and closing to D/N ≈ 20 would be worth roughly another 0.12 bpb with no architectural change.

## 5. Where I differ from your schedule

You said add nothing else to the queue. Agreed, with one exception already taken: a ~2-minute node
probe, because the user asked directly whether I could inspect the machine. It failed on a missing
`local-paths:` — and **that failure is what exposed the wandb API key sitting in plaintext in 18
tracked config files and 18 commits**. Scrubbed; branch `review` is a single squashed commit verified
clean; the key still needs rotating. So the marginal-arm rule was right and the exception paid for
itself for an unrelated reason.
