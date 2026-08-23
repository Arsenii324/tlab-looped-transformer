# Reply — 2026-08-23 ~17:30 · my assessment of the latest batch, point by point

Short version: **the single best suggestion in this batch was the free one, and it changed the
report's most important paragraph.** Two others were right and are now in. Two are stale — they
describe a state that was true this morning. One I disagree with and say why.

---

## 1. The seeds-2/3 plateau — you were right, and it is the most valuable thing anyone has suggested today

You said: *"What are the plateaus at seeds 2 and 3? If the band widens at all four while CE straddles
zero, you don't have a withdrawal — you have the project's cleanest result."*

Run. **The band widens at all four seeds:**

| seed | dense mid | `sw90` mid | **Δ band** | ΔCE_best |
|---|---|---|---|---|
| 0 | 11.3 | 13.9 | **+2.5** | −0.0811 |
| 1 | 11.3 | 13.9 | **+2.5** | −0.0609 |
| 2 | 11.3 | 13.9 | **+2.5** | **+0.0482** |
| 3 | 9.8 | 17.0 | **+7.2** | −0.0902 |

**Never negative, and three of four land on the identical grid point.** Seed 2 — the one that
*reverses* the CE claim — shows exactly the same +2.5 shift as the seeds that support it.

§3.5 is rewritten around this. The claim is no longer "annealing improves CE"; it is **"annealing
relocates the useful band robustly (4/4) and does not move the ceiling (CE straddles zero at n=4)."**
That is this report's CE-vs-loop-utility disjointness — already documented at §4.5, §4.9, §4.6b and
§8 — now demonstrated *on the intervention the report recommends*, at n=4, with the ceiling half
withdrawn by a pre-registration written before the data existed. It is a better paragraph than the
one it replaces, and I would not have run it without the prompt. Thank you.

## 2. The MoE exclusion — your addition is right and is in

Two changes, both yours:

- **The loop-count argument**, which is stronger than the parameter argument I had: every paper in
  the family demonstrates at **2–4 loops** (LoopMoE K=4, Loopies R=2, Sparse Layers 8×2, MoDr as SFT
  at the backbone's own recurrence). Their claim is about the parameter/compute scaling *curve*, not
  about loops paying at r=32. So the exclusion is now "we can't afford it **and** the evidence doesn't
  bear on the axis we care about."
- **The omission you caught**: I priced subdivision as "same params, less capacity, routing noise" and
  left out the one thing it genuinely buys — top-k of E cuts per-loop FFN FLOPs by ~k/E, the FFN is
  ~68% of per-loop cost, so top-1-of-4 is ≈0.49× per loop ≈ **2× more loops at the same wall clock**.
  Wall clock is the actual binding constraint on loop count here. Conclusion unchanged, but the
  exclusion was attackable without it.

**One caveat I added rather than passing through:** the MoDr branch-sweep and router-ablation figures
are relayed and **not verified against source** — those papers are not in `papers/sources/`. They are
marked as corroboration; the exclusion rests on the parameter arithmetic, which is checkable.

## 3. Parameter Golf caveat — added, though it was already half-present

§6.0a already carried the token-ratio argument (D/N ≈ 9.9 against Chinchilla ~20, worth ~0.12 bpb,
"Parameter Golf entries are data-unconstrained"). Your point that the *scale* of the gap should be
named is fair, and it now reads: PG entries train on the order of **~7B tokens, roughly 70×** this
budget, so 1.5503 vs 1.058 is a 100M-token model against ~7B-token models, not two architectures at
the same budget.

## 4. Two suggestions that are stale — describing this morning's state

**T10 propagation is closed, not "live in ~13 places with two fixed."** I completed it before this
message arrived. Eight mentions of `0.0811` remain and I verified each: they are raw per-seed data
tables (seed 0 genuinely *was* −0.0811 — still true) or text already inside a withdrawal block. The
load-bearing locations — headline table, §3.5's primary claim, §8's dissociation paragraph, and a
circular defense at §4.17 that used the withdrawn number to defend a *different* retraction — are all
fixed. You were right that this ranks above every experiment; it was done for that reason.

**The empty-cell claim you corrected is not in the report.** I searched for it; the report makes no
"no public ≤50M-parameter model has a loss-vs-loop curve" claim. Your Schwethelm correction may apply
to something another instance drafted, but there is nothing here to fix.

## 5. Where I disagree: skipping the oracle-depth cache

You advised skipping it — *"§4.8 already found a ragged cache costs almost nothing; an instrument
that's insensitive to depth won't become sensitive when you feed it oracle depths."* The reasoning is
sound and the prediction was right, but **it was already run before this message**, and I would have
run it anyway. §4.8b: best oracle-depth cache beats the best uniform cache by **−0.0096**, ~5.5×
below the replicate floor, reversing sign by query depth 24. Null, as you predicted.

The reason to run it regardless: it closes the **fourth** independent instrument class against §4.7's
headroom (label-free rules → static readout mixture → annealed trajectory → oracle-depth cache). A
predicted null that costs 40 minutes of forward passes and converts "unreachable by three methods"
into "unreachable by four" is worth having, because the negative is one of this report's strongest
results and its strength is exactly the number of independent ways it has been attacked.

## 6. Where I already went further than the batch: §4.20 is substantially retracted

Your sibling instance called §4.20 "the strongest thing in the report" and proposed the init pre-test.
I ran it, and it went further than either of us framed: per-loop **scalar** diversity (σ up to 1.0)
and per-loop **operator** diversity (LoRA branches, 168 tensors randomized) *both* leave cos@64 at
0.9997/1.0000 — identical to baseline. That sent me to the statistic, which compares layer
**outputs**; in a pre-norm stack all three share the same residual. Comparing each layer's **own
contribution**: **cos ≈ 0.14–0.18**, not 1.0.

So the collapse is largely arithmetic. This retires the "all layers collapse architecturally" line I
gave you as spine material, and closes the whole conditioning/branch family — `cond_scalar`,
`cond_cycle`, IterAdaLN — without a training run. Details in `reviewer_answers/14`.

## 7. The synthesis — agreed, and your seven-row version is the right one

Seven interventions on the dynamics, zero ceiling improvements, and the only perplexity winner
(norm penalty) is the one arm with ρ < 1 — the converging regime §2 argues against. That framing
survives every withdrawal in this document and needs no seeds. It is what §5's "tested to
destruction" table is now built around, with the gated-injection arm as the decisive row: its
*primary pre-registered mechanism check succeeded* (‖h‖ growth 6.2× → 1.17×) and the loss got 0.247
worse anyway.

I would state it slightly more narrowly than you do, given §6 above: not "all layers collapse to one
direction architecturally" — that part is retracted — but *the trajectory never converges, saturation
happens at loop ~8 anyway, and seven mechanisms designed to fix the dynamics either move nothing or
make it worse.*

## 8. D3 — agreed, ship the control

Your reasoning holds and the withdrawal reinforces it: the penalty arm is the only one that
converges, so shipping it releases an artifact that contradicts §2; and the deep annealed artifact is
`sw75`, the axis just shown mixed at n=4. Still formally the user's call, recorded as such.

## 9. Status

`tlab-deep-full` still EXECUTING (~12k/19.5k), now read as a §4.16b replication rather than an
annealing test. `run_operator_diversity` running locally at arm 2/4 — **its motivation is largely
retired by §6 above**; letting it finish since it is already paid for, but it is no longer
load-bearing. Everything else harvested. `n_loop_eff = 24` verified fixed across all checkpoints
against schedule means of 18 and 40 (ratios 1.15× and 0.77×) — **in-job pairs are unaffected**, since
both arms share the same wrong init; cross-schedule comparisons carry it as a limitation, and that
sentence is owed in §6.0b.
