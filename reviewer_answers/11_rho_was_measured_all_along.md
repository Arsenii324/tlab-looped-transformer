# Reply — 2026-08-23 13:40 · both of your hour-long items, run; one of them inverted

You asked for two things: `‖e‖/‖h_t‖` on the penalised checkpoint, and ρ. Both are done. The first
produced a mechanism and then killed it. The second turned out not to need running — **ρ has been
measured all along, under the wrong name**, and the report has been hedging against a claim it was
already entitled to.

---

## 1. ρ was never missing. The instrument was mislabelled.

`src/jacobian_spec.py` defined `sigma_max()`, documented as *"power iteration on `J^T J`"*. It never
applies `J^T`. The loop is `v ← Jv/‖Jv‖` — plain power iteration on `J`, which converges to
|λ₁|, the **spectral radius**. Computing σ_max would need a VJP; this file only ever does JVPs.

Null-tested on an operator where the two differ by 10× (now wired in as `--null`, because an
instrument that decides a hypothesis needs one):

```
A = [[1, 10], [0, 1]]      rho = 1.0000      sigma_max = 10.0990
the iteration returns 1.0889   ->  rho
```

**This corrects §2 in your favour, and the report was over-hedging itself twice.** §2 said these
numbers *"only bound ρ from above"* and therefore could not establish non-convergence. They **are**
ρ. And the criterion §2 states — *"converges iff σ_max < 1"* — is wrong as written: `σ_max < 1` is
the **sufficient** Banach condition; `ρ < 1` is the actual iff. A non-normal map can have σ_max > 1
and converge fine. So the quantity that was being measured is the appropriate one, not a loose bound
on it.

### Your predicted experiment, run — and ρ is scale-invariant

You said the 100× spread in ‖h‖ makes this a better experiment than it was this morning. It does:

| loop | 46M no-renorm | 90M control | 90M norm-penalty |
|---|---|---|---|
| 2 | 1.7006 | 2.2850 | 1.7766 |
| 8 | **1.0467** | **1.0692** | **1.0480** |
| 16 | 1.0162 | 1.0238 | 1.0103 |
| 32 | 1.0053 | 1.0074 | **0.9953** |
| 64 | 1.0015 | 1.0020 | **0.9915** |
| ‖h‖@8 | 6639.7 | 2334.4 | **17.5** |

**At loop 8 the three agree to within 2% while their state norms differ by 380×.** That is the
scale-invariance result you predicted would be usable, and it is the strongest available statement
that §4.6's "scale sets the rate, not the ceiling" is about scale as a *coordinate* rather than as a
mechanism — the local dynamics belong to the learned operator, not to the scale it is evaluated at.

**Second finding, which I did not expect: the norm penalty is the only arm that ever converges.** It
crosses below 1 at loops 32 and 64 while both others sit just above. The estimator's bias is *upward*
(the null overshoots a defective operator by ~9%), so a sub-1 reading is conservative and the
crossing is real. **That is a mechanism for its narrower plateau** — [6,14] against the control's
[6,17]: a converging map stops paying for extra loops sooner. It also puts that arm, alone, inside
the DEQ regime the task's framing objects to.

**And what this costs.** The 1.0015 / 1.0020 readings at loop 64 are *inside* the estimator's bias
and cannot be distinguished from exactly 1. §2's claim is now narrowed: **non-convergence is
established where the loops are doing work (ρ = 1.70–2.29 at loop 2, far outside any bias) and is
not established at loop 64.** Power iteration also oscillates for a complex-dominant eigenvalue and
12 iterations cannot detect that; unresolved, and stated in the code.

---

## 2. `‖e‖/‖h_t‖`: the regime break is real, the mechanism it suggests is refuted

Prediction and falsifier written into `src/injection_ratio.py`'s docstring before running.

| checkpoint | ‖e‖ | ‖h₁‖ | **e/h @1** | @8 | @64 |
|---|---|---|---|---|---|
| 46M no-state-renorm | 2.212 | 1659.5 | 1.33e-03 | 3.44e-04 | 7.86e-05 |
| 90M control | 1.504 | 466.6 | 3.22e-03 | 6.66e-04 | 1.31e-04 |
| **90M norm-penalty** | 1.573 | **4.379** | **3.59e-01** | 9.49e-02 | 2.09e-02 |

The 46M row reproduces §4.3's own 1.3e-3 → 7e-5, which is the instrument's null. **In the penalised
arm the re-injected input is 36% of the state norm at loop 1** — first-order, not a rounding error.
And the controlled comparison is clean: **‖e‖ barely moves** (1.504 → 1.573), while ‖h₁‖ collapses
107×. The penalty acts on the state and does not reach the embedding through the tied head, so the
pre-registered falsifier did not fire.

**Then I tested the mechanism and it failed.** The tempting reading — loop-1 readout decoding
substantially un-processed input — makes a sharp prediction, *because the head is tied to the
embedding*: loop-1 predictions should collapse toward copying the current token.

| checkpoint | loop | copy-rate | next-tok acc | **cos(h₁, e)** | CE |
|---|---|---|---|---|---|
| 90M control | 1 | 0.0024 | 0.3198 | −0.0246 | 3.8134 |
| 90M norm-penalty | 1 | 0.0005 | 0.2583 | **−0.0712** | 4.1527 |
| 46M no-renorm | 1 | 0.0005 | 0.2676 | −0.0308 | 4.0887 |

**Refuted.** Copy-rate ~0.002 everywhere, and `cos(h₁, e)` is slightly *negative* in all three. h₁ is
nearly orthogonal to `e`: large in magnitude, different in direction. So `ΔCE@1 = +0.2263` — the 88%
of that arm's loop-gain advantage that §3.5 turns on — **stays unexplained, with the most natural
explanation now eliminated.** I think that is worth more than the mechanism would have been, since
§4.6b's decomposition is what §3.5 rests on and it now rests on one fewer untested story.

*A note on the test I nearly ran instead.* My first instinct was an `inject_none` rollout at loop 1.
It is void: `model.py:413` sets `h = h0 + e` unconditionally and `model.py:429` injects only at
`t > 0`, so loop 1 is `block(h0+e)` in every arm and `inject_mode` cannot touch it. Same class as
§4.1's original "no injection" mislabelling, caught this time before it produced a number.

---

## 3. A live bug found on the way, and it is your class of concern

Verifying my own claim that `radial_clamp.py` derives levels per-checkpoint, I found it does — down
one branch. Down the other it set `levels = {}`, printed *"falling back to measured-on-the-fly
norms"*, and **there was no such fallback**: the clamp loop iterates over `levels`, so the script
produced only the unclamped control, wrote a results file, and exited 0. **Neither 90M checkpoint has
the dynamics json it needs**, so running §4.6's experiment on the *shipped* model was a silent no-op
that reads as a completed run.

Fixed for real (one forward pass; reproduces the json path to 0.3% — 78.41/313.70/504.30 against the
stored 78.18/313.22/502.36), plus a `RuntimeError` refusing to write a results file that contains
only the control. No published number changes — every §4.6 number came from the 46M checkpoint, which
does have the json. Logged as §6.0 row 24; the ρ mislabelling is row 25.

Your point about the clamp levels being a correctness issue rather than a scoping one is now in §4.6
as a warning block, with the three checkpoints' actual levels side by side.

---

## Run state

| stream | state |
|---|---|
| DataSphere `tlab-deep-full` | EXECUTING since 07:29 (~6.2 h). Curves only. Cancelling is the harvest |
| DataSphere, everything else | terminal and collected; `tlab-hyper-screen` written into §6.0b |
| Kaggle | idle since yesterday evening; both 90M checkpoints local |
| Local `local_anneal_sw75_s0` | **healthy** — 6 evals, step 912/1220, 1.87M/2.5M tokens, about to cross the k=5→1 switch at step 915. Before this morning's fix: 0 evals, step −1, three chunks restarting from zero |

**Still open and yours to judge:** the anchor account (QUEUE W6) is still the largest unwritten item,
and §1 is still empty.
