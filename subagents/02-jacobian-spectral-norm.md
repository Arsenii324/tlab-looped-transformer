> **Dated working record — not a maintained document.** Last committed 2026-08-23; the text itself may be older. Kept intact rather than rewritten, because this project's rule is that superseded statements stay visible with their corrections. **Current numbers are in [`submission/`](submission/) and [`report.md`](report.md); where this file disagrees with them, they win.** See the repository README's *How to read this repository* table.

# §4.3's "no contraction" — tested against the actual Jacobian, not a finite perturbation

**From:** fork worker (third). **Status:** claim CONFIRMED and substantially sharpened. One wording
fix needed; one new quantitative result available.

## The gap

§4.3's central mechanistic claim — *"There is no contraction and no fixed point"* — rests entirely
on a **finite h₀ perturbation** propagated through the loop (`state_dynamics.py`, `noise_scale=1.0`
against ‖h₀‖≈1.7, i.e. a ~60% perturbation, well outside the linear regime). That is a global
sensitivity probe. **Contraction is technically a property of the map's Jacobian**: F is contractive
iff its local Lipschitz constant σ_max(∂F/∂h) < 1. That quantity had never been measured in this
project, so the report's most load-bearing claim rested on a proxy.

It was a live risk, not a formality: a map can be contracting in nearly every direction while having
one neutral/expanding direction (here, plausibly the radial one §4.3 identifies), and a finite
perturbation aligned with that direction would show "no contraction" while the map is contracting
almost everywhere.

## Method

Power iteration with finite-difference JVPs at the *actual trajectory points*, ε = 10⁻³·‖h_t‖ so the
linearisation is genuine: `w = (F(h_t+εv) − F(h_t))/ε`, `σ = ‖w‖/‖v‖`, 12 iterations, on the real
loop map `F(h) = [loop_norm](block(inject(h,e)))`. CPU, no training, ~2 min.

## Result

| loop | `no_state_renorm` σ_max | ‖h‖ | `center` σ_max | ‖h‖ |
|---|---|---|---|---|
| 2 | **1.4714** | 1,724 | **0.4910** | 29.3 |
| 4 | 1.1465 | 3,431 | 0.5392 | 29.6 |
| 8 | 1.0484 | 6,052 | 0.5253 | 29.7 |
| 16 | 1.0170 | 10,171 | 0.5437 | 29.7 |
| 32 | 1.0059 | 17,236 | 0.5600 | 29.7 |
| 64 | **1.0015** | 30,256 | **0.5324** | 29.7 |

**The claim survives**: σ_max ≥ 1 at every loop for the winning config — it never contracts, measured
properly. **But it is not "expanding" either**, and the report should not imply that: σ_max decays
monotonically toward 1 from above (1.47 → 1.0015). The map is **asymptotically neutral**.

**`center` is quantified for the first time**: σ_max ≈ 0.49–0.56 uniformly from loop 2 on. That is a
*hard* contraction and it explains the fixed point §4.3 observes by loop ~16 — 0.53¹⁶ = 3.9×10⁻⁵, so
any perturbation is more than four orders down by then, matching the measured collapse to 0.0000
(0.53⁸ = 6.2×10⁻³ at loop 8, where §4.3 measures rel. perturbation 0.0042 — same order).

**Two independent instruments agree**, which is the useful part: the finite-perturbation probe and
the Jacobian spectral norm give the same verdict on both checkpoints. §4.3's conclusion was right and
is now rigorously grounded rather than inferred.

## What to change in report.md

1. §4.3: state σ_max directly — it is stronger than the perturbation argument and one line.
2. Replace any "expanding map" phrasing with **asymptotically neutral (σ_max → 1⁺)**. The superseded
   reading in §4.3 already uses "expanding"; the corrected text says "does not contract", which is
   right, but the new number makes the precise form available.
3. §4.3/§4.6: `center`'s σ_max ≈ 0.53 turns "contracts hard" from a qualitative statement into a
   measured one, and predicts the loop-16 fixed point quantitatively.

## Not verified
Whether σ_max → 1⁺ has a closed form here. `(σ−1)·t` is not constant (0.94 → 0.096 over t=2..64), so
the approach is faster than 1/t; I did not fit it and would not without more trajectory points.
Also: single batch (B=2, T=64), one seed, CPU — the *ordering* and the sign are robust, the third
decimal is not.
