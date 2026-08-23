# DRAFT for `report.md` — gated injection. NOT applied; parent to verify and place.

Proposed home: **§4.1b**, immediately after §4.1, because it completes that section's own axis.
Results below are filled from `checkpoints/gated_inject_results.json` — verify against that file
before applying (this project's rule: open the artifact, don't trust a summary).

---

### 4.1b The third cell on the normalisation axis — gated injection

**§4.1 swept an axis with a missing option, and the missing one is what the field actually uses.**
That section compared inter-loop RMSNorm (`state_renorm=True`) against nothing (`False`) and found
the second worth **−0.744 nats**, the largest single effect in this project. Stated that way it reads
as "don't normalise the carried state". But the two arms are not the only choices, and neither is the
one the reference implementations make. Parcae and *Looped Transformers Done Right* both use a
**diagonal state-space write** — a learned per-channel decay on the carry plus a learned write
strength:

```
delta = softplus(b)                  per-channel WRITE strength
alpha = exp(-delta * exp(a))         per-channel CARRY decay, in (0,1)
h_in  = alpha * h + delta * e        replaces  h_in = h + e
```

This is a genuinely different cell rather than a fourth arbitrary knob: it **bounds ‖h‖ without
projecting the state onto a sphere**, which is exactly what neither `state_renorm=True` (which
projects, and goes inert — §4.3) nor `state_renorm=False` (which bounds nothing) does.

**Two of this report's own findings say this is the cell to test.**

1. §4.3 measures `‖e‖/‖h‖` falling from 3.2e-3 to 1.3e-4 and calls the re-injected input *drowned*.
   That is **a consequence of plain addition**, not a fact about looped transformers: `e` has fixed
   magnitude, `‖h‖` grows without bound, and nothing in `h + e` controls the ratio. Under the gated
   form `alpha < 1` bounds the state at `delta·‖e‖/(1−alpha)`, making the write ratio a *learned*
   quantity rather than an artefact of how far the state has drifted.
2. §4.3 also shows the state never settles — it drifts logarithmically (log-drift R² 0.986 against a
   power law's 0.748). A carry decay is the only mechanism on this axis that could stop that without
   the contraction-to-inertness `state_renorm=True` produces.

**Cost and construction.** `2·H = 896` parameters (**0.0099%**; total 9,065,504, under the 10M cap),
zero extra FLOPs. Initialised **at the additive model** — `alpha = 0.9999939`, `delta = 1.0`, verified
to a relative forward difference of **1.2e-05** — so the arm begins as the control and must *learn*
to decay. That makes it a strictly-larger hypothesis class rather than a different model, the same
discipline the scale clock uses. **The `W_in` projection on the write that both reference
implementations also carry is deliberately omitted** (it would add 200,704 params), so that anything
measured here is the decay mechanism and not 200k extra parameters. The projected variant is
untested and this report does not speak to it.

**Read order, pre-registered in `src/run_gated_inject.py` before the run, and the first item is not
the loss:** (1) did the model take the parameter — `alpha` staying ~1 means it declined a
strictly-larger hypothesis class and everything else is moot; (2) `‖e‖/‖h‖` against the control,
which is the mechanism the form claims to fix and is the **primary result whether or not CE moves**;
(3) the `‖h‖` trajectory; (4) plateau/onset, grid-matched; (5) CE, **reported and not used to
decide** — 896 params against a measured MPS replicate floor of 0.031–0.068 nats cannot be resolved
on loss at 2.5M tokens.

<!-- RESULTS TABLE — fill from checkpoints/gated_inject_results.json -->

**A replicate that arrived for free.** `gi_additive` is config-identical to `sc_ctrl` from the
scale-clock run (same reference-derived config, same seed, same eval cadence, separate invocation).
That makes the pair a **same-config MPS replicate** — which §4.15 flagged as missing when it admitted
the annealed-arm noise floor was never measured. Their difference is a floor measurement for this
configuration, obtained at no extra cost.
