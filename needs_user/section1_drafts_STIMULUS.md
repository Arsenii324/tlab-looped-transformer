# Three §1 drafts — STIMULUS ONLY, not for the report

**Read this first.** You asked for these after I declined once, so here they are — but the reason I
declined still stands and you should hold it while reading: **§1 is graded as your own idea
generation, and the task cautions specifically about LLM involvement there.** These are written from
`LOG.md`'s chronology, so the *events* are yours; the framing, emphasis and vocabulary are mine.

The useful way to use them is the one the reviewer proposed: **react, don't select.** If one makes you
think "no, that's not what happened" or "that buries the thing I actually care about", that reaction
is the signal — it is information about your account that neither of us could state directly. Then
write §1 yourself and throw these away. If instead you find yourself editing one of them into place,
the section stops being yours, and that is the outcome worth avoiding.

None of these is in `report.md`. §1 there is still empty.

---

## Draft A — narrative ("I expected X, the arms said Y")

I started from the assumption everyone starts from: that a weight-tied model stops improving with
depth because it converges. Loop the same block enough times and it settles into a fixed point, after
which further iterations are arithmetic without information. That is what the DEQ framing says, it is
what the task's own premise says, and it is what I built the first diagnostics to measure.

It is not what happens. The winning configuration never contracts at any depth — the Jacobian's
spectral norm runs 1.7019 → 1.0015 from loop 2 to loop 64, approaching neutrality from *above* and
never crossing it — and it saturates at loop 8 anyway. Saturation without convergence. The state does
not settle; it travels along a nearly straight ray whose norm grows linearly while the useful step
stays roughly constant, so the part of the motion the readout can see decays as 1/t. The model is not
stuck. It is diluting.

That reframing cost me the intervention I had planned and pointed at a different one. If the problem
is how the state *travels*, then controlling the travel should help — so I clamped the radius, gated
the update convexly, and rescaled the residual branch. All three relocated where the optimum sits and
none of them raised it. Three independent ways of changing the trajectory, three nulls.

What did move was the loss. The depth at which a looped model stops improving turns out to be a
roughly fixed fraction of the depth it was *trained* at, and the fraction is set by which loop indices
the objective asks about. Supervise densely and it sits near half the trained depth; supervise only
the final loop and it moves to nearly all of it. That is not a property of the architecture at all.

So the answer I arrived at is a training schedule rather than a mechanism: train deep, and move the
supervision to the end for the last tenth of training. It costs no parameters.

---

## Draft B — method-first ("the supervision schedule sets the depth")

**Where a looped model's useful depth sits is set by the supervision schedule, not by its dynamics.**
Everything below is downstream of that sentence.

The evidence is a double dissociation. On the dynamics side, three independent interventions —
inference-time radial clamping, a learned convex gate with a fixed-`g` sweep beside it, and
`ε=λ/(N√L)` residual scaling — all relocate the optimum without raising the ceiling. On the
supervision side, one knob moves it reliably: dense supervision puts the useful-depth band at 0.50–0.71
of trained depth, terminal-only at 0.98–1.09, measured across three schedules and reproduced on two
devices with the band identical to the digit.

The knob is not a dial, which was the surprise. Supervising 2, 3, 5 or 8 loops behaves alike;
supervising exactly *one* is different. So a partial dose cannot be bought in density — but it can be
bought in **time**. Train dense, then switch to terminal-only for the final 10–25% of steps. Against a
control run in the same job, that beat the dense baseline on loss at both seeds while widening the
useful band and raising loop gain. Reversing the order — the same exposure placed at the *start* —
produces no depth effect at all and the worst loss in the series, which is what makes the mechanism
specific rather than mysterious: **the final phase decides, and later dense training erases it.**

At a deep schedule it compounds. Training at 32–48 loops with the last quarter supervised terminally
produces a model whose useful band runs from 32 to 64 loops — every depth in that range within 0.01
nats of its best, and still within 0.01 at 64, which is a third beyond anything it ever trained on.

The method is a loss schedule. It adds no parameters, so there is nothing in it whose benefit can die
as the model grows.

---

## Draft C — honest-failure ("I set out to make many loops pay, and here is what stopped it")

The task asks for low perplexity obtained *by exploiting many loops*, and the honest headline is that
those two goals fought each other for most of this project.

The single largest gain I measured was not an idea. Finishing the token budget — 46M to 90M tokens —
bought 0.39–0.42 nats. Every architectural intervention I tested across nineteen experiments moved
between 0.002 and 0.19. Perplexity is bought with data; that was true at the start and stayed true.

Worse, the configurations that made the loop *matter* were usually the ones that made the model
*worse*. Adding an unshared prelude — the topology every reference implementation uses — bought 0.355
nats and rendered the model completely depth-inert: identical loss from 1 loop to 96. Training deeper
raised loop gain seventeenfold while making absolute loss worse past L=8. I found four instances of
this split and asserted it four times before finally testing it as a correlation over all 43 arms, at
which point it came out at −0.081 and I had to demote my own most-repeated claim.

Several things I believed turned out to be artifacts. A headline finding was retracted entirely when
the same arms trained fine on different hardware. The statistic I used to locate every optimum turned
out to be decided, in 63 of 82 curves, by margins an order of magnitude below the noise — and I only
know the noise because two runs I had never intended as replicates turned out to be the same
configuration.

What survives is narrower than I hoped and better founded than I expected: the ceiling belongs to the
loss, not the dynamics, and one training-schedule change moves it. Whether that survives a budget
larger than the one I could afford, I do not know, and the report says so.
