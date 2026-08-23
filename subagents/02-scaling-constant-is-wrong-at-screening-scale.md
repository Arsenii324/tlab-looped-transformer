# The 0.398 nats/e-fold constant is the wrong rate where the report uses it most

**From:** fork worker. **Status:** a correction the parent has partly anticipated but never
quantified, and one place where the caveat is missing entirely.

## The gap

`0.398 nats/e-fold` appears in 4 places in `report.md` and does two jobs: projecting 46M→90M (§8,
correct regime) and **token-correcting arms at ~1M tokens** (§4.1 screening table, §4.1 seed spread).
It was measured over **14.6M → 46.0M**. Loss-vs-log-token slopes flatten with scale, so it is the
wrong local rate for the ~1M-token corrections — and those corrections are what flip conclusions.

§4.1 says of the screening table: *"the loss curve is steeper at ~1M tokens than in the 15–46M range
this rate was measured over, so those are lower bounds."* True, never quantified. **The seed-spread
passage carries no such caveat at all** and states `0.117 nats` as a bare result.

## Measurement

Same config (`state_renorm=False`, U[4,32]), one instrument throughout (in-training final
`val_curve`, so no post-hoc/in-training mixing):

| run | tokens | best CE |
|---|---|---|
| screening | 985,088 | 6.0281 |
| full-local | 14,600,192 | 4.4034 |
| kaggle | 45,975,552 | 3.9542 |

| interval | e-folds | **nats/e-fold** |
|---|---|---|
| 0.99M → 14.6M | 2.70 | **0.603** |
| 14.6M → 46.0M | 1.15 | **0.392** |

**The low-token slope is 1.54× the high-token slope.** Instrument-insensitive: the same 14.6M→46.0M
interval gives 0.398 from post-hoc dense evals and 0.392 in-training, so the 0.603/0.392 gap is a
real regime difference, not an artifact of which eval produced it. All three runs completed their own
cosine, so these are valid cross-run comparisons (not the within-run slope, which is LR-confounded —
see `01-lr-confound-in-4.12.md`).

## What changes

### Screening table (§4.1)
| arm | raw Δ | token ratio | corrected @0.398 (report) | corrected @0.603 (local) |
|---|---|---|---|---|
| `no_state_renorm` | -0.7442 | 1.111× | -0.7023 | **-0.6807** |
| `inject_none` | +0.1790 | 1.111× | +0.2209 | **+0.2425** |
| `no_depth_init` | +0.1416 | 1.111× | +0.1835 | **+0.2051** |
| `truncate8` | -0.0157 | 1.339× | +0.1005 | **+0.1603** |
| `fixed_loops16` | -0.1284 | 1.339× | -0.0122 | **+0.0476** |
| `inject_concat` | -0.0158 | 1.000× | -0.0158 | **-0.0158** |

**`fixed_loops16` flips sign a third time.** Raw −0.128 → "null" at 0.398 → **+0.048 (worse than
center)** at the local rate. The report currently calls it "null, not a win"; at the correct rate it
is a small loss. `truncate8` moves −0.016 → +0.10 → **+0.160**, strengthening the existing
"reversed — full BPTT wins" conclusion.

### Seed spread (§4.1) — the bigger one
| | corrected @0.398 | corrected @0.603 |
|---|---|---|
| seed 0 | −0.7024 | **−0.6808** |
| seed 1 | −0.5851 | **−0.6307** |
| **spread** | **0.117** | **0.050** |

**The token-corrected seed spread falls from 0.117 to 0.050 nats.** Both corrections move the two
seeds *toward each other*, because the token imbalances point in opposite directions and a larger
rate amplifies both. `state_renorm` is therefore even more seed-robust than §4.1 now claims, and the
noise floor used elsewhere to dismiss small effects drops again — from the original 0.25, to 0.117,
to ~0.05.

## Caveats
- Three points, so the slope is two intervals, not a fit. 0.603 is an *average* over 0.99M→14.6M;
  the true local rate at exactly 1M is probably steeper still, making these corrections themselves
  lower bounds — the same direction the parent already flagged.
- The screening run and the full run have different cosine spans. For **CE** this is a valid
  completed-run comparison (each ran its full schedule); it would not be valid for a within-run slope.
- I did not re-derive the §4.1 screening verdicts beyond the arithmetic above.

## Suggested action
Use ~0.60 for corrections at screening scale and keep 0.398 for the 46M→90M projection; state both
rates and why they differ. At minimum, add the lower-bound caveat to the seed-spread passage, which
currently has none.
