# §4.12's "loop gain emerges with tokens" — the LR confound, tested and EXCLUDED

**From:** fork worker. **Status:** finding is safe; §4.12 can be strengthened, not corrected.

## The gap

§4.12 claims loop gain *emerges with training tokens* (0.02 @0.44M → 0.24 @13.57M) and uses that to
correct §4.2 and to argue every screening-scale loop-gain number is "measuring the token budget."
**Every run here uses a cosine LR schedule spanning its own `total_tokens`** (`lr_at()` in
`src/train.py`, 3e-3 → 3e-4). So token count is structurally confounded with LR-schedule position,
and the alternative reading — *loop gain emerges as the LR decays* — fits the same trajectory. The
parent already knows LR decay drives the ‖h‖ peak-then-fall (§4.3) but never connected it here.

This matters because the two readings give opposite advice: if it were LR, "more tokens at constant
LR" would buy nothing and the "saturates at 10–15M tokens" reading would be wrong.

## The test

Same config, same seed (`no_state_renorm`), two runs whose **only** structural difference is the
cosine span: screening `total_tokens=1,188,000` vs full `total_tokens=15,840,000`.

**Matched tokens, very different LR:**

| run | tokens | sched frac | LR | loop gain |
|---|---|---|---|---|
| screening | 985,088 | **82.8%** | **5.22e-04** | 0.0525 |
| full | 985,088 | **6.2%** | **2.99e-03** | 0.0593 |

A **5.7× LR difference** produces a **0.007-nat** gain difference.

**Matched schedule fraction, very different tokens:**

| sched frac | screening | full | Δ gain |
|---|---|---|---|
| 30% | 0.0076 (395k tok) | 0.1250 (4.77M tok) | **+0.1175** |
| 58% | 0.0357 (690k) | 0.1726 (9.19M) | **+0.1370** |
| 75% | 0.0504 (887k) | 0.2003 (11.90M) | **+0.1500** |

**Correlations over the overlapping token region (n=36):**

| | r |
|---|---|
| gain vs log tokens | **+0.845** |
| gain vs log LR | +0.190 |
| gain vs schedule fraction | +0.160 |
| **partial: gain vs log LR, given log tokens** | **−0.170** |

## Conclusion

**Tokens, not LR.** Once token count is known, LR adds essentially nothing — and the small residual
is *negative*, i.e. the opposite sign from the "low LR causes gain" hypothesis. §4.12 stands, and can
now say the obvious confound was tested and excluded rather than leaving it unaddressed.

Suggested one-line addition to §4.12: *"The cosine LR schedule spans each run's own token budget, so
tokens and LR position are confounded by construction. Tested directly on two runs of the same
config differing only in cosine span: at matched tokens a 5.7× LR difference changes loop gain by
0.007 nats, while at matched schedule fraction a 12–13× token difference changes it by 0.12–0.15.
Partial correlation of gain with log LR given log tokens is −0.170. The effect is token-driven."*

## Caveats
- The two runs differ in `warmup_steps` (40 vs 150) — irrelevant at the token counts compared, both
  are long past warmup.
- Within-config, single seed each. The comparison is clean on the axis tested but is not a seed study.
- Reproduce: the numbers above come only from `checkpoints/screening_results.json` and
  `checkpoints/full_no_state_renorm_history.json`; no new compute was used.

## Unrelated observation, noted and dropped per scope
`checkpoints/_arm_configs/rs_depth_init_s0.json` still carries `total_tokens=1000` from the
smoke test. Harmless (the real run rewrites it), but it will look odd if anyone reads that directory.
