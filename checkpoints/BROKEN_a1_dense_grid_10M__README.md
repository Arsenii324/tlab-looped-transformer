# QUARANTINED — this artifact is a vocabulary mismatch, not a result

`BROKEN_a1_dense_grid_10M__VOCAB_MISMATCH.json` is a local dense-grid re-evaluation of the
`rec_dense_s2` / `rec_sw90_s2` DataSphere checkpoints. **It is invalid and nothing may be derived
from it.**

**How it was caught:** best CE reads **9.2692** and **9.4278**. Chance for this vocabulary is
`ln(4096) = 8.3178`. **A number above chance is not a model result** — and the curve's optimum sits
at the last grid point, the signature of a model that is not reading its own vocabulary.

**Cause:** the DataSphere training kernel trains and uses its **own** BPE and **does not return
`tokenizer.json`** with the checkpoint (`/tmp/ds_rec2/` contains only the two `.pt`, `results.json`
and logs). Evaluating those weights locally against the shipped `configs/tokenizer.json` is exactly
the failure `src/check_tokenizer_identity.py` exists to catch: **a mismatch does not raise, it reports
CE near chance and looks like a broken model rather than a broken setup.**

**What is NOT affected:** every claim in `report.md` §4.23e about these arms comes from the **in-job**
`val_curve` computed inside the DataSphere job with that job's own tokenizer (`rec_dense_s2` best
**4.4907**, `rec_sw90_s2` best **4.6025**). In-job paired comparisons are valid. Only *cross-job local
re-evaluation* of these checkpoints is impossible.

**Consequence for the open queue:** the denser-eval-grid check on annealing's `end 16 → 24` **cannot
be run locally on these checkpoints.** It needs either the DataSphere job's tokenizer or a re-run of
the eval inside a DataSphere job. Recorded rather than silently dropped.

*Kept rather than deleted, per this project's rule that superseded and broken artifacts stay visible.*
