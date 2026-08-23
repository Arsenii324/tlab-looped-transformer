# papers/ — the sources every external claim was verified against

Included so a reviewer can check citations rather than trust them. `VERIFICATION.md` marks each claim
VERIFIED / SECOND-HAND / REFUTED; this directory holds the text those checks were made against.

- `TASK_STATEMENT_ru.txt` — the task as given (Russian). `report.md` §2.0 maps every clause of it to
  where it is answered.
- `sources/<arxiv-id>/` — **LaTeX only** (`.tex`, `.bbl`, `.bib`) extracted from the arXiv tarball.
  Figures are excluded deliberately: a reviewer needs the prose and the numbers, and images would
  bloat the bundle without being diffable.

## Present (14) — claims checked against the paper's own LaTeX

| id | short name | what it decided here |
|---|---|---|
| 2606.24898 | dilution / Lemma 2 | §4.3's mechanism is *their* Lemma 2; §4.3 reframed from discovery to replication |
| 2502.05171 | Huginn (recurrent depth) | the "8–12" ceiling is the **zero-shot** figure (20 with 1 example, 32 with 25–50) — §2 |
| 2510.25741 | Ouro | R=4 is a **stability** decision taken down **from 8**, not a measured ceiling — §2 |
| 2607.16051 | Loopie | R=2 is a **FLOP-allocation** decision — §2 |
| 2608.18230 | MixerLoop | **refuted** an earlier §8.1 draft of mine; corrected |
| 2606.04438 | IterMoE / IterAdaLN | the only loop-conditioning mechanism surviving §3.4's function-vs-table rule |
| 2607.15456 | LLA | KV is a codec; final-loop reuse "collapses GSM8K generation to zero" — §4.8 |
| 2607.27656 | SCSE | step-conditioning numbers used in §3.4 and §8.0 |
| 2605.26733 | STARS | norm **placement** taxonomy; peaks at 4 recurrents — §8.0 |
| 2608.03624 | LoopMTP | BPB degrades with loop count without MTP; aggregation gate — §8.0 |
| 2606.18524 | residual scaling | `ε=λ/(N√L)`; LM transfer validated at N≤8 — §5.0 |
| 2204.03276 | PALBERT | **caught my Q-exit head in PonderNet's weaker configuration**; §4.7 now uses their best row |
| 2509.26314 | LTO | latent classifier predicts correctness from partial trajectories — §4.7 |
| 2605.23872 | Training-Free Looped | pre-norm block as a forward-Euler step — §4.6 |

## Absent (verified another way, and flagged here rather than hidden)

| id | short name | how it was checked |
|---|---|---|
| 2311.12424 | Looped Transformers are Better at Learning Learning Algorithms (ICLR 2024) | **fetched from arXiv HTML this session.** Bears directly on §4.9 — it is the **prior art for that section's mechanism**, and §4.9 is repositioned in-text as supplying the constants, not the mechanism |
| 2511.08577 | Think-at-Hard | fetched from arXiv HTML. Three of four relayed quotes confirmed verbatim; a relayed "over 73%" is actually **"over 85%"** in the paper, corrected in `VERIFICATION.md` |
| 2605.09165 | Sparse Layers | **SECOND-HAND**, not checked. Its claim — that *dense* looped models scale worse than sparse ones — is the live threat to §3.3 and is named as untested in `DECISIONS.md` |
| others | 2503.08524, 2603.15031, 2603.21365, 2606.22325, 2607.14427 | secondary references; see `VERIFICATION.md` for each one's status |

**One deliberate asymmetry worth flagging to a reviewer:** the two papers that most constrain this
report's novelty claim (2311.12424 and 2511.08577) are in the *absent* column. Both were found late,
both were verified from arXiv HTML rather than tarball, and both are cited in-text. They are listed
here rather than quietly omitted precisely because they are the inconvenient ones.
