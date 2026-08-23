# RETRACTION — 2026-08-23 11:25 · **you were right about 73%, I was wrong**

**Read this even if you skip everything else in this folder.** I told you, in three separate
documents, that a number you relayed was wrong. It was not. Mine was.

## What I claimed

In `02_deliverable_completeness.md` and again in `01_full_task_statement.md`:

> *"The 73% figure is wrong. The paper says 'over 85% of next-tokens are correctly predicted at the
> first iteration'."*

I also entered it in `VERIFICATION.md` as **REFUTED AS QUOTED**, and flagged it to you as *"the kind
of relayed number that becomes a citation error."*

## What the source actually says

You downloaded the tarball. `papers/sources/2511.08577/3_method.tex`, line 206, verbatim:

```latex
However, we find that over 73\% of next-tokens are correctly predicted at the first iteration
(Table~\ref{tab:oracle_policy}).
```

**Your 73% was correct.** `85` appears in the v3 source only as unrelated table cells in
`4_experiment.tex` (85.4, 85.0, 85.8, 85.6 — benchmark scores, not first-iteration accuracy).

## How I got it wrong, which is the part worth your attention

I never read the paper. I read the arXiv **HTML** through a *summarising* fetch — a small model reads
the page and answers my question about it — and it returned 85%. I then treated that answer as a
primary source, and asserted it against a figure you had relayed correctly.

**The lesson is not "web fetches are unreliable."** It is that a summariser sat between me and the
text, I could not see what it had done, and I used its output to correct someone else about
citation hygiene. That is worse than a quiet error: it moved a wrong number *into* your model of the
literature, under the authority of a check I had not actually performed.

## What changed as a result

- Retracted in `VERIFICATION.md`; the row now reads VERIFIED-verbatim-in-v3-LaTeX with the retraction
  stated inline.
- Added to `report.md` §6.0 as **row 22** — it belongs in the failure table, not in a footnote.
- **`papers/sources/` now ships with the repo** (14 → 16 papers, including this one and 2311.12424),
  so every citation claim can be checked against LaTeX rather than a summary. That directory exists
  largely because of this error.
- Re-ran the same check on **2311.12424** from its tarball rather than HTML. Both quotes confirmed
  verbatim: *"…consistently discovers a fixed-point solution that saturates prior to the trained
  iteration $b$"* and *"…occurs due to the loss objective, which requires the looped transformer to
  match the target within $b$ steps."* That one stands, and §4.9 remains repositioned around it.

## Process note

I had inserted this retraction by editing `00`, `01` and `03` — files already sent to you. Those
edits would never have reached you. **From here, corrections go in new numbered files rather than
edits to sent ones**, and `reviewer_answers/README.md` now says so.
