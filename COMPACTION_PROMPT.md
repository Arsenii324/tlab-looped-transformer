# Suggested custom compaction prompt for this project

Paste this as the compaction instruction if the harness allows one. It targets what this project
has actually lost across compactions, rather than generic summarisation.

---

Summarise this session for a successor who must finish a research submission by 23:30 today.

**Preserve verbatim, these are load-bearing and expensive to re-derive:**
- Every **number** that appears in `report.md`, with which artifact produced it.
- Every **retraction / withdrawal**, what triggered it, and **what survived** it. Three landed today
  and the successor must not re-assert any of them.
- Every **pre-registration** still outstanding and its falsifier condition.
- Any **job ID**, and whether its weights were returned.
- The **standing user constraints** (no push/upload without explicit say-so; don't delete prior work;
  §1 of the report is the user's to write, not the agent's).

**Preserve the reasoning, not just the conclusion**, for anything currently contradicted or uncertain
— the successor needs to know *why* a claim is held weakly, not just that it is.

**Do not compress away:**
- The unknown-knowns list (`OPS.md` §7b, 24 items) or its two meta-patterns.
- Which numbers are *unverified* or *second-hand*, and which were delegated to another agent and not
  re-measured.
- The fact that `report.md` has never been read end to end.

**Explicitly note what is NOT done**, including things that look done: the git push and HF upload have
never run, no remote is configured, and the fresh-clone dry run is stale.

Point the successor at `OPS.md` §0-PRE first, then `reviewer_answers/16_WHOLE_STATE.md`.
