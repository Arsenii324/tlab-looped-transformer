> **Dated working record — not a maintained document.** Last committed 2026-08-23; the text itself may be older. Kept intact rather than rewritten, because this project's rule is that superseded statements stay visible with their corrections. **Current numbers are in [`submission/`](submission/) and [`report.md`](report.md); where this file disagrees with them, they win.** See the repository README's *How to read this repository* table.

# subagents/ — communication channel

Drop files here to talk to the main agent (and vice versa). Convention:
- `from-<name>.md`  — a subagent's findings for the main agent
- `to-<name>.md`    — the main agent's brief/answers for a subagent

The main agent polls this directory. Keep findings concrete: what you checked, what the source
actually says, and what it changes. State explicitly what you did NOT verify.

**Current project state for any incoming subagent:** read `INDEX.md` first, then `OPS.md` (live
status), `STATE_FOR_REVIEWER.md` (full technical dump incl. self-identified weaknesses), and
`VERIFICATION.md` (which external claims are source-verified vs relayed). `QUEUE.md` holds open
items. The report is `report.md`; its §1 is reserved for the user and must not be written by an agent.
