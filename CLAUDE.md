# CLAUDE.md — T-Lab looped-transformer test task

Standalone project. Not part of the Huginn/Geometry-of-Reasoning-Trajectories work in the sibling
directory — do not read that project's CLAUDE.md as governing this one; its rules are scoped to
Huginn-3.5B and Barannikov's H1-H3. This file restates the parts of that project's working style
that are general-purpose, in this project's own terms, because they earned their keep there.

## The task, one paragraph

Pretrain a looped (weight-tied, block-applied-r-times) transformer on next-token prediction over
FineWeb. Budget: <=10M params, <=100M training tokens. Base the architecture on Qwen3 and modify as
needed. Loop count: more is better. Optionally implement early loop-exit. Goal: lowest validation
perplexity achieved BY exploiting many loops, with a design that plausibly keeps working at larger
scale (a mechanism whose benefit is a fixed matrix/table that stops mattering as params grow does
not count). Grading has two parts: idea quality, and implementation/verification quality without
losing track of what an agent wrote. Full spec in `../files/test_task_tlab.txt` (Russian).

**Division of labour, stated because it changes how I should behave.** The task explicitly grades
the *user* on idea generation and explicitly warns against using an LLM for that part. My job is
Phase 2: build a platform that can test ideas fast and correctly, run the ablations that are
already strongly motivated by this workspace's own prior evidence (see PLAN.md), and implement
whatever the user specifies. Where I propose a mechanism myself, it must be labeled as mine and
grounded in something checkable — not offered as if it were the graded idea.

## Check the premise before building on it

Before trusting a per-loop diagnostic, ask what it actually reads — logits at a fixed position,
a probe on the state, or the training loss itself — and whether that channel could be flat/rising/
falling for a reason that has nothing to do with the thing you want to measure. This bit the sibling
project hard (a per-loop capability curve that was actually reading "is the model about to write
'The'"). The fix that generalizes: prefer instruments that don't touch a readout at all (state norm,
online contraction rate) alongside the ones that do, and prefer teacher-forced loss over any
generation-based proxy unless the generation IS the object of interest.

The cheapest check that could refute a design choice comes before scaling it up. Smoke test on CPU
before MPS, MPS before Kaggle, a reduced token budget before the full 100M.

## Instrument generously, abstract minimally

Log per-loop loss, per-loop hidden-state norm, per-loop predictive entropy, and an online
contraction-rate estimate on every run, not just the ones you think will be interesting — this is
cheap relative to the training compute itself and is where the report's honest negative results will
come from. Code stays minimal: no config framework beyond a plain dataclass, no abstraction for
model families this project doesn't run, no plugin layer for a training loop that runs a dozen times.

## Instruments and hypotheses

A screening result on a reduced token budget may motivate scaling a config up. It may not, by itself,
retire a design axis — a negative screening result and a negative full-budget result are different
findings, and PLAN.md says which is which as they land.

## The documents

Three, matching the sibling project's pattern, scoped to this one:

- `PLAN.md` — design decisions and the phased roadmap, with rejected alternatives and what would
  flip each choice. Updated when a decision changes, not appended to indefinitely.
- `LOG.md` — one line per step as it happens: what ran, what it showed, what's next. Ledger, not
  prose.
- `report.md` — the actual deliverable. Has a section reserved for the user's own idea-generation
  narrative (not to be filled in by me) and sections for measured results (mine to fill in as data
  lands).

## Working style

State assumptions explicitly, especially ones that would be expensive to be wrong about (vocab size,
compute path, whether full backprop-through-loops is affordable at this scale). Say what was rejected
and why. Raw output over printed verdict — before writing a number into report.md, open the actual
run output it came from.

Nothing here is pushed to a remote (GitHub or Hugging Face) without the user's explicit say-so —
those are the task's literal submission targets and that action is his to take or approve.
BENJAMIN-PLUS MODE ACTIVE

# Benjamin-Plus

Every request you send re-reads the whole conversation so far. The bill is
steps × context, not words. Save by taking fewer steps and keeping bulky tool
output out of the transcript — never by skimping on the work itself. Solve the
task exactly as you otherwise would; these rules change how you look things
up, not what you build.

**1. Recon in one pass.**
Before changing anything, collect every independent fact in a single step:
chain probes with `;` and label the sections
(`echo == layout ==; ls -la; echo == deps ==; head -30 requirements.txt`),
or issue several tool calls in one message. A second lookup round is for
questions the first round's answers created. Copying a convention (a DSL,
schema, or file format)? Sample two existing examples of the exact construct
you will write, not one.

**2. Look through a keyhole.**
A command that only inspects ends with a limiter: `| head -50`, `| tail -20`,
`grep -m 20`, `wc -l` before contents, Read with offset/limit. Size unknown?
Measure first, then read the slice you need. Read a file whole only when you
are about to edit it or copy from it verbatim — truncating data you will
transform corrupts output, so keyhole rules apply to inspection, never to
ingestion. If a peek was too narrow, take exactly one wider look.

**3. Probe the environment once.**
Before running code with several dependencies, test them in one probe
(`python3 -c "import x, y, z"`; `command -v tool1 tool2`), and install
everything missing in one command — not one traceback at a time.

**4. Green means the task's own check.**
If the task names verification commands, those are the check: run them
exactly as written, and green means exit status zero. A failure you judge
environmental (missing package, compiler, or tool) is still your failure —
fix the environment and re-run; "unrelated to my change" is not a green
check. The same check failing twice on the same approach means the approach
is wrong: name one alternative and try it before patching the next symptom.
When the check passes, stop: no victory laps, no re-reading files you just
wrote. Close with at most two lines.

**5. Polling is a step.**
A running command that hasn't finished is not new information — but every
status check re-reads the whole conversation. If your harness returns while a
command is still running, wait in large slices (30 seconds or more; minutes
for builds and test suites) before checking again. Never re-poll at
one-second intervals, and never send empty input just to peek. Where
execution blocks until completion, this rule costs nothing.

Never build a verification harness, test suite, or checker the task didn't
ask for — verify stated properties with the shortest command that measures
them, and spend the saved steps on the task itself. If saving a step risks a
wrong result, spend the step: efficiency never outranks correctness, a
failing check, or anything the task explicitly asks you to produce.
