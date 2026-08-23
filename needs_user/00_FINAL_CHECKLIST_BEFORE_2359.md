# Final checklist — the four things that must be true at 23:59

*Two are done and verified. **Two are yours and cannot be done by the agent.** Ordered by how badly
each fails if missed.*

---

## 1. ❗ MAKE BOTH REPOSITORIES PUBLIC — yours

They are **private right now.** A grader following the links in `report.md`'s first paragraph gets
**two 404s and nothing else matters.**

- GitHub: **`Arsenii324/tlab-looped-transformer`** → Settings → General → Danger Zone → Change
  visibility → Public
- Hugging Face: **`Arsen4ikVar/tlab-looped-transformer`** → Settings → Change visibility → Public

**Verify after flipping**, in a logged-out browser or private window:
`https://github.com/Arsenii324/tlab-looped-transformer` and
`https://huggingface.co/Arsen4ikVar/tlab-looped-transformer`

## 2. ❗ ROTATE THE WANDB API KEY — yours

The key was committed to local git and uploaded to Yandex DataSphere earlier in the project. **It has
been scrubbed from history and the repo passes a secret scan over 121 commits — but scrubbing does not
un-send it**, and the repository goes public in step 1.

→ https://wandb.ai/authorize (or Settings → API keys → revoke and regenerate).
**Do this before or immediately after step 1**, not after.

## 3. ✅ DONE — the parameter-cap note is live on the Hugging Face model card

*This is the single most catastrophic misreading available*: a grader who checks the hardest
constraint the obvious way runs `sum(v.numel() for v in state_dict.values())`, gets **10,899,616**
against a **10M cap**, and concludes the submission is disqualified. The real count is **9,064,608** —
the tied embedding appears under two keys (`9,064,608 + 4096×448 = 10,899,616`, verified exactly).

**Verified live at 22:31** by downloading the card back from Hugging Face and diffing it: identical to
local, note present. Also in `submission/README.md` and `submission/METHOD.md` §1.

## 4. ✅ DONE — all gates green on the current commit

```
src/test_model.py            13/13 PASS      src/check_caveats.py --strict    0 missing
src/test_plateau.py           8/8  PASS      src/check_crossref.py --strict   0 orphans
src/headline.py check        consistent      src/make_inventory.py            generated from artifacts
```

---

## What ships

**The 90M control** — `checkpoints/full_control90_kaggle`, CE 3.6599 / **ppl 38.86** / bpb 1.5829,
band loops 6–17, 9,064,608 params, 89,999,360 tokens.

**Not** the norm-penalty arm, despite its better perplexity (37.52). The reasons are measured and
stated on the model card itself: 88% of its loop-gain advantage is loop-1 damage (`ΔCE@1 = +0.2263`),
its band narrows [6,17] → [6,14], it is the only arm whose map converges — the regime the report
argues against — and it carries a clipping confound the artifacts cannot resolve.

## Reading path for a grader

`submission/README.md` → whichever of the eight documents they want →
`report.md` for the evidence behind any number.

**If they read only one more thing, `reviewer_answers/29` is the honest one**: every attack we can see
against our own work, including §6 — the numbers that will **not** reproduce from the shipped
artifacts, and why each divergence is legitimate.
