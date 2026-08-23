# Push + HF upload — ready, blocked on your say-so

**Everything below is verified and rehearsed. Nothing has been sent anywhere.** Standing constraint:
nothing goes to GitHub or Hugging Face without your explicit authorisation, and no git remote exists.

## What is already verified (2026-08-23 18:15, fresh clone of the ship branch)

Cloned `review` cold into a scratch dir, 670 files, and ran the repo's own gates:

| gate | result |
|---|---|
| `src/test_model.py` | **ALL PASS** — 13 checks incl. the 4 added today for the new arms |
| `src/test_plateau.py` | **ALL PASS** — incl. its deliberate falsification probe |
| `src/headline.py check` | **consistent**, 0 numbers missing, after repointing `HEADLINE.json` at the 90M control |
| `src/check_tokenizer_identity.py` on the **shipped** checkpoint | **PASS**, \|diff\| **0.0020** vs chance 8.3178 |

## The two facts that decide the commands

1. **Ship `review`. It is current** — `report.md` 463,908 B, carries the §0 abstract, has
   `requirements.txt`.
2. **NEVER push `submission`.** It is 5.5h stale (`report.md` 303,289 B, no `requirements.txt`) **and
   its history carries the wandb key that was scrubbed from `review`.**
3. **NEVER `git push --tags` or `--mirror`.** `refs/tags/main-backup-20260823` holds 4 blobs over
   GitHub's hard 100 MB limit (1.83 GB total). Per-branch pushes are clean — verified per-ref.

## GitHub — the exact sequence

```bash
cd ~/build-projs/barannikov-work/tlab-loop-transformer
git remote add origin <YOUR-REPO-URL>          # no remote is configured yet
git push -u origin review:main                 # ship branch -> their default branch
#   NOT: git push --all      (would push `submission`)
#   NOT: git push --tags     (would be rejected on the 100MB blobs)
```

Then confirm what actually landed, rather than trusting the push output:

```bash
git ls-remote --heads origin                   # expect ONLY main
```

## Hugging Face — one command, and it refuses rather than shipping something unverifiable

```bash
python src/upload_checkpoint.py checkpoints/full_control90_kaggle --repo <user>/<name>
```

`upload_checkpoint.py` was repaired today (§6.0 rows 26/27). It now ships `configs/tokenizer.json`
and `model.py` alongside the weights, generates the model card from the checkpoint's **own**
`eval_*.json`, prints the gate command with the number substituted, and **raises rather than
uploading** if either file is missing. It has been dry-run verified and **never executed against the
network**.

**Which checkpoint — this is D3 and it is yours.** My recommendation is the **control**
(`full_control90_kaggle`, ppl 38.86): it is the configuration §3.5 describes, and the norm-penalty arm
(37.52) buys its perplexity with **88% loop-1 damage**, a narrower band, is the only arm whose map
converges — the regime §2 argues against — and now carries an **open clipping confound** the artifacts
cannot resolve. The control is the arm with no confound on either axis.

## After uploading, verify against the DOWNLOADED artifact, not the local one

```bash
huggingface-cli download <user>/<name> --local-dir /tmp/hfcheck
python src/check_tokenizer_identity.py /tmp/hfcheck --expect-ce1 3.9622
```

Expect **PASS** with \|diff\| ≈ 0.002. A vocabulary mismatch lands at chance ≈ **8.32**, not near the
tolerance — which is the whole reason this gate judges against chance rather than a fixed epsilon.

## Also still yours

- **§1** — reserved and empty. Raw material assembled in `SECTION1_RAW_MATERIAL.md` (record, not a
  draft). It is the only carrier of criterion 1.
- **Rotate the wandb key** — `ROTATE_WANDB_KEY.md`. Scrubbing the repo did not un-send it.
