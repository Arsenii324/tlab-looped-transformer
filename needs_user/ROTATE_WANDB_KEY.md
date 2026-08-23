# Rotate the wandb API key

**What happened.** The key was written in plaintext into 18 DataSphere `config.yaml` files (as
`export WANDB_API_KEY=...` inside the job `cmd`), those files were committed, and the job configs were
uploaded to Yandex DataSphere on every submit. I found it at 11:40 on 2026-08-23 while debugging an
unrelated probe-job failure.

**What I already did.**
- Scrubbed all 18 configs; they now read the key from the launching shell (`"$WANDB_API_KEY"`).
- Created branch **`review`** — a single squashed commit with **0 commits touching the key** and
  **0 occurrences in the diff**. This is the branch to review or push.
- `submission` and tag `main-backup-20260823` still contain it in history and **must not be pushed**.
- Nothing has been pushed to any remote.

**What only you can do: rotate it.** Scrubbing a repo does not un-send a secret. It went to Yandex
DataSphere in ~18 job submissions and sat in local git history. Generate a new key at
`wandb.ai/authorize`, revoke the old one, and keep the new one out of any file that git tracks.

**Convention going forward (already applied):** the key is passed via the environment at launch —
`WANDB_API_KEY=$(cat ~/path/to/key) datasphere ... job execute ...` — never written into a config.
