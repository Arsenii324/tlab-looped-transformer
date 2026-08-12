# LOG — one line per step, as it happens

2026-08-13 00:00  Recon: M2 Pro/12-core/32GB, MPS available, raw fp32 matmul 5.6 TFLOPS (2048^2,
  unfused). Kaggle CLI authenticated (non-Yandex fallback). Disk 41GB free of 926GB -- streaming-only
  for FineWeb, confirmed working (sample-10BT, 3 docs in 19s, schema ok). transformers==4.53.3 has a
  real Qwen3 reference implementation locally (modeling_qwen3.py) -- read directly rather than from
  memory: QK-norm is per-head on head_dim before RoPE, GQA, SwiGLU, no biases, pre-norm RMSNorm.
  barannikov-work/ itself is not a git repo; new repo created at tlab-loop-transformer/, no remote.
2026-08-13 00:05  Checked task's own references: Ouro (arXiv 2510.25741) and "Loop the Loopies!"
  (arXiv 2607.16051) confirmed. "Q-exit" -- no confirmable named source, treating generically.
  Found unsolicited: "The Readout Blind Spot in Looped Language Models" (arXiv 2606.24898) --
  independently motivates the state-renorm ablation axis already suggested by Huginn's own D23/D26.
2026-08-13 00:10  PLAN.md written. Five ablation axes fixed, each tied to a specific prior result.
  Coordinated once with sibling session (fork, working the paper report in the other repo) -- no
  shared files, received a courtesy summary of their D203/D204 eval-instrument findings, consistent
  with what's already in PLAN.md's eval section.
