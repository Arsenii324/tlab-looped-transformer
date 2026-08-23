#!/bin/zsh
# Durable overnight work queue. Lives in the repo (not /tmp) so it is version-controlled and
# reviewable. Design rules, each learned from a real failure in this project:
#   * STRICTLY SERIAL on the GPU. MPS is single-tenant here; two trainings at once both slows them
#     and risks the documented driver corruption.
#   * IDEMPOTENT. Every step has a "done check". Re-running the script re-runs nothing that finished,
#     so it is safe to launch again at any time.
#   * LOUD ON FAILURE. A step that fails is recorded with its tail, and the queue CONTINUES to the
#     next step rather than dying -- the failure mode to avoid is the whole queue stopping silently.
#   * NEVER touches published eval_*.json.
set -u
# NULL_GLOB is load-bearing: zsh treats an unmatched glob as a FATAL error (bash silently passes the
# pattern through). The per-arm eval loops below glob over directories that do not exist until their
# training arm has run -- `checkpoints/sc_*_s0` before scale_control, say -- and without this the
# script ABORTS there. That silently killed this queue three times tonight, each time after a
# different step, which is why it looked like a different bug each time.
setopt NULL_GLOB
cd "$(dirname "$0")"
LOG="${1:-queue_run.log}"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
wait_gpu() { while pgrep -f "run_supervision.py|run_scale_control.py|run_residual|paired_eval.py|cross_depth_kv.py|eval.py|exit_dump.py" >/dev/null; do sleep 60; done; }

step() {  # step <name> <done-check-file> <command...>
  local name="$1"; local done_file="$2"; shift 2
  if [[ -e "$done_file" ]]; then say "SKIP  $name (exists: $done_file)"; return 0; fi
  # Wait for the GPU before EVERY step, not once at the top. A single up-front wait has a race: the
  # other queue driver (queue.sh) launches run_scale_control only after run_supervision exits, and
  # this loop polls every 60s, so it can break out in the gap and then run GPU work alongside it.
  # MPS is single-tenant here; that contention is what the driver-corruption mitigation exists for.
  wait_gpu
  say "START $name"
  if "$@" >> "$LOG" 2>&1; then
    if [[ -e "$done_file" ]]; then say "OK    $name"
    else say "WARN  $name exited 0 but did not produce $done_file"; fi
  else
    say "FAIL  $name (exit $?) -- continuing to next step; tail:"
    tail -12 "$LOG" | sed 's/^/        /' | tee -a "$LOG" > /dev/null
  fi
}


say "QUEUE START"
wait_gpu

# --- item 6: cross-depth KV grid on the headline checkpoint (the user's own early-exit idea) ---
step "crossdepth/kaggle" checkpoints/full_no_state_renorm_kaggle/crossdepth_full_no_state_renorm_kaggle.json \
     python src/cross_depth_kv.py checkpoints/full_no_state_renorm_kaggle --loops 1,2,4,8,16,32,64 --n-seq 256 --batch-size 4
sleep 30

# --- paired scoring of the headline checkpoints (common random numbers, per-sequence CE) ---
step "paired/kaggle" checkpoints/full_no_state_renorm_kaggle/paired_full_no_state_renorm_kaggle.npz \
     python src/paired_eval.py score checkpoints/full_no_state_renorm_kaggle --max-loops 64 --batch-size 8
sleep 30
step "paired/local14M" checkpoints/full_no_state_renorm/paired_full_no_state_renorm.npz \
     python src/paired_eval.py score checkpoints/full_no_state_renorm --max-loops 64 --batch-size 8
sleep 30


# --- exit dump locally as a backstop if DataSphere never returns the npz ---
step "exitdump/local" checkpoints/full_no_state_renorm_kaggle/exitdump_full_no_state_renorm_kaggle.npz \
     python src/exit_dump.py checkpoints/full_no_state_renorm_kaggle --max-loops 64 --batch-size 8
sleep 30

# --- dense post-hoc evals of every new training arm, so nothing is compared on in-training numbers ---
for d in checkpoints/sup_*_s0 checkpoints/sup_*_s1 checkpoints/sc_*_s0 checkpoints/sc_*_s1; do
  [[ -d "$d" ]] || continue
  n=$(basename "$d")
  step "eval/$n" "$d/eval_$n.json" python src/eval.py "$d" --max-loops 48 --batch-size 4 --n-batches 15
  sleep 20
done

# --- cheap post-hoc analyses: no training, several with no GPU at all ---
step "argmin_anatomy" checkpoints/full_no_state_renorm_kaggle/argmin_anatomy.txt \
     zsh -c 'python src/argmin_anatomy.py checkpoints/full_no_state_renorm_kaggle/exitdump_full_no_state_renorm_kaggle.npz > checkpoints/full_no_state_renorm_kaggle/argmin_anatomy.txt 2>&1'
sleep 10

# clamp on the CONTRACTING checkpoint: does §4.6's rate-not-value story hold when the map contracts?
step "clamp/center" checkpoints/center/clamp_center.json \
     python src/radial_clamp.py checkpoints/center --max-loops 48 --n-batches 15 --batch-size 4
sleep 20

# dynamics on the sandwich arms: does a prelude/coda change the ray geometry?
for d in checkpoints/sand_P1R1C1 checkpoints/sand_P1R2C0 checkpoints/sand_P0R2C1; do
  [[ -d "$d" ]] || continue; n=$(basename "$d")
  step "dynamics/$n" "$d/dynamics_$n.json" python src/state_dynamics.py "$d" --max-loops 48
  sleep 20
done

# exit dump on the 14.6M checkpoint: is depth demand present earlier in training, or does it emerge?
step "exitdump/local14M" checkpoints/full_no_state_renorm/exitdump_full_no_state_renorm.npz \
     python src/exit_dump.py checkpoints/full_no_state_renorm --max-loops 64 --batch-size 8
sleep 20

# --- learned exit probe: makes §4.7's negative honest (hand-crafted rules were a weak search) ---
step "exit_probe" checkpoints/full_no_state_renorm_kaggle/exit_probe.txt \
     zsh -c 'python src/exit_probe.py checkpoints/full_no_state_renorm_kaggle/exitdump_full_no_state_renorm_kaggle.npz --k-obs 4 > checkpoints/full_no_state_renorm_kaggle/exit_probe.txt 2>&1'
sleep 10

# --- CALIBRATE the oracle headroom. §4.7's 0.3084 nats is the report's headline and is currently
# uncalibrated: a min over 64 correlated noisy values using the label. Nulls decide how much is real.
step "oracle_null" checkpoints/full_no_state_renorm_kaggle/oracle_null.txt \
     zsh -c 'python src/oracle_null.py checkpoints/full_no_state_renorm_kaggle/exitdump_full_no_state_renorm_kaggle.npz > checkpoints/full_no_state_renorm_kaggle/oracle_null.txt 2>&1'
sleep 10

# --- Q-exit head (PALBERT criterion). The task names Q-exit explicitly, so this is the mechanism
# most likely to be read closely. Runs both head variants PALBERT ablate.
step "qexit" checkpoints/full_no_state_renorm_kaggle/qexit.txt \
     zsh -c 'python src/qexit.py checkpoints/full_no_state_renorm_kaggle/exitdump_full_no_state_renorm_kaggle.npz > checkpoints/full_no_state_renorm_kaggle/qexit.txt 2>&1'
sleep 10

# --- sliding-window eval on the headline checkpoint: a fairer ABSOLUTE bpb, reported alongside ---
step "sliding/kaggle" checkpoints/full_no_state_renorm_kaggle/sliding_full_no_state_renorm_kaggle.json \
     python src/sliding_eval.py checkpoints/full_no_state_renorm_kaggle --loops 8 --stride 64 --n-tokens 400000
sleep 20

# --- terminal-only vs dense per-loop supervision: the largest untested axis in the design.
# Two papers say terminal-only wins on loss (Sharma&Vu Tab.2: 5.40 vs 6.04 at 44M; LoopFormer 10.91
# vs 11.60 at ~1B) while making intermediate exits unusable (their Tab.14). Measures that trade at 9M.
step "supervision_depth" checkpoints/supervision_depth_results.json \
     python src/run_supervision_depth.py --tokens 2500000 --seeds 0,1
sleep 30

# --- item 5: residual scaling arms (token-budgeted, 2 seeds) ---
step "residual_scale" checkpoints/residual_scale_results.json \
     python src/run_residual_scale.py --tokens 2500000 --seeds 0,1
sleep 30

# --- LONG training sweep goes LAST, deliberately. It is ~5h (8 arms x 2 seeds); every step above is
# minutes and produces a decomposition the report turns on. Ordering cheap-first means results land
# within the hour instead of after the sweep, and an early stop loses the least valuable item.
step "scale_control" checkpoints/scale_control_results.json \
     python src/run_scale_control.py --tokens 2500000 --seeds 0,1

say "QUEUE DONE"

