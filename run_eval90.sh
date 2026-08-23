#!/bin/zsh
# Protocol-matched local eval of the two 90M-token Kaggle runs.
#
# WHY A SEPARATE SCRIPT: run_queue.sh is executing, and zsh reads a script incrementally by byte
# offset -- appending to a running script can make the shell execute a partial line. So this waits
# for the local GPU to go idle and then runs, instead of being bolted onto the live queue.
#
# WHY THESE EVALS: the 46.0M headline (CE 4.0071 / ppl 54.99) was produced by src/eval.py locally.
# The 90M numbers so far come from the Kaggle kernel's own in-run eval -- different code, different
# val batches. The headline must not be swapped on a cross-protocol comparison, so both 90M
# checkpoints are re-scored under exactly the protocol that produced the current headline.
setopt NULL_GLOB
cd "$(dirname "$0")"
log() { echo "$(date '+%m-%d %H:%M') $*" >> eval90.log }
log "waiting for local GPU to go idle..."
while pgrep -f "run_scale_control|run_residual_scale|run_supervision|run_sandwich" >/dev/null; do sleep 120; done
log "GPU idle -- starting protocol-matched evals"
for d in full_control90_kaggle full_normpen_kaggle; do
  out="checkpoints/$d/eval_${d}.json"
  if [[ -f "$out" ]]; then log "skip $d (already evaluated)"; continue; fi
  log "eval $d ..."
  python src/eval.py "checkpoints/$d" --max-loops 64 >> eval90.log 2>&1 \
    && log "eval $d DONE" || log "eval $d FAILED (see tail above)"
  sleep 20
done
log "all done"
