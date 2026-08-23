#!/bin/zsh
# Scan every job log for GENUINE failures and record each one ONCE.
#
# Replaces a Monitor-tool task that re-fired on already-handled terminal states every cycle. That is
# alert fatigue: a repeated stale alert can bury a new one. Here each distinct failure signature per
# file is reported once, to failures.log.
#
# The datasphere CLI's own poller dies on `assert current_iam_token` while the REMOTE job keeps
# running -- that fired repeatedly tonight and every instance was benign, so it is excluded. What is
# NOT excluded, because each one cost something real today: CUDA/MPS OOM (lost the mu=56 pair),
# KeyError (crashed trainL-s1's summary after 6.5h of training), degenerate output, NaN loss.
setopt NULL_GLOB
cd "$(dirname "$0")"
S=/private/tmp/claude-501/-Users-a2mogus-build-projs-barannikov-work-tlab-loop-transformer/8d0bbec0-a97a-4cfd-898c-ff91777e2e65/scratchpad
typeset -A seen
while true; do
  for f in "$S"/*.log; do
    [[ -f "$f" ]] || continue
    b=$(basename "$f")
    sig=$(grep -vE "\.local/bin/datasphere|current_iam_token|site-packages/datasphere" "$f" 2>/dev/null \
          | grep -oE "CUDA out of memory|MPS backend out of memory|DEGENERATE|loss=nan|loss=inf|Killed|KeyError|AttributeError|RuntimeError" \
          | sort | uniq -c | tr '\n' ' ')
    [[ -z "${sig// /}" ]] && continue
    if [[ "${seen[$b]:-}" != "$sig" ]]; then
      echo "$(date '+%m-%d %H:%M') $b :: $sig" >> failures.log
      seen[$b]="$sig"
    fi
  done
  sleep 300
done
