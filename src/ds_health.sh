#!/bin/zsh
# Mid-run health probe for DataSphere jobs. The terminal-state monitor only fires when a job ENDS;
# this catches a run that is alive but doing the wrong thing -- NaN loss, degenerate zero loss,
# collapsed throughput, or no progress at all -- while there is still time to act.
#
# Uses SHORT `attach` windows (30s) rather than a persistent stream: attach's poller reliably dies on
# the documented auth.py AssertionError over long runs, but a short window works every time.
setopt NULL_GLOB
export PATH="$HOME/.local/bin:$HOME/yandex-cloud/bin:$PATH"
P=bt12q57tmrs03pnt8drc
S=/private/tmp/claude-501/-Users-a2mogus-build-projs-barannikov-work-tlab-loop-transformer/8d0bbec0-a97a-4cfd-898c-ff91777e2e65/scratchpad
typeset -A lastbeat
while true; do
  ids=$(GRPC_DNS_RESOLVER=native timeout 90 datasphere --profile default project job list -p $P 2>/dev/null \
        | grep -iE "tlab-" | grep -E "EXECUTING" | awk '{print $1" "$2}')
  [ -z "$ids" ] && { echo "DS-HEALTH: no EXECUTING tlab jobs at $(date '+%H:%M')"; sleep 900; continue; }
  echo "$ids" | while read -r id name; do
    out=$(timeout 45 env GRPC_DNS_RESOLVER=native datasphere project job attach --id "$id" 2>/dev/null \
          | grep -E "step |EVAL|loss=" | tail -4)
    if [ -z "$out" ]; then
      echo "DS-HEALTH WARN $name: no step/EVAL lines in a 45s attach window -- stalled or between arms?"
      continue
    fi
    if echo "$out" | grep -qiE "loss=nan|loss=inf|loss=0\.0000|nan|inf"; then
      echo "DS-HEALTH ALERT $name: NaN/Inf/zero loss detected -- $(echo "$out" | tail -1)"
      continue
    fi
    now=$(date +%s); last=${lastbeat[$id]:-0}
    if [ $((now - last)) -ge 1800 ]; then
      echo "DS-HEALTH ok $name: $(echo "$out" | tail -1 | cut -c1-150)"
      lastbeat[$id]=$now
    fi
  done
  sleep 480
done
