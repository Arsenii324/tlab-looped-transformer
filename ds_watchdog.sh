#!/bin/zsh
# Re-attach DataSphere jobs whose LOCAL attach has died.
#
# The datasphere CLI's poller dies on `assert current_iam_token` (auth refresh) while the REMOTE job
# keeps running -- twice tonight. The remote job is unharmed, but the attach carries the automatic
# output download that fires on completion, so a dead attach means results are silently lost at the
# end of a multi-hour run. Alerting on it is not enough; it has to be repaired.
#
# Staleness, not the error string, is the trigger: any attach log with no write for STALE_MIN while
# its job is still EXECUTING gets re-attached. That also catches deaths with no error line at all.
setopt NULL_GLOB
export PATH="$HOME/.local/bin:$HOME/yandex-cloud/bin:$PATH"
cd "$(dirname "$0")"
S=/private/tmp/claude-501/-Users-a2mogus-build-projs-barannikov-work-tlab-loop-transformer/8d0bbec0-a97a-4cfd-898c-ff91777e2e65/scratchpad
STALE_MIN=4
while true; do
  for f in "$S"/ds_*.log; do
    b=$(basename "$f" .log)
    case "$b" in *_reattach) continue;; esac          # don't chase our own re-attach logs
    jid=$(grep -oE 'created job `[a-z0-9]+`' "$f" 2>/dev/null | head -1 | tr -d '`' | awk '{print $3}')
    [[ -z "$jid" ]] && continue
    age=$(( ( $(date +%s) - $(stat -f %m "$f") ) / 60 ))
    [[ $age -lt $STALE_MIN ]] && continue
    # only act while the job is genuinely still running
    st=$(GRPC_DNS_RESOLVER=native timeout 60 datasphere --profile default project job get --id "$jid" 2>/dev/null \
         | grep -oE "EXECUTING|SUCCESS|ERROR|CANCELLED" | head -1)
    [[ "$st" != "EXECUTING" ]] && continue
    ra="$S/${b}_reattach.log"
    ra_age=999
    [[ -f "$ra" ]] && ra_age=$(( ( $(date +%s) - $(stat -f %m "$ra") ) / 60 ))
    [[ $ra_age -lt $STALE_MIN ]] && continue          # a live re-attach already covers it
    echo "$(date '+%m-%d %H:%M') WATCHDOG re-attaching $b ($jid), attach stale ${age}m" >> ds_watchdog.log
    nohup env GRPC_DNS_RESOLVER=native datasphere --profile default project job attach --id "$jid" >> "$ra" 2>&1 &
  done
  sleep 120
done
