#!/bin/zsh
# Poll every kaggle kernel named in KAGGLE_SLUGS.txt; on any status change from RUNNING,
# append a loud line to kaggle_monitor.log and auto-download the output.
setopt NULL_GLOB
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")"
typeset -A last
while true; do
  grep -v '^#' KAGGLE_SLUGS.txt | while read -r slug; do
    [[ -z "$slug" ]] && continue
    st=$(kaggle kernels status "$slug" 2>&1 | head -1 | sed 's/.*status //;s/"//g')
    if [[ "${last[$slug]}" != "$st" ]]; then
      echo "$(date '+%m-%d %H:%M') $slug -> $st" >> kaggle_monitor.log
      last[$slug]=$st
      if [[ "$st" != *RUNNING* && "$st" != *QUEUED* ]]; then
        out="kaggle_out/${slug##*/}"; mkdir -p "$out"
        kaggle kernels output "$slug" -p "$out" >> kaggle_monitor.log 2>&1
        echo "$(date '+%m-%d %H:%M') DOWNLOADED -> $out" >> kaggle_monitor.log
      fi
    fi
  done
  sleep 180
done
