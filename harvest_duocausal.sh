#!/usr/bin/env bash
# One-command harvest for tlab-duocausal-s0/-s1 and tlab-recmethod-s2.
# Written 18:46, BEFORE the data landed, so the procedure is not improvised at harvest time.
#   ./harvest_duocausal.sh
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
declare -A JOBS=(
  [bt1qvi35v7gsejmvn1it]=dc_s0
  [bt1lkbri6cqj6q9fssoa]=dc_s1
  [bt1s4mag4kdvsvts536m]=recmethod_s2
)
OUT=/tmp/ds_harvest; mkdir -p $OUT
for id in "${!JOBS[@]}"; do
  tag=${JOBS[$id]}
  st=$(GRPC_DNS_RESOLVER=native timeout 60 datasphere project job get --id "$id" 2>/dev/null | tail -1 \
        | grep -oE "EXECUTING|SUCCESS|ERROR|CANCELLED")
  echo "== $tag ($id): ${st:-unknown}"
  [ "$st" = "EXECUTING" ] && { echo "   still running, skipping"; continue; }
  d=$OUT/$tag; mkdir -p "$d"
  GRPC_DNS_RESOLVER=native timeout 600 datasphere project job download-files --id "$id" --output-dir "$d" 2>&1 | tail -2
  # sec6.0 row 34: a declared-but-missing output marks the job ERROR. Check the REAL discriminator.
  ld=$(ls -dt /private/var/folders/*/T/datasphere/job_* 2>/dev/null | head -40 | xargs grep -l "$id" 2>/dev/null | head -1)
  echo "   output-upload errors: $(grep -c 'Error while processing file' "$ld/log.txt" 2>/dev/null || echo '?')"
  ls -la "$d" | tail -6
  # arrange each returned <arm>_last.pt as checkpoints/<arm>/last.pt for state_dynamics.py
  for pt in "$d"/*_last.pt; do
    [ -e "$pt" ] || continue
    arm=$(basename "$pt" _last.pt); mkdir -p "checkpoints/$arm"; cp "$pt" "checkpoints/$arm/last.pt"
    echo "   -> checkpoints/$arm/last.pt"
  done
done
echo; echo "===== reads (a),(c),(d) — the pre-registered table"
python3 src/harvest_duocausal.py $OUT/dc_s0/results.json $OUT/dc_s1/results.json 2>/dev/null
echo; echo "===== read (b) — cos(du_t,du_{t-1}) vs sec4.3's 0.9999; needs the checkpoints"
for a in dc_control_s0 dc_w2_s0 dc_w3_s0; do
  [ -f "checkpoints/$a/last.pt" ] && python3 src/state_dynamics.py "checkpoints/$a" --max-loops 32 2>/dev/null | grep -iE "incr_cos|loop|^ *[0-9]+" | head -8
done
