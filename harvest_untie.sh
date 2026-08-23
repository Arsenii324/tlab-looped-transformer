#!/usr/bin/env bash
# Harvest tlab-untie-s0 and resolve GATES A/B/C, which were registered in RUNS.md BEFORE the data
# existed. Written before the job landed so the gates cannot be reinterpreted after seeing numbers.
#   ./harvest_untie.sh [job_id]
set -uo pipefail
ID="${1:-bt1anqsjuulfo4061jrd}"
OUT=/tmp/ds_untie_out
export GRPC_DNS_RESOLVER=native
export PATH="$HOME/.local/bin:$HOME/yandex-cloud/bin:$PATH"

echo "== downloading $ID =="
datasphere project job download-files --id "$ID" --with-logs --output-dir "$OUT" 2>&1 | tail -2
ls -la "$OUT" 2>/dev/null | tail -6

echo; echo "== is a DS ERROR cosmetic? (OPS trap) =="
grep -c "Error while processing file" "$OUT/log.txt" 2>/dev/null || echo "0 (no log.txt)"
tail -4 "$OUT/stdout.log" 2>/dev/null

echo; echo "== GATE B + C: in-job deltas, with eval-count parity checked first =="
python3 - "$OUT" <<'PY'
import json,sys,os
sys.path.insert(0,"src")
from plateau import plateau, plateau_mid
from chance_guard import assert_not_chance
d=json.load(open(os.path.join(sys.argv[1],"results.json")))
cur={k:{int(a):b for a,b in v["history"][-1]["val_curve"].items()} for k,v in d.items()}
nev={k:len(v["history"]) for k,v in d.items()}; stp={k:v["history"][-1]["step"] for k,v in d.items()}
for k,c in cur.items():
    assert_not_chance(c, d[k]["model_cfg"]["vocab_size"], label=k)
    print(f"  {k:16s} step={stp[k]} nev={nev[k]} params={d[k]['params']:,} "
          f"CE@1={c[1]:.4f} best={min(c.values()):.4f}@r{min(c,key=c.get)} band={plateau(c)[:2]} mid={plateau_mid(c):.1f}")
ok=len({(stp[k],nev[k]) for k in cur})==1
print(f"\n  eval/step parity across arms: {'OK' if ok else '*** MISMATCH -- DO NOT DIFFERENCE ***'}")
def dd(a,b):
    ca,cb=cur[a],cur[b]; ba,bb=min(ca.values()),min(cb.values())
    d1=ca[1]-cb[1]; db=ba-bb
    return d1,db,(d1/db*100 if db else float('nan'))
if {"ut_b4_s0","ut_b4_gate_s0","ut_ctrl_s0"} <= set(cur):
    for a,b,lab in [("ut_b4_s0","ut_ctrl_s0","untying alone vs tied control"),
                    ("ut_b4_gate_s0","ut_ctrl_s0","untying+gate vs tied control"),
                    ("ut_b4_gate_s0","ut_b4_s0","GATE B: the GATE's contribution AT HIGH RANK")]:
        d1,db,sh=dd(a,b)
        print(f"  {lab:44s} dCE@1={d1:+.4f}  dCE_best={db:+.4f}  share@r1={sh:5.0f}%")
    print("\n  GATE B reference -- the same gate at rank ~1.6 (dg_norm, sec4.23): -0.0012 / +0.0023")
    print("  floor: 0.0150 (CUDA dense). GATE C predicts 67-101% at r=1 if the gain is capacity.")
PY

echo; echo "== GATE A: did untying actually raise the depth-key rank? (must exceed ~4) =="
for arm in ut_ctrl_s0 ut_b4_gate_s0; do
  if [ -f "$OUT/${arm}_last.pt" ]; then
    echo "  --- $arm ---"
    python3 src/depth_key_rank.py "$OUT/${arm}_last.pt" --loops 32 2>&1 | grep "layer 0"
  fi
done
echo
echo "  GATE A reference: tied control measures ~1.6/32 (sec4.7e). sec4.28 predicts ~5.7 at 4 buckets."
echo "  If the b4 arm comes back near 1.6, the buckets did not take and NOTHING BELOW IS DECIDED."
echo "  NOTE: this reads token ids the job did not train on (DS returns no tokenizer). That is"
echo "  common-mode to both arms and rank is structural -- see the RUNS.md amendment of 21:52."
