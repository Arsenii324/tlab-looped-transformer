"""Single source of truth for the headline numbers, and a checker that report.md agrees with it.

Two jobs, both about the same endgame risk. The headline is currently the 46.0M-token checkpoint;
two 90M runs land at ~06:30 and ~09:30. Every derived figure -- bpb, loop gain, §4.7's baselines --
is keyed to whichever checkpoint is the headline. Swapping by find-and-replace across a 1400-line
report at hour 8 is exactly how a stale number ships.

  `python src/headline.py show`            print the authoritative numbers from the eval JSON
  `python src/headline.py check`           verify every one of them appears in report.md
  `python src/headline.py set <run_name>`  repoint the headline at a different checkpoint

`check` is the useful one: it does not rewrite anything, it tells you which numbers in report.md no
longer match the artifact they came from. Run it before shipping.
"""
from __future__ import annotations
import json, math, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PTR = ROOT / "checkpoints" / "HEADLINE.json"
BYTES_PER_TOKEN = 3.3358


def load(run=None):
    run = run or (json.loads(PTR.read_text())["run"] if PTR.exists() else "full_no_state_renorm_kaggle")
    ev = ROOT / "checkpoints" / run / f"eval_{run}.json"
    d = json.loads(ev.read_text())
    ce = d["val_ce"]; b = str(d["best_loop"])
    return dict(run=run, source=str(ev.relative_to(ROOT)), tokens=d["tokens"], best_loop=int(b),
                best_ce=ce[b], ppl=math.exp(ce[b]), bpb=d["val_bits_per_byte"][b],
                ce_at_1=ce["1"], loop_gain=ce["1"] - ce[b],
                ce_at_max=ce[max(ce, key=lambda k: int(k))],
                max_loop=max(int(k) for k in ce))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "set":
        PTR.write_text(json.dumps({"run": sys.argv[2]}, indent=2))
        print(f"headline repointed to {sys.argv[2]}")
        cmd = "show"
    h = load()
    if cmd == "show":
        print(f"HEADLINE = {h['run']}   (source: {h['source']})")
        for k in ("tokens", "best_loop", "best_ce", "ppl", "bpb", "ce_at_1", "loop_gain",
                  "max_loop", "ce_at_max"):
            v = h[k]
            print(f"  {k:12} {v:,.4f}" if isinstance(v, float) else f"  {k:12} {v:,}")
    elif cmd == "check":
        rep = (ROOT / "report.md").read_text()
        checks = [(f"{h['best_ce']:.4f}", "best CE"), (f"{h['bpb']:.4f}", "bits/byte"),
                  (f"{h['loop_gain']:.4f}", "loop gain"), (f"{h['ce_at_1']:.4f}", "CE at loop 1"),
                  (f"{h['ppl']:.2f}", "perplexity"), (str(h["best_loop"]), "best loop")]
        bad = [(v, lab) for v, lab in checks if v not in rep]
        print(f"headline = {h['run']} ({h['tokens']:,} tokens)")
        for v, lab in checks:
            print(f"  {'OK  ' if (v, lab) not in bad else 'MISS'} {lab:14} {v}")
        print(f"\n{len(bad)} headline number(s) missing from report.md"
              + (" -- report is stale relative to the artifact" if bad else " -- consistent"))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
