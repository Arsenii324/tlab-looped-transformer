"""Does every number the submission quotes actually appear in the report it cites?

The gap this closes, named by an external reviewer at 20:55: `headline.py` checks headline numbers
against artifacts and `check_caveats.py` checks that a deflated claim carries its caveat -- but
NOTHING checked that a figure quoted in `submission/NEGATIVE_RESULTS.md` equals the figure in
`report.md`. Seven documents cite into a 6,700-line file, both edited concurrently, and tonight three
of the worst defects lived in exactly that consistency surface.

A number in `submission/` that does not appear anywhere in `report.md` is either (a) a transcription
slip, (b) a number that was updated on one surface and not the other, or (c) legitimately
submission-only (a count, a percentage computed for the summary). This flags all three; (c) is
whitelisted by KNOWN so the signal stays usable.

    python src/check_crossref.py           # report
    python src/check_crossref.py --strict  # exit 1 if anything is unexplained
"""
from __future__ import annotations
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT = (ROOT / "report.md").read_text()

# Numbers that are legitimately submission-only: derived summaries, counts, and arithmetic the
# report states in another form. Each needs a reason -- an unexplained entry here defeats the check.
KNOWN = {
    "9,064,608": "parameter count, stated with commas in submission and without in some report lines",
    "89,999,360": "exact token count; report rounds to 90.0M",
    "0.2398": "mean of XSA's two seeds, computed for the summary; report gives both seeds",
    "1,806,336": "MoE arithmetic done in SCALE.md only",
    "0.0220": "diversity residual, computed in-line in RESULTS from two report figures",
    "1.4512": "the tokenizer gate's own printed threshold |CE_local-chance|/3 with CE_local=3.9642; "
              "verified live 20:53 -- the report quotes the gate's PASS but not this intermediate",
    "6,600": "a description of report.md's own line count, not a measurement",
}

def numbers(text: str) -> set[str]:
    # 3+ significant figures only: 2-digit numbers are section refs, years, grid points.
    out = set()
    for m in re.finditer(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+\.\d{3,4})(?![\w])", text):
        out.add(m.group(1))
    return out

def main() -> int:
    strict = "--strict" in sys.argv
    bad = 0
    for f in sorted((ROOT / "submission").glob("*.md")):
        if f.name == "EXPERIMENTS.md":
            continue  # generated from the artifacts themselves; the report is not its source
        miss = sorted(n for n in numbers(f.read_text())
                      if n not in REPORT and n not in KNOWN)
        if miss:
            bad += len(miss)
            print(f"  [MISS] {f.name}: {len(miss)} figure(s) not found in report.md")
            for n in miss:
                print(f"           {n}")
        else:
            print(f"  [OK  ] {f.name}")
    print(f"\n  {bad} figure(s) in submission/ do not appear in report.md.")
    return 1 if (strict and bad) else 0

if __name__ == "__main__":
    raise SystemExit(main())
