"""Does every reader-facing file that states a DEFLATED claim also carry its caveat?

Three of 2026-08-23's defects were the same shape: a deflation living in `report.md` and not in the
document a reader actually opens (sec8 carried neither LoRA caveat; `NEGATIVE_RESULTS.md` carried none
of three; `SCALE.md` sec1 defended a claim whose CE half was withdrawn). `FAILURES.md` said plainly
that "nothing in this project enforces it". This is the thing that enforces it.

Each rule pairs a NUMBER that only appears when the claim is being made with a TOKEN that must appear
in the same file. A file matching the number but not the token is flagged. It cannot check that the
caveat sits *next to* the number -- only that the file carrying the claim also carries the caveat --
which is the failure that actually occurred three times.

    python src/check_caveats.py          # report
    python src/check_caveats.py --strict # exit 1 if anything is missing (for a pre-ship gate)
"""
from __future__ import annotations
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "report.md"] + sorted((ROOT / "submission").glob("*.md"))

RULES = [
    dict(token="[WITHDRAWN-ANNEAL-CE]",
         claim=r"−0\.0811|-0\.0811|−0\.0710|-0\.0710",
         why="annealing's CE advantage was WITHDRAWN at n=4 (seed 2 gives +0.0482; t-interval covers "
             "zero). The band result at 4/4 seeds survives."),
    dict(token="[POSTHOC-LORA-RANK]",
         claim=r"−0\.0857|-0\.0857|−0\.0936|-0\.0936|−0\.094\b|-0\.094\b",
         why="the rank>=4 restriction is POST HOC; over all arms the interval covers zero; there is "
             "no dose-response above the threshold."),
    dict(token="[CAPACITY-NOT-DIVERSITY]",
         claim=r"−0\.1251|-0\.1251|−0\.1011|-0\.1011",
         why="a branch pinned to one index (identical params, zero diversity) recovers 82% in-job, "
             "and ~90% of the gain sits at r=1 where cycling is inert. It is a CAPACITY result."),
    dict(token="[XSA-N1]",
         claim=r"−0\.2162|-0\.2162|−0\.216\b|-0\.216\b",
         why="XSA is ONE SEED at 2.5M tokens; 84% of the effect is at r=1."),
    dict(token="[RANK-PROJECTION]",
         claim=r"31\.8[03]|11\.7×|11\.7x",
         why="most of the untied-vs-tied key-rank gap is per-layer PROJECTION randomness; at the "
             "representation level it is 3.1x (4.36 vs 1.40), and both streams are collinear."),
]

def main() -> int:
    strict = "--strict" in sys.argv
    missing = 0
    for r in RULES:
        pat, tok = re.compile(r["claim"]), r["token"]
        carriers = [p for p in TARGETS if p.exists() and pat.search(p.read_text())]
        bad = [p for p in carriers if tok not in p.read_text()]
        status = "OK  " if not bad else "MISS"
        print(f"  [{status}] {tok:26s} stated in {len(carriers)} file(s)"
              f"{'' if not bad else '  -> MISSING in: ' + ', '.join(p.name for p in bad)}")
        if bad:
            missing += len(bad)
            print(f"         {r['why']}")
    print(f"\n  {missing} file(s) state a deflated claim without carrying its caveat token.")
    return 1 if (strict and missing) else 0

if __name__ == "__main__":
    raise SystemExit(main())
