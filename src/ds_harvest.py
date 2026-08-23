"""Pull finished arms out of a live DataSphere attach log and report them on the plateau statistic.

DS jobs stream `=== ARM <name> ===` then a final `EVAL step <n>: r1=... r2=...` per arm. Results are
only written to disk when the whole job ends, so mid-run the attach log IS the data. This reads
whatever has finished so far, which is what makes analysing a multi-hour job midway possible instead
of waiting for the harvest.

Reports plateau, not argmin: on these curves argmin is decided far below the noise floor (report
§4.15). Plateau bands are grid-dependent, so `--grid` restricts to a comma-separated common grid
before comparing arms from different jobs.
"""
from __future__ import annotations
import re, sys, argparse, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from plateau import plateau, plateau_mid

ARM = re.compile(r"=== ARM (\S+) ===")
EVAL = re.compile(r"EVAL step (\d+): (.+)$")
NOISE = re.compile(r"fork_posix|ev_poll_posix|FD from fork")


def parse(path, carry=None):
    """-> [(arm_name, step, {loop: ce})], last eval per arm only.

    `carry`: arm name to attribute EVALs to before this file's first `=== ARM ===` line. A job whose
    attach died mid-arm resumes logging EVALs with no header, and dropping them silently would make
    a finished arm look unfinished -- which is exactly the failure this harvester exists to avoid."""
    arms, cur, out = [], carry, {}
    for line in open(path, errors="replace"):
        if NOISE.search(line):
            continue
        m = ARM.search(line)
        if m:
            cur = m.group(1)
            continue
        m = EVAL.search(line)
        if m and cur:
            c = {}
            for tok in m.group(2).split():
                mm = re.fullmatch(r"r(\d+)=([0-9.]+)", tok)
                if mm:
                    c[int(mm.group(1))] = float(mm.group(2))
            if c:
                out[cur] = (int(m.group(1)), c)
    return [(k, v[0], v[1]) for k, v in out.items()], cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="+", help="one or more logs for the SAME job, in time order -- a job whose attach died mid-run has its ARM headers in the original log and later EVALs in the re-attach log, so both must be read together")
    ap.add_argument("--grid", default=None, help="comma-separated loop counts to restrict to")
    ap.add_argument("--tol", type=float, default=0.01)
    a = ap.parse_args()
    grid = {int(x) for x in a.grid.split(",")} if a.grid else None
    rows, carry = {}, None
    for lg in a.log:                      # merge in order; later files override earlier evals
        parsed, carry = parse(lg, carry)  # carry the open arm across an attach death
        for name, step, c in parsed:
            rows[name] = (step, c)
    rows = [(k, v[0], v[1]) for k, v in rows.items()]
    if not rows:
        print("no completed arms yet"); return
    print(f"{'arm':<18} {'step':>6} {'best CE':>9} {'CE@1':>9} {'gain':>8} {'argmin':>6} "
          f"{'plateau':>11} {'mid':>6}")
    for name, step, c in rows:
        r = {t: v for t, v in c.items() if grid is None or t in grid}
        if len(r) < 3:
            print(f"{name:<18} {step:>6}  (too few points on the requested grid)"); continue
        b = min(r, key=r.get)
        lo, hi, ct = plateau(r, a.tol)
        one = r[min(r)]
        print(f"{name:<18} {step:>6} {r[b]:>9.4f} {one:>9.4f} {one-r[b]:>8.4f} {b:>6} "
              f"{'['+str(lo)+','+str(hi)+']':>11} {plateau_mid(r, a.tol):>6.1f}"
              + ("" if ct else "  NON-CONTIG"))


if __name__ == "__main__":
    main()
