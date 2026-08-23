"""Does a DataSphere job that is CANCELLED MID-COMPUTE yield anything recoverable?

Operational question with a real consequence: a 10h run is safe to launch past the nominal cutoff
ONLY if stopping it early still yields usable partial results. Two channels could survive a cancel --
stdout (via `download-files --with-logs`) and declared `outputs:` files. This job exercises both:
it prints a heartbeat AND rewrites a results file every few seconds, so after cancelling we can see
exactly which channel survives and up to what point.

Known from earlier jobs: a job cancelled during SETUP yields only system.log; a job that finishes
compute then fails at output collection yields stdout but not files. The gap is mid-compute cancel.
"""
import json, os, pathlib, sys, time

if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    out = here / "results"; out.mkdir(exist_ok=True)
    rows = []
    for i in range(200):
        rows.append({"tick": i, "t": round(time.time(), 2)})
        print(f"HEARTBEAT tick={i} elapsed={i*3}s", flush=True)
        (out / "partial.json").write_text(json.dumps(rows))
        time.sleep(3)
    print("FINISHED NORMALLY", flush=True)
