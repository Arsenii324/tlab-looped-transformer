"""Is §4.3's 'near-straight ray' an artefact of sampling the state once per loop?

THE RISK, and it is a documented one from a prior project rather than a hypothetical: every
instrument there read ONE hook inside a multi-block loop, and hooking all blocks revealed a period-4
cycle across them that single-block reading had been blind to *by construction*. The licensed
statement was never "no loop anywhere" -- only "no loop in the iteration-to-iteration map at a fixed
hook".

This model has **layers_per_loop = 3** and `LoopedTransformer.forward` appends the state exactly once
per loop iteration, after all three layers. So §4.3's cos(du_t, du_{t-1}) -> 0.9999 and its monotone
radial drift are measured at the loop boundary and CANNOT see a period-3 intra-loop cycle.

This captures the state after EACH of the three layers via forward hooks -- no change to model.py,
no training -- and asks whether the fine-grained path is still a ray, or whether the loop-boundary
samples are stroboscopic snapshots of something that circulates within each iteration.
"""
from __future__ import annotations
import argparse, pathlib, sys
import numpy as np, torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval import load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--loops", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()

    d = pathlib.Path(a.ckpt)
    model = load_checkpoint(d / "last.pt" if d.is_dir() else d, "cpu")[0]
    model.eval()
    layers = list(model.block.layers)               # the 3 reused DecoderLayers
    print(f"layers per loop: {len(layers)}")

    caught = []
    hs = [l.register_forward_hook(lambda m, i, o, _l=n: caught.append((_l, o.detach())))
          for n, l in enumerate(layers)]

    val = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    x = torch.from_numpy(np.stack([val[i:i + a.seq].astype(np.int64)
                                   for i in rng.integers(0, len(val) - a.seq - 2, size=a.batch)]))
    with torch.no_grad():
        model(x, n_loops=a.loops, supervise_idx=set())
    for h in hs:
        h.remove()

    L = len(layers)
    assert len(caught) == a.loops * L, f"expected {a.loops*L} captures, got {len(caught)}"
    u = [s.float() / s.float().norm(dim=-1, keepdim=True).clamp_min(1e-12) for _, s in caught]

    def cos_consec(seq):
        cs = []
        for t in range(2, len(seq)):
            d1, d0 = seq[t] - seq[t - 1], seq[t - 1] - seq[t - 2]
            c = (d1 * d0).sum(-1) / (d1.norm(dim=-1) * d0.norm(dim=-1)).clamp_min(1e-12)
            cs.append(c.mean().item())
        return cs

    fine = cos_consec(u)                                    # every layer output, 3 per loop
    coarse = cos_consec(u[L - 1::L])                        # loop boundary only -- what §4.3 used
    print(f"\ncos(du_t, du_t-1)  LOOP-BOUNDARY sampling (what §4.3 reports):")
    print(f"  first={coarse[0]:+.4f}  last={coarse[-1]:+.4f}  mean(last 5)={np.mean(coarse[-5:]):+.4f}")
    print(f"cos(du_t, du_t-1)  PER-LAYER sampling (3x finer):")
    print(f"  first={fine[0]:+.4f}  last={fine[-1]:+.4f}  mean(last 15)={np.mean(fine[-15:]):+.4f}")

    # a period-3 cycle would show as the three within-loop phases occupying distinct radii/directions
    print(f"\nper-layer structure within a loop (mean over the last 8 loops):")
    for ph in range(L):
        idx = [t for t in range(len(u)) if t % L == ph][-8:]
        nrm = np.mean([caught[t][1].float().norm(dim=-1).mean().item() for t in idx])
        print(f"  phase {ph} (after layer {ph}): mean ||h|| = {nrm:9.2f}")
    # cosine between the SAME phase one loop apart vs ADJACENT phases -- a cycle makes these differ
    same, adj = [], []
    for t in range(L, len(u)):
        c = (u[t] * u[t - L]).sum(-1).mean().item(); same.append(c)
    for t in range(1, len(u)):
        c = (u[t] * u[t - 1]).sum(-1).mean().item(); adj.append(c)
    print(f"\n  cos(u_t, u_t-3)  same phase, one loop apart : {np.mean(same[-15:]):.6f}")
    print(f"  cos(u_t, u_t-1)  adjacent phases            : {np.mean(adj[-15:]):.6f}")
    print("  -> if these are both ~1.0 the fine path is a ray too; a period-3 cycle would make")
    print("     the adjacent-phase cosine markedly SMALLER than the same-phase one.")



def _persist_stdout(name, text):
    # PERSIST (traceability audit 2026-08-23): this printed its numbers and saved nothing, so
    # every claim it supports was reproducible but not traceable -- verifying one meant
    # re-running it, which only works while its inputs survive.
    import pathlib as _pl
    _dst = _pl.Path(__file__).resolve().parents[1] / "checkpoints" / f"{name}_report.txt"
    _dst.write_text(text)
    print(f"wrote {_dst}")

if __name__ == "__main__":
    import io as _io, contextlib as _cl
    _buf = _io.StringIO()
    with _cl.redirect_stdout(_buf):
        main()
    _out = _buf.getvalue()
    print(_out, end="")
    _persist_stdout("intraloop_states", _out)

