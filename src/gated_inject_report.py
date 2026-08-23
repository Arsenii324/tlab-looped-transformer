"""Read the gated-injection screen in the pre-registered order (alpha first, CE last).

ONE CORRECTNESS POINT THIS FILE EXISTS FOR. §4.3's injection ratio is `||e|| / ||h_t||`, which is the
right quantity under ADDITIVE injection because the write is literally `e`. Under gating the write is
`delta * e` and the carry is `alpha * h`, so the comparable quantity is

    effective write ratio = || delta * e || / || alpha * h_t ||

Reporting the raw `||e||/||h||` for a gated arm and comparing it to §4.3's number would be comparing
two different things -- the same class of error as the `sigma_max`/rho and `pert_rel`/`||dh||`
mislabels this project has already made twice. Both are printed below, labelled.

Usage: python src/gated_inject_report.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig, LoopedTransformer  # noqa: E402
from plateau import plateau, plateau_mid, onset  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def ratios(name: str, loops=(1, 8, 64), batch=4, seq=256):
    ck = ROOT / "checkpoints" / name / "last.pt"
    if not ck.exists():
        return None
    d = torch.load(ck, map_location="cpu", weights_only=False)
    m = LoopedTransformer(ModelConfig(**d["model_cfg"]))
    m.load_state_dict(d["model"]); m.eval()
    val = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)          # same tokens for every arm
    ix = rng.integers(0, len(val) - seq - 1, size=batch)
    x = torch.from_numpy(np.stack([val[i:i + seq] for i in ix]).astype(np.int64))
    with torch.no_grad():
        e = m.embed(x)
        _, sn, st = m(x, n_loops=max(loops), return_all_loops=False, supervise_idx=set(),
                      return_states=True)
        gated = m.cfg.inject_mode == "gated"
        if gated:
            delta = F.softplus(m.inj_b)
            alpha = torch.exp(-delta * torch.exp(m.inj_a))
            wr = (delta * e).float().norm(dim=-1)
        else:
            delta = alpha = None
            wr = e.float().norm(dim=-1)
        out = {}
        for t in loops:
            h = st[t - 1].float()
            carry = (alpha * h) if gated else h
            out[t] = dict(raw=float((e.float().norm(dim=-1) / h.norm(dim=-1)).mean()),
                          effective=float((wr / carry.norm(dim=-1).clamp_min(1e-8)).mean()),
                          h_norm=float(sn[t - 1]))
    g = None
    if gated:
        g = dict(alpha_mean=float(alpha.mean()), alpha_min=float(alpha.min()),
                 alpha_max=float(alpha.max()), delta_mean=float(delta.mean()),
                 decaying=int((alpha < 0.99).sum()))
    return out, g


def main():
    res = json.loads((ROOT / "checkpoints" / "gated_inject_results.json").read_text())
    print("=" * 96)
    print("READ 1 -- DID THE MODEL TAKE THE PARAMETER? (alpha ~ 1 => declined; everything else moot)")
    print("=" * 96)
    for name in res:
        r = ratios(name)
        if r and r[1]:
            g = r[1]
            print(f"  {name:<14} alpha mean={g['alpha_mean']:.6f} min={g['alpha_min']:.6f} "
                  f"max={g['alpha_max']:.6f}  delta mean={g['delta_mean']:.4f}")
            print(f"  {'':<14} channels with alpha<0.99: {g['decaying']}/448   "
                  f"(init was alpha=0.9999939, delta=1.0)")
        elif r:
            print(f"  {name:<14} (additive control -- no gate)")

    print("\n" + "=" * 96)
    print("READ 2 -- THE MECHANISM: does the write stop being drowned?  (§4.3: 3.2e-3 -> 1.3e-4)")
    print("=" * 96)
    print(f"  {'arm':<14} {'ratio':<12} " + "  ".join(f"{'@'+str(t):>10}" for t in (1, 8, 64)))
    for name in res:
        r = ratios(name)
        if not r:
            continue
        for kind in ("raw", "effective"):
            lbl = "||e||/||h||" if kind == "raw" else "||de||/||ah||"
            print(f"  {name:<14} {lbl:<12} " + "  ".join(f"{r[0][t][kind]:10.3e}" for t in (1, 8, 64)))

    print("\n" + "=" * 96)
    print("READ 3 -- STATE NORM   |   READ 4 -- PLATEAU   |   READ 5 -- CE (reported, not deciding)")
    print("=" * 96)
    print(f"  {'arm':<14} {'|h|@1':>9} {'|h|@8':>9} {'|h|@64':>10} {'CE@1':>8} {'CE_best':>8} "
          f"{'@d':>3} {'plateau':>12} {'mid':>5} {'onset':>5}")
    rows = {}
    for name, v in res.items():
        h = v.get("history") or []
        if not h:
            print(f"  {name:<14} (no evals)"); continue
        c = {int(a): b for a, b in h[-1]["val_curve"].items()}
        r = ratios(name)
        n = r[0] if r else {}
        rows[name] = c
        print(f"  {name:<14} {n.get(1,{}).get('h_norm',float('nan')):9.1f} "
              f"{n.get(8,{}).get('h_norm',float('nan')):9.1f} "
              f"{n.get(64,{}).get('h_norm',float('nan')):10.1f} "
              f"{c[1]:8.4f} {min(c.values()):8.4f} {min(c,key=c.get):3d} "
              f"{str(plateau(c)):>12} {plateau_mid(c):5.1f} {onset(c):5d}")

    if "gi_additive" in rows and "gi_gated" in rows:
        a, g = rows["gi_additive"], rows["gi_gated"]
        dB, d1 = min(g.values()) - min(a.values()), g[1] - a[1]
        print(f"\n  gated vs in-job additive:  dCE_best {dB:+.4f}   dCE@1 {d1:+.4f}   "
              f"dgain {d1-dB:+.4f}")
        print(f"  MPS replicate floor is 0.031-0.068 -> dCE_best is "
              f"{'OUTSIDE' if abs(dB) > 0.068 else 'INSIDE'} it "
              f"(896 params; this was pre-registered as unresolvable)")

    # the free replicate: gi_additive vs the scale-clock run's sc_ctrl, config-identical
    sc = ROOT / "checkpoints" / "scale_clock_results.json"
    if sc.exists() and "gi_additive" in rows:
        s = json.loads(sc.read_text()).get("sc_ctrl", {}).get("history")
        if s:
            c = {int(a): b for a, b in s[-1]["val_curve"].items()}
            a = rows["gi_additive"]
            print(f"\n  FREE REPLICATE (config-identical, separate invocation):")
            print(f"    sc_ctrl     CE_best {min(c.values()):.4f}  CE@1 {c[1]:.4f}")
            print(f"    gi_additive CE_best {min(a.values()):.4f}  CE@1 {a[1]:.4f}")
            print(f"    |diff| CE_best {abs(min(a.values())-min(c.values())):.4f}  "
                  f"CE@1 {abs(a[1]-c[1]):.4f}   <- an MPS same-config floor for THIS arm (§4.15 gap)")


if __name__ == "__main__":
    main()
