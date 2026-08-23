"""Scale clock: three in-job arms at 2.5M tokens. Pre-registered read in RUNS.md 2026-08-23 14:00.

IN-JOB means all three arms train in ONE invocation with the same seed, the same data order and the
same eval cadence, so the comparison is paired rather than two independent draws. This project has
made the unpaired-comparison error twice (§6.0) and the eval-RNG interaction that breaks pairing
across differing `eval_every_tokens` is documented in §6.0b -- so every field except the
intervention is derived from one config object below, not re-typed per arm.

Arms:
  sc_ctrl        clock off                      -- the control the other two are read against
  sc_clock       clock on                       -- does the block use ||h|| if it is given it?
  sc_clock_sw90  clock on + supervision annealing at 0.90 -- the two levers are complementary in
                 principle: annealing manufactures depth demand, the clock removes a supply ceiling.
                 Neither is expected to be sufficient alone (§4.7's negative says demand may bind
                 entirely), which is exactly why the pair is worth one job.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch  # noqa: E402
from model import Config as ModelConfig  # noqa: E402
from train import TrainConfig  # noqa: E402
from chunked_runner import run_chunked  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAYER_APPS_PER_S = 1100.0 * 18 * 3
REF = "sd_dense_k5_s0"   # config donor, same as run_anneal_local.py -- derived, never re-typed


def main():
    ref = torch.load(ROOT / "checkpoints" / REF / "last.pt", map_location="cpu",
                     weights_only=False)
    base_m = dict(ref["model_cfg"])
    base_t = dict(ref["train_cfg"])
    tok = base_t["total_tokens"]
    cap = tok * 18 * 3 / LAYER_APPS_PER_S * 2.2

    arms = {
        "sc_ctrl":       (dict(scale_clock=False), dict()),
        "sc_clock":      (dict(scale_clock=True),  dict()),
        "sc_clock_sw90": (dict(scale_clock=True),  dict(supervise_k_final=1,
                                                        supervise_switch_frac=0.90)),
    }

    out = ROOT / "checkpoints" / "scale_clock_results.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    for name, (mover, tover) in arms.items():
        if name in res:
            print(f"{name}: already done, skipping", flush=True)
            continue
        mcfg = ModelConfig(**{**base_m, **mover})
        tcfg = TrainConfig(**{**base_t, **tover, "run_name": name})
        steps_per_eval = tcfg.eval_every_tokens // (tcfg.batch_size * tcfg.seq_len)
        assert steps_per_eval <= 200, f"{name}: eval too sparse for 240s chunks, run would not save"
        print(f"\n=== {name}  scale_clock={mcfg.scale_clock} "
              f"k={tcfg.supervise_k}->{tcfg.supervise_k_final} "
              f"cap {cap/60:.0f}min ===", flush=True)
        res[name] = run_chunked(name, mcfg, tcfg, cap, fresh=True)
        h = res[name]["history"]
        if h:
            c = h[-1]["val_curve"]
            b = min(c, key=c.get)
            print(f"  {name}: best r={b} CE={c[b]:.4f} gain={c['1']-c[b]:.4f}", flush=True)
        # PRIMARY read #2: did the model actually take the parameter?
        ck = ROOT / "checkpoints" / name / "last.pt"
        if ck.exists():
            w = torch.load(ck, map_location="cpu", weights_only=False)["model"].get("clock_w")
            if w is not None:
                res[name]["clock_w_norm"] = float(w.norm())
                res[name]["clock_w_absmax"] = float(w.abs().max())
                print(f"  clock_w: ||w||={w.norm():.5f}  max|w|={w.abs().max():.5f} "
                      f"(0 => the model declined the clock)", flush=True)
        out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
