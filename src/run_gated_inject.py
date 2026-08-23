"""Gated injection vs additive injection: two in-job arms at 2.5M tokens.

THE GAP THIS FILLS. §4.1 swept the normalisation axis as {hard inter-loop RMSNorm, nothing} and the
second is worth -0.744 nats, the largest effect in the project. Both reference implementations
(Parcae; *Looped Transformers Done Right*) choose NEITHER -- they use a diagonal state-space write
with a learned per-channel carry decay, which bounds the state without projecting it onto a sphere.
That cell has never been run here. See `model.py::_inject` for the form and the parameter accounting.

IN-JOB, and both arms train in ONE invocation. §6.0b records that `evaluate()` draws from the
TRAINING rng, so two runs differing in eval cadence are not paired even at the same seed; and this
project has made the unpaired-comparison error twice (§6.0). Every field except `inject_mode` is
derived from the same reference checkpoint rather than re-typed -- hand-matching a config cost 727s
and produced nothing on 2026-08-23 (§6.0 row 32).

PRE-REGISTERED READ, in order. Written before the run, and the FIRST item is not the loss:

  1. Did the model take the parameter? Report mean/min alpha and mean delta. The arm is initialised
     AT the additive model (alpha = 0.9999939, delta = 1.0), so alpha staying ~1 means the model
     declined a strictly-larger hypothesis class and every other reading is moot. This is the same
     first-question discipline as `||clock_w||` in run_scale_clock.py.
  2. ||e||/||h|| at loops 1/8/64 vs the control. This is the mechanism the form is supposed to fix:
     §4.3 measures 3.2e-3 -> 1.3e-4 under plain addition, and calls the re-injected input "drowned".
     If alpha < 1 the state is bounded and the ratio should stop collapsing. **This is the primary
     result whether or not CE moves.**
  3. State norm trajectory: does ||h|| stop growing ~linearly (§4.3)?
  4. plateau / onset, grid-matched to the control.
  5. CE_best and dCE@1 via the gain_decomp convention -- REPORTED, NOT USED TO DECIDE. 896 params
     against a measured MPS replicate floor of 0.031-0.068 nats cannot be resolved on loss at 2.5M,
     and saying so in advance is the point.

  FALSIFIER for the mechanism: if alpha learns to ~1 (no decay) the model prefers plain addition and
  the "field's choice is better" hypothesis is refuted at this scale. If alpha < 1 but ||e||/||h||
  is unchanged, the decay is not doing what the form claims.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from model import Config as ModelConfig  # noqa: E402
from train import TrainConfig  # noqa: E402
from chunked_runner import run_chunked  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAYER_APPS_PER_S = 1100.0 * 18 * 3
REF = "sd_dense_k5_s0"          # config donor, same convention as run_anneal_local / run_scale_clock


def gate_stats(name: str) -> dict | None:
    ck = ROOT / "checkpoints" / name / "last.pt"
    if not ck.exists():
        return None
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]
    if "inj_a" not in sd:
        return None
    delta = F.softplus(sd["inj_b"])
    alpha = torch.exp(-delta * torch.exp(sd["inj_a"]))
    return dict(alpha_mean=float(alpha.mean()), alpha_min=float(alpha.min()),
                alpha_max=float(alpha.max()), delta_mean=float(delta.mean()),
                n_channels_decaying=int((alpha < 0.99).sum()))


def main():
    ref = torch.load(ROOT / "checkpoints" / REF / "last.pt", map_location="cpu", weights_only=False)
    base_m, base_t = dict(ref["model_cfg"]), dict(ref["train_cfg"])
    tok = base_t["total_tokens"]
    cap = tok * 18 * 3 / LAYER_APPS_PER_S * 2.2

    # gi_gated (alpha_init 0.9999939) is KEPT: it is the record that the near-identity init
    # makes alpha untrainable. gi_gated_a874 is the valid test.
    arms = {"gi_additive": "additive", "gi_gated": "gated", "gi_gated_a874": "gated"}
    out = ROOT / "checkpoints" / "gated_inject_results.json"
    res = json.loads(out.read_text()) if out.exists() else {}

    for name, mode in arms.items():
        if name in res:
            print(f"{name}: already done, skipping", flush=True)
            continue
        over = {"inject_mode": mode}
        if name == "gi_gated":
            over["gate_alpha_init"] = 0.9999939   # the failed near-identity init, on record
        mcfg = ModelConfig(**{**base_m, **over})
        tcfg = TrainConfig(**{**base_t, "run_name": name})
        steps_per_eval = tcfg.eval_every_tokens // (tcfg.batch_size * tcfg.seq_len)
        assert steps_per_eval <= 200, (
            f"{name}: eval_every_steps={steps_per_eval} too sparse for 240s chunks; the run would "
            f"never checkpoint (see train.py's max_seconds branch)")
        print(f"\n=== {name}  inject_mode={mode}  {tok:,} tok  cap {cap/60:.0f}min ===", flush=True)
        res[name] = run_chunked(name, mcfg, tcfg, cap, fresh=True)
        h = res[name]["history"]
        if h:
            c = h[-1]["val_curve"]
            b = min(c, key=c.get)
            print(f"  {name}: best r={b} CE={c[b]:.4f} CE@1={c['1']:.4f} "
                  f"gain={c['1']-c[b]:.4f}", flush=True)
        g = gate_stats(name)
        if g:
            res[name]["gate"] = g
            print(f"  GATE: alpha mean={g['alpha_mean']:.6f} min={g['alpha_min']:.6f} "
                  f"delta mean={g['delta_mean']:.4f}  channels with alpha<0.99: "
                  f"{g['n_channels_decaying']}/448   (alpha~1 => model DECLINED the decay)",
                  flush=True)
        out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
