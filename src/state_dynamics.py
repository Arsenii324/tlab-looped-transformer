"""What the loop map actually does to the hidden state, measured in the space the readout can see.

Why this exists as a separate instrument from eval.py's `contraction_estimate`. That one reports
||h_clean - h_noisy|| and its consecutive ratio, and on the winning checkpoint the ratio decays
1.43 -> 1.014 and never drops below 1. That was read as "the map is marginally non-contracting".
Two things are wrong with reading it that way, both visible in the raw numbers:

  1. From loop ~20 on, the distance grows by a CONSTANT ADDITIVE ~500/loop (33883.7, 34388.0,
     34892.7, 35397.7, ... -- first differences 504.3, 504.7, 505.1, 505.4). For any d_t = a + b*t,
     the ratio d_{t+1}/d_t -> 1 no matter what b is. So "ratio -> 1" is arithmetic forced by linear
     drift, not a statement about contraction. The instrument cannot separate a contracting map
     with linear drift from a neutral one.
  2. `LoopedTransformer.readout` applies `final_norm` (RMSNorm) before the tied LM head, so the
     logits depend on h ONLY through its direction. A raw L2 distance between states measures a
     quantity the model's predictions are exactly invariant to. With state_renorm=False the state
     norm grows without bound, so raw distance is dominated by the component the readout discards.

So this reports, per loop, both the raw quantities (for continuity with the published table) and
the normalized ones, and adds the quantity the perturbation test cannot supply at all:

  ||h_t - h_{t-1}||         is the state still MOVING? (a perturbation test cannot tell "converged
                            to a fixed point" from "orbiting a limit cycle" -- both give a stable
                            clean/noisy distance.)
  cos(dh_t, dh_{t-1})       is that motion PERSISTENT (~+1, traveling somewhere) or OSCILLATING
                            (~-1, two-cycle) or a random walk (~0)?
  and the same two on the unit state h/||h||, which is the readout-visible version.

Predictions written before running (per CLAUDE.md sec 1):
  - if saturation at loop ~8 is fixed-point convergence, unit-state step ||u_t - u_{t-1}|| should
    decay toward 0 well before loop 8 and be negligible past it.
  - if it is a limit cycle, unit step stays bounded away from 0 with unit increment cosine < 0.
  - if the state keeps traveling (unit step bounded away from 0, increment cosine > 0) while val CE
    stops improving, then saturation is NOT a dynamics property at all -- it is the readout, or the
    training distribution over loop counts, and the contraction story in report.md sec 4.3 does not
    survive.

Third probe: is the thing the loop settles toward an attractor of the BLOCK, or an attractor of
the constant forcing term? With inject_mode="additive" the loop is h_{t+1} = block(h_t + e), and `e`
is constant in t -- a standing force with its own equilibrium, entirely separate from whether the
block itself is contractive. So this also rolls the SAME trained weights out with inject_mode flipped
to "none" (block iterated alone from the same h0+e start) and reports the same motion metrics. If the
block alone converges while the injected run keeps moving, the motion is forced, not intrinsic; if the
block alone keeps moving while the injected run settles, the forcing term is what pins the state.
That rollout is deliberately off-distribution -- the weights were trained with injection at every
loop -- so it is a probe of the map, not a performance measurement, and no CE is reported from it.

Also probes one specific claim about early exit: Qwen3 normalizes q and k per head but not v, so a
mixed-exit-depth KV cache is said to be dominated by deeply-processed tokens' larger value vectors.
In THIS architecture attention reads `norm1(h)`, not h, so v_proj's input is already scale-free --
the hooks below measure ||v|| per loop to settle it by measurement rather than by reading the code.

Usage: python src/state_dynamics.py <ckpt_dir> [<ckpt_dir> ...] [--max-loops 64]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig, LoopedTransformer  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
BATCH = 8          # matches eval.py's contraction_estimate, so pert_abs is directly comparable
NOISE = 1.0        # ditto
PERT_SEED = 12345  # ditto -- eval.py seeds this immediately before drawing the h0 perturbation

if torch.backends.mps.is_available():
    torch.mps.set_per_process_memory_fraction(12.0e9 / torch.mps.recommended_max_memory())


def _mean(x):
    return float(x.mean().item())


def _norm(h):                      # [B,T,H] -> [B,T]
    return h.float().norm(dim=-1)


def _unit(h):
    return h.float() / _norm(h).unsqueeze(-1).clamp_min(1e-12)


def _cos(a, b):                    # [B,T,H] x [B,T,H] -> scalar mean cosine
    a, b = a.float(), b.float()
    num = (a * b).sum(-1)
    den = (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)
    return _mean(num / den)


@torch.no_grad()
def rollout_with_v_probe(model, x, n_loops, h0_noise=0.0, seed=None):
    """One forward through the model's OWN loop (model.forward, not a transcription of it), with
    hooks recording ||v|| and ||norm1(h)|| per sublayer call. Returns (states, v_norm, attn_in_norm),
    the last two averaged over the layers_per_loop calls inside each loop."""
    v_hits, n1_hits = [], []
    handles = []
    for layer in model.block.layers:
        handles.append(layer.attn.v_proj.register_forward_hook(
            lambda m, i, o: v_hits.append(_mean(_norm(o)))))
        handles.append(layer.norm1.register_forward_hook(
            lambda m, i, o: n1_hits.append(_mean(_norm(o)))))
    try:
        if seed is not None:
            torch.manual_seed(seed)
        _, _, states = model(x, n_loops=n_loops, return_all_loops=False, h0_noise=h0_noise,
                             supervise_idx=set(), return_states=True)
    finally:
        for h in handles:
            h.remove()
    L = len(model.block.layers)
    v_per_loop = [float(np.mean(v_hits[t * L:(t + 1) * L])) for t in range(n_loops)]
    n1_per_loop = [float(np.mean(n1_hits[t * L:(t + 1) * L])) for t in range(n_loops)]
    return states, v_per_loop, n1_per_loop


@torch.no_grad()
def analyse(model, val, seq_len, n_loops, device):
    rng = np.random.default_rng(0)   # same draw as eval.py's contraction_estimate
    ix = rng.integers(0, len(val) - seq_len - 1, size=BATCH)
    x = torch.from_numpy(np.stack([val[i:i + seq_len] for i in ix]).astype(np.int64)).to(device)

    clean, v_norm, attn_in_norm = rollout_with_v_probe(model, x, n_loops, 0.0, seed=PERT_SEED)
    noisy, _, _ = rollout_with_v_probe(model, x, n_loops, NOISE, seed=PERT_SEED)

    saved_mode = model.cfg.inject_mode          # forcing-term probe: same weights, no standing input
    model.cfg.inject_mode = "none"
    try:
        noinj, _, _ = rollout_with_v_probe(model, x, n_loops, 0.0, seed=PERT_SEED)
    finally:
        model.cfg.inject_mode = saved_mode
    un = [_unit(h) for h in noinj]

    m = {k: [] for k in ("state_norm", "pert_abs", "pert_rel", "pert_unit", "pert_cos",
                         "step_abs", "step_rel", "step_unit", "incr_cos", "unit_incr_cos",
                         "noinj_state_norm", "noinj_step_unit", "noinj_unit_incr_cos",
                         "noinj_unit_dist")}
    uc = [_unit(h) for h in clean]
    for t in range(n_loops):
        c, nz = clean[t].float(), noisy[t].float()
        cn = _norm(c)
        m["state_norm"].append(_mean(cn))
        d = c - nz
        m["pert_abs"].append(_mean(_norm(d)))
        m["pert_rel"].append(_mean(_norm(d) / cn.clamp_min(1e-12)))
        m["pert_unit"].append(_mean(_norm(_unit(c) - _unit(nz))))
        m["pert_cos"].append(_cos(c, nz))
        if t >= 1:
            s = c - clean[t - 1].float()
            m["step_abs"].append(_mean(_norm(s)))
            m["step_rel"].append(_mean(_norm(s) / _norm(clean[t - 1]).clamp_min(1e-12)))
            m["step_unit"].append(_mean(_norm(uc[t] - uc[t - 1])))
        else:
            for k in ("step_abs", "step_rel", "step_unit"):
                m[k].append(float("nan"))
        if t >= 2:
            s1 = c - clean[t - 1].float()
            s0 = clean[t - 1].float() - clean[t - 2].float()
            m["incr_cos"].append(_cos(s1, s0))
            m["unit_incr_cos"].append(_cos(uc[t] - uc[t - 1], uc[t - 1] - uc[t - 2]))
        else:
            m["incr_cos"].append(float("nan"))
            m["unit_incr_cos"].append(float("nan"))
        m["noinj_state_norm"].append(_mean(_norm(noinj[t])))
        m["noinj_unit_dist"].append(_mean(_norm(un[t] - uc[t])))
        m["noinj_step_unit"].append(_mean(_norm(un[t] - un[t - 1])) if t >= 1 else float("nan"))
        m["noinj_unit_incr_cos"].append(
            _cos(un[t] - un[t - 1], un[t - 1] - un[t - 2]) if t >= 2 else float("nan"))
    m["v_norm"] = v_norm
    m["attn_in_norm"] = attn_in_norm
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--max-loops", type=int, default=64)
    ap.add_argument("--val", type=str, default=str(ROOT / "data" / "val.bin"))
    ap.add_argument("--device", type=str, default=None, help="force cpu for a smoke test")
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    val = np.memmap(args.val, dtype=np.uint16, mode="r")

    for i, cp in enumerate(args.checkpoints):
        if i:
            time.sleep(20)   # MPS driver dislikes rapid back-to-back GPU workloads (LOG.md, report sec 6)
        p = pathlib.Path(cp)
        if p.is_dir():
            p = p / "last.pt"
        ckpt = torch.load(p, map_location=device, weights_only=False)
        cfg = ModelConfig(**ckpt["model_cfg"])
        model = LoopedTransformer(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        seq_len = ckpt["train_cfg"]["seq_len"]

        print(f"\n=== {p.parent.name}  step={ckpt.get('step')} tokens={ckpt.get('tokens')} "
              f"state_renorm={cfg.state_renorm} inject={cfg.inject_mode} ===", flush=True)
        m = analyse(model, val, seq_len, args.max_loops, device)

        cols = ("state_norm", "pert_abs", "pert_rel", "pert_unit", "pert_cos", "step_abs",
                "step_rel", "step_unit", "incr_cos", "unit_incr_cos", "v_norm", "attn_in_norm",
                "noinj_step_unit", "noinj_unit_incr_cos", "noinj_unit_dist")
        hdr = ("loop", "||h||", "pert_abs", "pert_rel", "pert_u", "pert_cos", "step_abs",
               "step_rel", "step_u", "incr_cos", "u_incr_cos", "||v||", "||n1h||",
               "NI_step_u", "NI_icos", "NI_dist")
        print(("{:>5}" + "{:>10}" * 15).format(*hdr))
        for t in range(args.max_loops):
            if t + 1 > 12 and (t + 1) % 4 and t + 1 != args.max_loops:
                continue     # dense early where things move, every 4th later; full table in the JSON
            row = [t + 1] + [m[k][t] for k in cols]
            print(("{:>5}" + "{:>10.4f}" * 15).format(*row))

        out = p.parent / f"dynamics_{p.parent.name}.json"
        out.write_text(json.dumps(dict(checkpoint=str(p), step=ckpt.get("step"),
                                       tokens=ckpt.get("tokens"), state_renorm=cfg.state_renorm,
                                       n_loops=args.max_loops, batch=BATCH, noise=NOISE, **m),
                                  indent=2))
        print(f"wrote {out}", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
