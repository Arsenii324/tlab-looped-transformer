"""Train ONE annealed model locally, so an annealed CHECKPOINT exists.

WHY THIS RUNS LOCALLY WHEN EVERYTHING ELSE IS ON DATASPHERE. Every DS job in this project declared
only `results.json` under `outputs:`, so the weights it wrote were never returned -- ~20 checkpoints
are unrecoverable (§6.0 row 23). That blocks the one test the reviewer ranked highest for §4.7:

    §4.7's exit-rule negative was measured on a DENSE-supervised checkpoint. Published work argues
    exit quality is a property of the TRAJECTORY rather than the halt head. Under the anchor reading,
    a dense-supervised model has every loop pinned to the output manifold, so confidence signals are
    near-saturated everywhere; an ANNEALED model's intermediate states are unpinned and a
    trajectory-level signal would have something to read.

So: the missing cell is the exit rules on an annealed checkpoint, and no annealed checkpoint exists.
This makes one. Config is matched to `sd_dense_k5_s0` (already local) so the pair is comparable:
same shape, same schedule U[4,32], same 2.5M tokens, same seed -- differing ONLY in that supervision
switches to terminal-only for the final 25% of steps.

Outcomes, both worth having and neither assumed:
  rules WORK on the annealed model -> §4.7 sharpens from "depth demand is unpredictable" to
      "unpredictable in a dense-supervised trajectory, and predictable once the anchor is released",
      which ties the method to the negative.
  rules STILL FAIL -> the negative gets materially stronger, because the trajectory explanation the
      literature offers has been ruled out rather than left open.
"""
from __future__ import annotations
import json, pathlib, sys
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config as ModelConfig
from train import TrainConfig
from chunked_runner import run_chunked

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAYER_APPS_PER_S = 1100.0 * 18 * 3      # measured MPS throughput at mu_rec=18


REF = "sd_dense_k5_s0"   # the dense-supervised half of the pair; already trained and local


def main():
    # Derive the config FROM the reference checkpoint rather than re-typing it. Hand-matching got
    # this wrong once in a way that cost 727s and produced nothing: `eval_every_tokens` was typed as
    # 1_250_000 against the reference's 312_500, which at batch_size=8 puts the only checkpoint-save
    # site 610 steps away while a chunk reaches ~250 -- so the run never saved, never resumed, and
    # restarted from zero every chunk. Deriving makes every axis except the intervention identical
    # by construction, which is also exactly what "matched pair" is supposed to mean.
    name = "local_anneal_sw75_s0"
    ref = torch.load(ROOT / "checkpoints" / REF / "last.pt", map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**ref["model_cfg"])
    tc = dict(ref["train_cfg"])
    tc.update(run_name=name, supervise_k_final=1, supervise_switch_frac=0.75)
    tcfg = TrainConfig(**tc)
    tok = tcfg.total_tokens

    # The chunking workaround must not be able to starve the save site (see train.py's max_seconds
    # branch). Stated as a check rather than a comment because it is cheap and it already failed.
    steps_per_eval = tcfg.eval_every_tokens // (tcfg.batch_size * tcfg.seq_len)
    assert steps_per_eval <= 200, (
        f"eval_every_steps={steps_per_eval} is too sparse for 240s chunks (~250 steps); "
        f"the run would never checkpoint")
    diff = {k: (v, tc[k]) for k, v in ref["train_cfg"].items() if tc[k] != v}
    print(f"PRE-FLIGHT  {name} derived from {REF}; differs only in: {diff}", flush=True)

    cap = tok * 18 * 3 / LAYER_APPS_PER_S * 2.2
    print(f"  k=5 -> 1 at 75% of steps, {tok:,} tokens, cap {cap/60:.0f} min", flush=True)
    print(f"  matched to the existing local dense checkpoint sd_dense_k5_s0 "
          f"(same shape/schedule/tokens/seed; supervision is the only difference)", flush=True)
    out = ROOT / "checkpoints" / "anneal_local_results.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    if name in res:
        print("already done"); return 0
    res[name] = run_chunked(name, mcfg, tcfg, cap, fresh=True)
    h = res[name]["history"]
    if h:
        c = h[-1]["val_curve"]; b = min(c, key=c.get)
        print(f"  {name}: best r={b} CE={c[b]:.4f} gain={c['1']-c[b]:.4f}", flush=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
