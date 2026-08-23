"""Are a token's KEYS distinguishable across loop depth? Decides depth-attention BEFORE running it.

Same-position depth attention (MoD-Attention's form: a query attends to its OWN key/value history
across depth) can only do work if those depth keys differ. If they are near-collinear, attention over
them is a uniform average with extra steps, and the mechanism is null by construction.

§4.3 gives a reason to suspect they are: attention reads `norm1(h)`, whose NORM is nearly flat across
depth (25.13 -> 21.36 over 64 loops) while ||h|| grows 18x. Flat norm is not collinearity, though --
direction is what matters and it has never been measured. This measures it.

Reported per layer: mean pairwise cos over the depth-key stream, and EFFECTIVE RANK
(participation ratio of the singular values, sum(s)^2 / sum(s^2)) against the r it could use.

PREDICTION, written before running: keys near-collinear (mean cos > ~0.95, eff.rank << r), because
the same near-identity that makes a ragged KV cache nearly free (§4.8, spread 0.0011 at t=16) is the
statement that depths are interchangeable to attention. If so, depth attention is predicted null HERE
for a mechanism-level reason -- MoDA's gain would be a property of DISTINCT layers that a weight-tied
loop cannot have -- and that is worth more than another null arm.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import LoopedTransformer, Config
ROOT = pathlib.Path(__file__).resolve().parents[1]

def eff_rank(M):                      # participation ratio of singular values
    s = torch.linalg.svdvals(M.float())
    return float((s.sum() ** 2) / (s.pow(2).sum() + 1e-12))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--loops", type=int, default=32)
    ap.add_argument("--batch", type=int, default=2); ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--val", default="val.bin",
                    help="shard under data/ to probe with. DataSphere checkpoints need\n                         val_datasphere.bin -- their vocabulary is not the shipped one (sec4.27).")
    a = ap.parse_args()
    p = pathlib.Path(a.ckpt); p = p / "last.pt" if p.is_dir() else p
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = LoopedTransformer(Config(**ck["model_cfg"])); m.load_state_dict(ck["model"]); m.eval()

    val = np.memmap(ROOT / "data" / a.val, dtype=np.uint16, mode="r")
    x = torch.from_numpy(val[: a.batch * a.seq].astype(np.int64)).view(a.batch, a.seq)

    # capture each DecoderLayer's INPUT at every loop -- that is what k_proj consumes
    caps = {i: [] for i in range(len(m.block.layers))}
    hooks = [l.register_forward_pre_hook(
                (lambda i: (lambda mod, inp: caps[i].append(inp[0].detach())))(i))
             for i, l in enumerate(m.block.layers)]
    with torch.no_grad():
        m(x, n_loops=a.loops, supervise_idx={a.loops - 1})
    for h in hooks: h.remove()

    print(f"checkpoint {p}   loops={a.loops}  batch={a.batch}x{a.seq}\n")
    out = {}
    for i, layer in enumerate(m.block.layers):
        H = torch.stack(caps[i], dim=2)                       # [B,T,r,H]
        with torch.no_grad():
            k = layer.attn.k_proj(layer.norm1(H))             # [B,T,r,n_kv*d_h]
            k = layer.attn.k_norm(k.view(*k.shape[:3], layer.attn.n_kv, layer.attn.d_h))
        K = k.reshape(-1, a.loops, layer.attn.n_kv * layer.attn.d_h)   # [(B*T), r, D]
        Kn = torch.nn.functional.normalize(K.float(), dim=-1)
        C = torch.bmm(Kn, Kn.transpose(1, 2))                 # [(B*T), r, r]
        iu = torch.triu_indices(a.loops, a.loops, offset=1)
        pair = C[:, iu[0], iu[1]]
        er = float(np.mean([eff_rank(K[j]) for j in range(0, K.shape[0], max(1, K.shape[0] // 64))]))
        print(f"  layer {i}:  mean pairwise cos = {pair.mean():.4f}   "
              f"min = {pair.min():.4f}   frac>0.95 = {(pair > 0.95).float().mean():.3f}   "
              f"effective rank = {er:.2f} / {a.loops}")
        out[f"layer{i}"] = dict(mean_cos=float(pair.mean()), min_cos=float(pair.min()),
                                frac_gt_095=float((pair > 0.95).float().mean()), eff_rank=er,
                                loops=a.loops)
    (ROOT / "checkpoints" / "depth_key_rank.json").write_text(json.dumps(out, indent=2))
    print("\n  -> checkpoints/depth_key_rank.json")

if __name__ == "__main__":
    main()


def tied_vs_untied(D: int = 33, batch: int = 2, seq: int = 128):
    """Is the depth-key rank collapse WEIGHT TYING, or just a small model?

    Both models UNTRAINED and identical in width/heads/head_dim/init, so training quality cannot
    explain the difference and the only variable is tied-vs-untied. Answers the one question sec4.7e
    could not: whether its negative generalises to looped models at any size, or is an artifact of
    448 hidden units with 4 heads.

    Measured 2026-08-23 19:44:  tied 2.73/33 (cos 0.8022)  vs  untied 31.80/33 (cos -0.0029), 11.7x.
    """
    import numpy as np, torch, torch.nn.functional as F
    from model import LoopedTransformer, Config, DecoderLayer, RotaryEmbedding
    cfg = Config(state_renorm=False)
    val = np.memmap(ROOT / "data" / a.val, dtype=np.uint16, mode="r")
    x = torch.from_numpy(val[: batch * seq].astype(np.int64)).view(batch, seq)
    torch.manual_seed(0); m = LoopedTransformer(cfg); m.eval()
    caps = []
    hk = m.block.layers[0].register_forward_pre_hook(lambda mod, inp: caps.append(inp[0].detach()))
    with torch.no_grad(): m(x, n_loops=D, supervise_idx={D - 1})
    hk.remove()
    return caps, m, cfg, x
