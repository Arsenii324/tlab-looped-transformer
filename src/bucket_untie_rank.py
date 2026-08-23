"""If weight tying costs depth-key rank, how much does PARTIAL untying buy? Zero training.

sec4.7e's corrected mechanism is that a tied loop has one `W_K` and therefore cannot decorrelate a
collinear depth-state stream, while an untied stack gets that decorrelation free from distinct
per-layer projections. That is an explanation. This turns it into a DOSE-RESPONSE prediction, and it
is the cheap check that must come before spending a training run on partial untying.

Take the trained tied model's real depth states, then project them with `nb` distinct random W_K
assigned by loop index (t % nb) and measure the effective rank of the resulting key stream.

    python src/bucket_untie_rank.py [ckpt]
"""
from __future__ import annotations
import pathlib, sys
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import LoopedTransformer, Config
ROOT = pathlib.Path(__file__).resolve().parents[1]
LOOPS, B, S = 32, 8, 128

def eff_rank(K):
    s = torch.linalg.svdvals(K.float())
    return float(((s.sum(-1) ** 2) / (s.pow(2).sum(-1) + 1e-12)).mean())

def main():
    ck_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sd_dense_k5_s0/last.pt"
    torch.manual_seed(0)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    m = LoopedTransformer(Config(**ck["model_cfg"])); m.load_state_dict(ck["model"]); m.eval()
    val = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    x = torch.from_numpy(val[: B * S].astype(np.int64)).view(B, S)
    caps = []
    h = m.block.layers[0].register_forward_pre_hook(lambda mod, inp: caps.append(inp[0].detach()))
    with torch.no_grad():
        m(x, n_loops=LOOPS, supervise_idx={LOOPS - 1})
    h.remove()
    layer = m.block.layers[0]
    D = layer.attn.n_kv * layer.attn.d_h
    H = torch.stack(caps, dim=2)
    with torch.no_grad():
        Hn = layer.norm1(H)
        Ktied = layer.attn.k_norm(layer.attn.k_proj(Hn).view(*Hn.shape[:3], layer.attn.n_kv,
                                                             layer.attn.d_h))
    Hf = Hn.reshape(-1, LOOPS, Hn.shape[-1])
    print(f"  states, no projection      : {eff_rank(Hf):6.3f} / {LOOPS}")
    print(f"  TIED W_K (this model)      : {eff_rank(Ktied.reshape(-1, LOOPS, D)):6.3f} / {LOOPS}")
    Hdim = Hn.shape[-1]
    for nb in (2, 4, 8, 32):
        torch.manual_seed(1)
        Ws = [torch.randn(Hdim, D) / Hdim ** 0.5 for _ in range(nb)]
        out = torch.empty(Hf.shape[0], LOOPS, D)
        for t in range(LOOPS):
            out[:, t, :] = Hf[:, t, :] @ Ws[t % nb]
        out = torch.nn.functional.normalize(out, dim=-1)
        cost = (nb - 1) * Hdim * D
        print(f"  W_K in {nb:2d} buckets          : {eff_rank(out):6.3f} / {LOOPS}"
              f"   (+{cost:,} params, +{cost/9_064_608*100:.1f}%)")

if __name__ == "__main__":
    main()
