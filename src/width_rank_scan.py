"""Is the depth-key rank collapse width-independent? sec4.7e claims it; this measures it.

sec4.7e argues the collapse is structural -- a tied loop has one W_K and cannot decorrelate a
collinear depth stream, where an unshared stack gets that free from one projection per layer. That
argument does not depend on width, but until now it was measured at ONE width (448), which
`submission/LIMITATIONS.md` named as a gap.

Rank at initialisation needs no training, so the gap closes for free: build the same tied
architecture at several widths and measure the depth-key effective rank of each.

    python src/width_rank_scan.py
"""
from __future__ import annotations
import pathlib, sys
import numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config, LoopedTransformer
ROOT = pathlib.Path(__file__).resolve().parents[1]
LOOPS = 32
# (hidden, heads, kv_heads, head_dim) -- head_dim held near-constant so only width really varies
WIDTHS = [(224, 2, 1, 112), (320, 4, 2, 80), (448, 4, 2, 112), (640, 8, 4, 80), (896, 8, 4, 112)]

def rank_at(width, heads, kv, hd, x):
    cfg = Config(hidden_size=width, n_heads=heads, n_kv_heads=kv, head_dim=hd,
                 intermediate_size=width * 3, state_renorm=False)
    torch.manual_seed(0)
    m = LoopedTransformer(cfg); m.eval()
    caps = []
    h = m.block.layers[0].register_forward_pre_hook(lambda mod, inp: caps.append(inp[0].detach()))
    with torch.no_grad():
        m(x, n_loops=LOOPS, supervise_idx={LOOPS - 1})
    h.remove()
    L = m.block.layers[0]
    D = L.attn.n_kv * L.attn.d_h
    H = torch.stack(caps, dim=2)
    with torch.no_grad():
        Hn = L.norm1(H)
        k = L.attn.k_norm(L.attn.k_proj(Hn).view(*Hn.shape[:3], L.attn.n_kv, L.attn.d_h))
    K = k.reshape(-1, LOOPS, D).float()
    s = torch.linalg.svdvals(K)
    er = float(((s.sum(-1) ** 2) / (s.pow(2).sum(-1) + 1e-12)).mean())
    Kn = torch.nn.functional.normalize(K, dim=-1)
    C = torch.bmm(Kn, Kn.transpose(1, 2))
    iu = torch.triu_indices(LOOPS, LOOPS, offset=1)
    return er, float(C[:, iu[0], iu[1]].mean()), sum(p.numel() for p in m.parameters())

def main() -> int:
    val = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    x = torch.from_numpy(val[: 2 * 128].astype(np.int64)).view(2, 128)
    print(f"UNTRAINED depth-key effective rank vs width (tied, 3-layer block, {LOOPS} loops)")
    for w, h, kv, hd in WIDTHS:
        er, cos, n = rank_at(w, h, kv, hd, x)
        print(f"  width {w:4d} ({n/1e6:5.2f}M params):  eff rank = {er:5.3f} / {LOOPS}   "
              f"mean pairwise cos = {cos:.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
