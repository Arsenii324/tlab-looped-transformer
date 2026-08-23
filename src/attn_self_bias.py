"""XSA's 'attention similarity bias' (arXiv 2603.09078), measured against LOOP index.

Their Figure 1 plots cos(y_i, v_i) -- attention output vs the token's OWN value vector -- per layer,
and reports an increasing trend in layer index. Their rationale (main.tex:126): the self-position
information "has a direct residual path to the following [FFN]", so attention capacity spent
re-encoding it is "unnecessary ... and harmful, because it creates a competition between modeling the
contextual vs point-wise feature".

In a WEIGHT-TIED loop, layer index IS loop index -- so their trend becomes a prediction about depth:
attention should increasingly re-add what the residual already carries, i.e. stop doing context work.
If it holds, it is a CAUSAL account of sec4.3's cos(du_t, du_{t-1}) -> 0.9999 rather than a restatement
of it, and it composes with sec4.3's other two numbers (keys flat 25.13->21.36 while ||v|| falls 2x).

PREDICTION registered before running: cos(y_i, v_i) RISES with loop index, most steeply past the
useful band (~loop 8). FALSIFIER: flat or falling => the drift is not attention re-encoding self, and
XSA's phenomenon does not transfer to the looped regime.
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import LoopedTransformer, Config, apply_rope
ROOT = pathlib.Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--loops", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2); ap.add_argument("--seq", type=int, default=128)
    a = ap.parse_args()
    p = pathlib.Path(a.ckpt); p = p / "last.pt" if p.is_dir() else p
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = LoopedTransformer(Config(**ck["model_cfg"])); m.load_state_dict(ck["model"]); m.eval()

    val = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    x = torch.from_numpy(val[: a.batch * a.seq].astype(np.int64)).view(a.batch, a.seq)

    caps = {i: [] for i in range(len(m.block.layers))}
    hooks = [l.register_forward_pre_hook(
                (lambda i: (lambda mod, inp: caps[i].append(inp[0].detach())))(i))
             for i, l in enumerate(m.block.layers)]
    with torch.no_grad():
        m(x, n_loops=a.loops, supervise_idx={a.loops - 1})
    for h in hooks: h.remove()

    cos_now, cos_v = m.rope(a.seq, x.device, torch.float32)
    out = {}
    print(f"checkpoint {p}   loops={a.loops}\n")
    print(f"  {'loop':>5s}  " + "  ".join(f"L{i} cos(y,v)" for i in range(len(m.block.layers))))
    for t in range(a.loops):
        row = []
        for i, layer in enumerate(m.block.layers):
            h = caps[i][t]; at = layer.attn
            with torch.no_grad():
                xn = layer.norm1(h); B, T, _ = xn.shape
                q = at.q_norm(at.q_proj(xn).view(B, T, at.n_h, at.d_h)).transpose(1, 2)
                k = at.k_norm(at.k_proj(xn).view(B, T, at.n_kv, at.d_h)).transpose(1, 2)
                v = at.v_proj(xn).view(B, T, at.n_kv, at.d_h).transpose(1, 2)
                q, k = apply_rope(q, k, cos_now, cos_v)
                kk = k.repeat_interleave(at.groups, dim=1); vv = v.repeat_interleave(at.groups, dim=1)
                y = F.scaled_dot_product_attention(q, kk, vv, is_causal=True, scale=at.scale)
                c = F.cosine_similarity(y.float(), vv.float(), dim=-1)   # [B, n_h, T]
            row.append(float(c.mean()))
        out[t + 1] = row
        if (t + 1) in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64):
            print(f"  {t+1:5d}  " + "  ".join(f"{r:11.4f}" for r in row))
    (ROOT / "checkpoints" / "attn_self_bias.json").write_text(json.dumps(out, indent=2))
    first, last = out[1], out[a.loops]
    print("\n  trend loop 1 -> %d: " % a.loops +
          "  ".join(f"L{i}: {first[i]:+.4f} -> {last[i]:+.4f} ({last[i]-first[i]:+.4f})"
                    for i in range(len(first))))

if __name__ == "__main__":
    main()
