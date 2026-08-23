"""Is the accumulated gradient of a WEIGHT-TIED projection low-rank compared to an untied one?

The open question behind §6.3. Muon orthogonalises the update via Newton-Schulz, which whitens the
gradient's singular-value spectrum -- it amplifies small singular directions by a large factor. In a
looped model the gradient reaching one shared projection is a SUM over up to 32 applications of the
same matrix, and this project measured those per-loop increments to be almost perfectly aligned
(cos(du_t, du_{t-1}) -> 0.9999, §4.3). Aligned increments plausibly make the accumulated gradient
LOW EFFECTIVE RANK, dominated by a few directions repeated across t.

If so, orthogonalisation is doing something qualitatively different under weight tying than in an
untied stack -- it would be amplifying directions that the tied structure has already suppressed.
Whether that rescues a real collapse or amplifies noise is unknown, and as far as I am aware nobody
reports the spectrum. This measures it.

Design: identical shapes, identical data, identical seed. One model applies ONE block N times
(tied); the other applies N DISTINCT blocks once each (untied). Compare the singular-value spectrum
of the gradient at the same projection. Reports stable rank (||G||_F^2 / ||G||_2^2), participation
ratio, and the mass in the top-k directions -- stable rank is the right summary because it is
insensitive to the many near-zero singular values that dominate a naive rank count.

CPU by default: a handful of forward/backward passes at this width, deliberately kept off the GPU so
it does not contend with training.
"""
from __future__ import annotations
import argparse, math, pathlib, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from model import Config, LoopedTransformer, DecoderLayer, RMSNorm, RotaryEmbedding  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


class UntiedStack(nn.Module):
    """N distinct blocks applied once each -- the untied control for the same total depth."""
    def __init__(self, cfg, n):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(n * cfg.layers_per_loop))
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings)
        self.final_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x):
        e = self.embed(x); h = e
        cos, sin = self.rope(x.shape[1], x.device, e.dtype)
        for l in self.layers:
            h = l(h, cos, sin)
        return F.linear(self.final_norm(h), self.embed.weight)


def spectrum(G):
    s = torch.linalg.svdvals(G.float()).cpu().numpy()
    s2 = s ** 2
    stable_rank = s2.sum() / (s2[0] + 1e-30)
    p = s2 / s2.sum()
    participation = 1.0 / (p ** 2).sum()
    return s, stable_rank, participation, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = args.device
    cfg = Config(state_renorm=False)
    val = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    x = torch.from_numpy(val[:args.batch * args.seq].astype(np.int64)).view(args.batch, args.seq).to(dev)
    y = torch.from_numpy(val[1:args.batch * args.seq + 1].astype(np.int64)).view(args.batch, args.seq).to(dev)

    torch.manual_seed(0)
    tied = LoopedTransformer(cfg).to(dev)
    lg, _ = tied(x, n_loops=args.loops, return_all_loops=False, supervise_idx={args.loops - 1})
    loss = F.cross_entropy(lg[-1].reshape(-1, lg[-1].size(-1)), y.reshape(-1))
    loss.backward()
    Gt = tied.block.layers[0].attn.q_proj.weight.grad.detach().clone()

    torch.manual_seed(0)
    untied = UntiedStack(cfg, args.loops).to(dev)
    out = untied(x)
    F.cross_entropy(out.reshape(-1, out.size(-1)), y.reshape(-1)).backward()
    # the untied control: gradient at the FIRST layer's q_proj (same position in the stack)
    Gu = untied.layers[0].attn.q_proj.weight.grad.detach().clone()

    print(f"loops={args.loops}  layers_per_loop={cfg.layers_per_loop}  "
          f"total applications={args.loops*cfg.layers_per_loop}  shape={tuple(Gt.shape)}")
    print(f"\n{'':22} {'||G||_F':>11} {'stable rank':>12} {'participation':>14} {'top1 mass':>10} {'top8 mass':>10}")
    for lab, G in (("TIED (1 block x N)", Gt), ("UNTIED (N blocks x1)", Gu)):
        s, sr, pr, p = spectrum(G)
        print(f"{lab:22} {G.norm().item():>11.4f} {sr:>12.2f} {pr:>14.2f} "
              f"{p[0]:>10.4f} {p[:8].sum():>10.4f}")
    st, srt, prt, pt = spectrum(Gt); su, sru, pru, pu = spectrum(Gu)
    print(f"\n  stable-rank ratio tied/untied = {srt/sru:.3f}")
    print("  <1 means the TIED gradient is concentrated in fewer directions, which is the condition")
    print("  under which Newton-Schulz orthogonalisation would amplify small directions hardest.")
    print(f"\n  singular values (normalised to s[0]), first 12:")
    print(f"    tied   " + " ".join(f"{v/st[0]:.3f}" for v in st[:12]))
    print(f"    untied " + " ".join(f"{v/su[0]:.3f}" for v in su[:12]))



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
    _persist_stdout("grad_spectrum", _out)

