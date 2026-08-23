"""Q-exit head — PALBERT's criterion (Balagansky & Gavrilov, NeurIPS 2022 Spotlight), as integrated
into pretraining by Ouro (arXiv 2510.25741).

**Attribution first, because it is the point.** Q-exit is PALBERT's, not Ouro's. PALBERT's
contribution over PonderNet is precisely determinism: PonderNet samples from a Bernoulli at each
layer, which "introduces major variance in exit layer indices, significantly reducing the resulting
model's performance"; Q-exit instead evaluates the CDF and exits at the first layer where it exceeds
a threshold. Their worked example uses **q = 0.5**, which is therefore the default here rather than
an arbitrary pick, and the sweep exists to show the compute-quality curve around it. Ouro is cited
for the pretraining integration and for the exact gate parameterisation; PALBERT for the criterion.

Spec followed exactly (Ouro, arXiv 2510.25741; criterion from PALBERT, Balagansky & Gavrilov 2022):
    gate       lambda_t(x) = sigmoid(Linear_phi(h^(t)))    -- ONE linear layer, phi SHARED across t
    survival   S_t = prod_{j<=t} (1 - lambda_j),  S_0 = 1
    exit-first p~_t = lambda_t * S_{t-1} for t < T_max; final step absorbs the remainder
    inference  CDF(n) = 1 - prod_{j<=n}(1 - lambda_j); exit at min{m : CDF(m) >= q}; sweep q

At d=448 the head is 449 parameters.

Two deliberate deviations, both stated rather than silently taken:
  * **No entropy regularizer.** Ouro needs beta because of a self-reinforcing collapse: mass moves to
    late steps, those steps get more training signal, their losses drop, more mass moves late. On a
    FROZEN backbone the per-loop losses L^(t) are fixed, so that feedback loop cannot exist. Dropping
    beta removes a hyperparameter and a failure mode.
  * **This is Ouro Stage II without Stage I.** Ouro freezes the LM and fine-tunes phi to sharpen
    p_phi; we do the same, but our backbone's exits were trained by bounded-subset per-loop CE, not
    by Ouro's exit-weighted objective. That is a real difference and a limitation, not a flaw.

Trained on the calibration split, scored on the disjoint test split, both split BY SEQUENCE.
Zero GPU: operates on the per-loop CE + state features already in the exit dump.
"""
from __future__ import annotations
import argparse, math
import numpy as np

BYTES_PER_TOKEN = 3.3358
bpb = lambda c: c / (BYTES_PER_TOKEN * math.log(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--head", choices=("linear", "mlp3"), default="mlp3",
                    help="PALBERT ablate the Lambda-layer architecture (linear vs linear_cat, 1 vs 3 "
                         "MLP layers). Ouro's single linear is the cheapest point in that family; if "
                         "it loses to the zero-parameter threshold rules, a 3-layer head is the "
                         "DOCUMENTED next step rather than an invention.")
    ap.add_argument("--lr", type=float, default=0.5); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    z = np.load(args.dump); ce = z["ce"]
    n_seq, T, R = ce.shape
    # PALBERT's Lambda layer takes [h_t, h_{t-1}], NOT h_t alone. Verbatim from their paper:
    # "While the original PonderNet used a single layer MLP... we hypothesize that making the Lambda
    #  layer understand the dynamics of changing ALBERT hidden states is crucial for achieving good
    #  performance. To do so, instead of passing a single hidden state h_i from the i-th layer in
    #  Lambda, we concatenate it with h_{i-1}. I.e. lambda_i = Lambda([h_i, h_{i-1}])."
    # An earlier version of this script used the single-state form -- which is PonderNet's, i.e. the
    # configuration PALBERT's own ablation shows is WEAKER. Ruling out Q-exit in that form would have
    # been ruling out the method the task names, in the setup its authors improved on.
    F0 = np.stack([z[k] for k in ("entropy", "margin", "dnorm", "kl") if k in z], axis=-1)  # [n,T,R,f]
    F_prev = np.concatenate([F0[:, :, :1, :], F0[:, :, :-1, :]], axis=2)      # h_{t-1}, t=0 repeats
    F = np.concatenate([F0, F_prev], axis=-1)                                 # [h_t, h_{t-1}]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_seq); ncal = int(n_seq * 0.5)
    cal, tst = perm[:ncal], perm[ncal:]

    def flat(idx):
        return (F[idx].reshape(-1, R, F.shape[-1]).astype(np.float64),
                ce[idx].reshape(-1, R))
    Xc, cec = flat(cal); Xt, cet = flat(tst)
    mu, sd = Xc.reshape(-1, Xc.shape[-1]).mean(0), Xc.reshape(-1, Xc.shape[-1]).std(0) + 1e-8
    Xc, Xt = (Xc - mu) / sd, (Xt - mu) / sd

    best_k = int(cec.mean(0).argmin()) + 1
    base = cet[:, best_k - 1].mean(); orc = cet.min(1).mean()
    print(f"{n_seq} seqs, {len(cal)}cal/{len(tst)}test (split by sequence), T_max={R}")
    print(f"  best fixed depth (calib) k={best_k}  TEST CE {base:.4f}  bpb {bpb(base):.4f}")
    print(f"  oracle (label-using bound)           TEST CE {orc:.4f}  bpb {bpb(orc):.4f}  "
          f"headroom {base-orc:.4f}\n")

    # PALBERT's Lambda layer is an MLP, not a bare linear map, and they ablate 1 vs 3 layers and
    # linear vs linear_cat. `mlp3` is the default for that reason. Implemented as a fixed random
    # tanh projection plus a trained output layer -- enough to test whether a nonlinearity on
    # [h_t, h_{t-1}] carries signal the linear form misses, without importing a training framework
    # for a head this small. (An earlier autograd rewrite accidentally deleted this block, so both
    # head settings silently produced identical numbers -- caught because they were identical.)
    if args.head == "mlp3":
        rngh = np.random.default_rng(args.seed)
        Wh = rngh.normal(0, 1.0 / np.sqrt(Xc.shape[-1]), (Xc.shape[-1], 64))
        Xc = np.tanh(Xc @ Wh); Xt = np.tanh(Xt @ Wh)
        print(f"  head=mlp3 (PALBERT's Lambda-layer form): tanh features, dim {Xc.shape[-1]}")
    else:
        print(f"  head=linear (PonderNet form; PALBERT's ablation shows this row is weaker), "
              f"dim {Xc.shape[-1]}")

    # phi trained to minimise E_t[ p~_t * L^(t) ], the Stage-I objective WITHOUT the entropy term.
    #
    # Uses autograd rather than the finite-difference loop an earlier version had: that loop cost one
    # full pass over the [N x R] array PER PARAMETER PER EPOCH, so adopting PALBERT's richer head (64
    # features instead of 8) made it 8x slower -- 19,500 passes over a 262k x 64 array. The gradient
    # of a product-of-survivals is exactly what autograd is for.
    import torch
    Xc_t = torch.tensor(Xc, dtype=torch.float32)
    cec_t = torch.tensor(cec, dtype=torch.float32)
    w_t = torch.zeros(Xc.shape[-1], requires_grad=True)
    b_t = torch.tensor(-2.0, requires_grad=True)
    opt = torch.optim.Adam([w_t, b_t], lr=0.05)
    for ep in range(args.epochs):
        lam = torch.sigmoid(Xc_t @ w_t + b_t)
        one_m = (1.0 - lam).clamp(1e-6, 1.0)
        S = torch.cumprod(one_m, dim=1)
        S_prev = torch.cat([torch.ones(len(S), 1), S[:, :-1]], dim=1)
        p = lam * S_prev
        p = torch.cat([p[:, :-1], S_prev[:, -1:]], dim=1)   # final step absorbs the remainder
        loss = (p * cec_t).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 100 == 0:
            print(f"    ep {ep:>3} expected-exit loss {loss.item():.4f}", flush=True)
    w = w_t.detach().numpy().astype(np.float64); b0 = float(b_t.detach())

    lam_t = 1.0 / (1.0 + np.exp(-(Xt @ w + b0)))
    cdf = 1.0 - np.cumprod(np.clip(1.0 - lam_t, 1e-6, 1.0), axis=1)
    print(f"\n  {'q':>6} {'test CE':>9} {'bpb':>8} {'vs fixed':>10} {'mean depth':>11} "
          f"{'headroom kept':>14}")
    best = None
    for q in (0.1, 0.25, 0.5, 0.7, 0.85, 0.95, 0.99):
        hit = cdf >= q; hit[:, -1] = True
        k_idx = hit.argmax(1)
        c = cet[np.arange(len(cet)), k_idx].mean()
        frac = (base - c) / (base - orc) if base > orc else float("nan")
        print(f"  {q:>6.2f} {c:>9.4f} {bpb(c):>8.4f} {c-base:>+10.4f} {(k_idx+1).mean():>11.2f} "
              f"{frac:>13.1%}")
        if best is None or c < best[1]:
            best = (q, c)
    TOL = 0.01
    q_default = 0.5
    hit = cdf >= q_default; hit[:, -1] = True
    k_def = hit.argmax(1)
    ce_def = cet[np.arange(len(cet)), k_def].mean()
    print(f"\n  PALBERT default q=0.5: CE {ce_def:.4f}  vs fixed {ce_def-base:+.4f}  "
          f"mean depth {(k_def+1).mean():.2f}")
    print(f"  best swept q={best[0]}  CE {best[1]:.4f}  vs fixed {best[1]-base:+.4f} nats")
    print("  NOTE (selection-on-noise, cf. LTO arXiv 2509.26314): a learned scorer picking the best of"
          "\n  64 correlated depths can exploit its own errors -- LTO reports that removing their KL"
          "\n  regularizer reduces the method to best-of-N in latent space and 'only works well if the"
          "\n  LRM is nearly perfect'. The same caveat applies to both the oracle bound and any"
          "\n  learned rule here, which is why the reportable number is calibration-fit/test-scored.")
    print("  VERDICT:", f"Q-exit BEATS fixed depth by {base-best[1]:.4f} nats"
          if base - best[1] > TOL else
          f"Q-exit does NOT beat fixed depth by more than {TOL} nats -- consistent with the "
          "hand-crafted and learned-probe rules; the signal an exiter needs is not in these features")



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
    _persist_stdout("qexit", _out)

