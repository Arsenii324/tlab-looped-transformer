"""GATE: does this checkpoint's vocabulary match `configs/tokenizer.json`?

WHY THIS EXISTS. The task statement names the failure directly — a capable agent will
"запросто возьмёт неправильный токенизатор". This project was one step from it: the Kaggle kernel
trains its BPE **fresh from a FineWeb stream** and **never saves it** (it writes only results.json
and the checkpoint), so the vocabulary that produced the headline weights existed solely as a
side-effect of that run. Identity with the local vocab was *inferred from the eval looking coherent*,
which is not a check — a mismatched vocab does not raise, it silently reports CE ≈ ln(V) = 8.32.

THE CHECK. Score the checkpoint on local validation data (tokenized with the local vocab) and compare
against the number the training run itself reported. Agreement to ~1e-3 at loop 1 is only possible if
the vocabularies are the same object; a mismatch lands near ln(vocab_size).

Run before shipping any checkpoint, and before quoting any local number for a remotely-trained one.
    python src/check_tokenizer_identity.py checkpoints/full_control90_kaggle --expect-ce1 3.9192
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys
import numpy as np, torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval import load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--expect-ce1", type=float, required=True,
                    help="CE at loop 1 as reported by the run that PRODUCED the checkpoint")
    ap.add_argument("--val", default=str(ROOT / "data" / "val.bin"))
    ap.add_argument("--tok", default=str(ROOT / "configs" / "tokenizer.json"))
    ap.add_argument("--batches", type=int, default=32)
    a = ap.parse_args()

    tok = json.load(open(a.tok))
    vocab = tok["model"]["vocab"]
    print(f"local tokenizer : {a.tok}  vocab={len(vocab)}")

    p = pathlib.Path(a.ckpt)
    if p.is_dir():
        p = p / "last.pt"
    model, mcfg = load_checkpoint(p, "cpu")[:2]
    model.eval()
    print(f"checkpoint      : {p}  cfg.vocab_size={mcfg.vocab_size}")
    if mcfg.vocab_size != len(vocab):
        print(f"FAIL: vocab size mismatch ({mcfg.vocab_size} vs {len(vocab)})")
        return 1

    val = np.memmap(a.val, dtype=np.uint16, mode="r")
    hi = int(val[:2_000_000].max())
    if hi >= len(vocab):
        print(f"FAIL: val data contains token id {hi} >= vocab {len(vocab)}")
        return 1

    # Two different questions, and they need two different tolerances. An earlier version of this
    # script used a single fixed 0.02 tolerance on 4 batches and FAILED both checkpoints -- 4 batches
    # of 256 tokens is ~1k tokens, whose sampling noise (0.07-0.11 nats here) swamps that tolerance.
    # The gate would have cried wolf on a checkpoint whose vocabulary is provably fine.
    T, rng = 256, np.random.default_rng(0)
    ces = []
    with torch.no_grad():
        for i in rng.integers(0, len(val) - T - 2, size=a.batches):
            xs = val[i:i + T + 1].astype(np.int64)[None, :]
            x, y = torch.from_numpy(xs[:, :-1]), torch.from_numpy(xs[:, 1:])
            lg = model(x, n_loops=1)[0][-1]
            ces.append(torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.size(-1)).float(), y.reshape(-1)).item())
    ce1 = sum(ces) / len(ces)
    sem = (sum((c - ce1) ** 2 for c in ces) / max(1, len(ces) - 1)) ** 0.5 / len(ces) ** 0.5
    chance = math.log(mcfg.vocab_size)
    d = abs(ce1 - a.expect_ce1)

    print(f"CE@1 local      : {ce1:.4f} +- {sem:.4f} (SEM over {len(ces)} batches)")
    print(f"producing run   : {a.expect_ce1:.4f}   |diff| = {d:.4f}")
    print(f"chance level    : {chance:.4f}  (a vocab mismatch lands HERE and does not raise)")

    # [1] THE GATE -- is the vocabulary the same object? A mismatch is not a small discrepancy; it
    #     puts CE at chance. So the decisive comparison is distance-to-expected vs distance-to-chance.
    vocab_ok = d < abs(ce1 - chance) / 3
    print(f"\n[gate] vocabulary identity : {'PASS' if vocab_ok else 'FAIL'} "
          f"(|diff|={d:.4f} vs |CE-chance|/3={abs(ce1-chance)/3:.4f})")

    # [2] SOFT -- do local and producing-run PROTOCOLS agree, within this sample's own noise?
    #     Informational: a fail here means different batches/protocol, not a broken checkpoint.
    k = 3.0
    prot_ok = d <= k * sem + 0.01
    print(f"[soft] protocol agreement  : {'ok' if prot_ok else 'differs'} "
          f"(|diff|={d:.4f} vs {k:.0f}*SEM+0.01={k*sem+0.01:.4f}) "
          f"-- differing here means sampling/protocol, not vocabulary")
    if not vocab_ok:
        print("\nFAIL: treat every local number on this checkpoint as invalid until resolved.")
    return 0 if vocab_ok else 1


if __name__ == "__main__":
    sys.exit(main())
