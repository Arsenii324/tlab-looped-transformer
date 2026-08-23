"""One implementation of "is this CE curve a result, or a broken vocabulary?"

The check existed inline in `src/eval.py` and worked. It did not stop the failure it was written for:
on 2026-08-23 a one-off analysis script re-evaluated DataSphere checkpoints locally, got best CE
**9.2692** against a chance level of **8.3178**, and wrote a plausible-looking JSON. Nothing raised,
because the script never went through `eval.py`.

**A guard that only fires inside one function is not a guard on the quantity, it is a guard on that
function.** This is the quantity's own check, importable by any script that produces a CE curve.

    from chance_guard import assert_not_chance
    assert_not_chance(ce_by_depth, vocab_size)     # raises on a broken curve
    assert_not_chance(ce, vocab, warn_only=True)   # eval.py's behaviour: print and continue
"""
from __future__ import annotations
import math

def chance_level(vocab_size: int) -> float:
    """CE of a uniform predictor. A vocabulary mismatch lands HERE and does not raise on its own."""
    return math.log(vocab_size)

def assert_not_chance(ce_by_depth: dict, vocab_size: int, *, margin: float = 0.5,
                      warn_only: bool = False, label: str = "") -> dict:
    """Flag any depth whose CE sits at or above chance. Returns the offending entries.

    `margin` is additive above chance: 0.5 nats, chosen so an ordinary bad-but-real model (which sits
    well below chance) never trips it, while a vocabulary mismatch (which sits AT or ABOVE it) always
    does. A curve whose minimum is above chance is not a weak result; it is not a result.
    """
    ch = chance_level(vocab_size)
    bad = {r: v for r, v in ce_by_depth.items() if v > ch + margin}
    if not bad:
        return {}
    msg = (f"CE ABOVE CHANCE ({ch:.4f}) at depths {sorted(bad)}"
           f"{' for ' + label if label else ''}: "
           f"{ {r: round(v, 4) for r, v in sorted(bad.items())[:8]} }. This is not a quality result "
           f"-- it is a vocabulary mismatch, a broken forward pass, or a degenerate checkpoint. "
           f"Verify with src/check_tokenizer_identity.py before reading anything derived from it. "
           f"NOTE: DataSphere jobs here train their own BPE and do NOT return tokenizer.json, so "
           f"their checkpoints cannot be evaluated against the shipped configs/tokenizer.json.")
    if warn_only:
        print(f"  ⚠ {msg}", flush=True)
        return bad
    raise ValueError(msg)
