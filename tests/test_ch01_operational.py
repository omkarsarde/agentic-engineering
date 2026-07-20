"""Executable invariants for the Chapter 1 teaching code.

Imports the tangled module ``code/ch01/_generated.py`` (produced from the
chapter's ``# @save`` cells by ``scripts/tangle.py``) and checks the real
properties the chapter claims: softmax normalizes and reshapes with temperature,
the sampler is seed-deterministic and matches its distribution, BPE round-trips,
and the latency and autonomy arithmetic is correct.
"""

from __future__ import annotations

import math
import random
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch01"))

from _generated import (  # noqa: E402
    Rung,
    autonomy_costs,
    bpe_encode,
    decide_response,
    sample_token,
    simulate_decode,
    softmax,
    token_confidence,
    tpot_ms,
    train_bpe,
)

LOGITS = [1.6, 0.9, 0.3, -0.4]


def test_softmax_normalizes_and_reshapes_with_temperature() -> None:
    for temperature in (0.5, 1.0, 2.0):
        probs = softmax(LOGITS, temperature)
        assert math.isclose(sum(probs), 1.0, rel_tol=1e-9)
        assert all(p > 0 for p in probs)
    # Lower temperature sharpens: the top token holds more mass; entropy is lower.
    top_cold = max(softmax(LOGITS, 0.5))
    top_hot = max(softmax(LOGITS, 2.0))
    assert top_cold > top_hot

    def entropy(temp: float) -> float:
        return -sum(p * math.log2(p) for p in softmax(LOGITS, temp))

    assert entropy(0.5) < entropy(1.0) < entropy(2.0)


def test_sampler_is_seed_deterministic() -> None:
    probs = softmax(LOGITS, 1.0)
    assert sample_token(probs, random.Random(3)) == sample_token(probs, random.Random(3))
    # A different seed is allowed to differ; determinism is per-seed, not global.
    draws = {sample_token(probs, random.Random(s)) for s in range(20)}
    assert len(draws) > 1


def test_sampler_matches_its_distribution() -> None:
    probs = [0.7, 0.2, 0.1]
    counts = Counter(sample_token(probs, random.Random(s)) for s in range(4000))
    empirical = counts[0] / 4000
    assert abs(empirical - 0.7) < 0.05


def test_bpe_roundtrips_and_tracks_familiarity() -> None:
    corpus = [
        "the refund policy covers the standard window",
        "a refund after the window needs an exception",
        "refunds and the refund policy and the window",
    ] * 6
    merges = train_bpe(corpus, num_merges=60)

    # A frequent word collapses to fewer pieces than an unseen string.
    common = bpe_encode("refund", merges)
    rare = bpe_encode("Wagnerslov", merges)
    assert len(common) < len(rare)

    # Encoding is lossless: dropping the end-of-word marker reconstructs the word.
    for word in ("refund", "Wagnerslov", "exception"):
        assert "".join(bpe_encode(word, merges)).replace("</w>", "") == word


def test_token_confidence_is_exponentiation() -> None:
    assert math.isclose(token_confidence(math.log(0.9)), 0.9, rel_tol=1e-9)
    assert token_confidence(-0.03) > token_confidence(-0.5)


def test_tpot_arithmetic() -> None:
    assert tpot_ms([10.0, 30.0, 50.0, 70.0]) == 20.0
    assert tpot_ms([42.0]) == 0.0
    assert tpot_ms([]) == 0.0


def test_decide_response_routes_on_evidence_not_confidence() -> None:
    assert decide_response(0.97, 0.0) == "abstain"   # confident confabulation
    assert decide_response(0.90, 0.9) == "answer"     # grounded and confident
    assert decide_response(0.60, 0.3) == "retrieve"   # weak on both axes


def test_simulate_decode_is_monotonic_and_prefill_scales() -> None:
    times = simulate_decode(prompt_tokens=800, output_tokens=20)
    assert len(times) == 20
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    # A longer prompt pushes the first token later (prefill scales with input).
    short = simulate_decode(prompt_tokens=100, output_tokens=5)
    long = simulate_decode(prompt_tokens=4000, output_tokens=5)
    assert long[0] > short[0]


def test_autonomy_costs_accumulate() -> None:
    ladder = [
        Rung("conventional", 0, 0.0, 0.0, 1),
        Rung("one model call", 1, 0.0, 0.5, 1),
        Rung("bounded agent", 6, 1.0, 3.0, 25),
    ]
    rows = autonomy_costs(ladder)
    names = [name for name, _, _ in rows]
    incrementals = [inc for _, inc, _ in rows]
    cumulatives = [cum for _, _, cum in rows]
    assert names == ["conventional", "one model call", "bounded agent"]
    assert incrementals[0] == 0.0
    assert cumulatives == sorted(cumulatives)  # non-decreasing
    assert math.isclose(cumulatives[-1], sum(incrementals), rel_tol=1e-9)
