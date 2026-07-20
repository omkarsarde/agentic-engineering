"""Tests for Chapter 9's tangled inference-behavior module.

Imports ``code/ch09/_generated.py`` under a unique module name so it does not
collide with other chapters' generated modules, then exercises the decoding,
determinism, hallucination, calibration, abstention, and streaming machinery.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ch09_generated", ROOT / "code" / "ch09" / "_generated.py"
)
ch09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ch09)


# --- decoding: temperature -------------------------------------------------

def test_softmax_greedy_limit_and_order():
    logits = [3.0, 2.0, 1.0]
    assert ch09.softmax(logits, 0.0) == [1.0, 0.0, 0.0]          # argmax at T=0
    assert ch09.softmax(logits, -1.0) == [1.0, 0.0, 0.0]         # any non-positive T
    p = ch09.softmax(logits, 2.0)
    assert abs(sum(p) - 1.0) < 1e-12
    assert sorted(range(3), key=p.__getitem__, reverse=True) == [0, 1, 2]  # order preserved


def test_softmax_ties_break_by_index():
    assert ch09.softmax([1.0, 1.0, 1.0], 0.0) == [1.0, 0.0, 0.0]


# --- decoding: truncation --------------------------------------------------

def test_truncate_supports_and_nonempty():
    probs = [0.5, 0.25, 0.15, 0.10]
    assert sum(x > 0 for x in ch09.truncate(probs, "top_k", 2)) == 2
    # top-p includes the token that crosses the threshold
    assert sum(x > 0 for x in ch09.truncate(probs, "top_p", 0.6)) == 2
    # min-p keys off the peak; a high alpha keeps only the mode
    assert sum(x > 0 for x in ch09.truncate(probs, "min_p", 0.9)) == 1
    for method, value in [("full", 1.0), ("top_k", 3), ("top_p", 0.8), ("min_p", 0.2), ("typical", 0.8)]:
        out = ch09.truncate(probs, method, value)
        assert abs(sum(out) - 1.0) < 1e-9 and any(x > 0 for x in out)


def test_truncate_unknown_method_raises():
    with pytest.raises(ValueError):
        ch09.truncate([0.5, 0.5], "nope", 1.0)


# --- decoding: processors --------------------------------------------------

def test_process_logits_is_sign_aware():
    # positive seen logit shrinks (÷r); negative seen logit grows more negative (×r)
    assert ch09.process_logits([2.0, -2.0], [1, 1], repetition=2.0) == [1.0, -4.0]
    # unseen token untouched; additive frequency penalty applied
    out = ch09.process_logits([1.0, 1.0], [0, 2], frequency=0.5)
    assert out[0] == 1.0 and out[1] == 0.0


# --- determinism -----------------------------------------------------------

def test_batch_invariance_probe_flips_on_reduction_order():
    probe = ch09.batch_invariance_probe()
    assert probe["greedy_flip"] == 1.0
    assert probe["serial_sum"] == 0.0 and probe["partitioned_sum"] == 1.0
    assert probe["max_drift"] > 0.2


# --- hallucination ---------------------------------------------------------

def test_support_fraction():
    claims = [("a", True), ("b", False), ("c", False), ("d", False)]
    assert ch09.support_fraction(claims) == 0.25
    assert ch09.support_fraction([]) == 0.0


# --- confidence: semantic entropy ------------------------------------------

def test_normalize_collapses_paraphrases():
    assert ch09.normalize_answer("Paris") == ch09.normalize_answer("The city is Paris")
    assert ch09.normalize_answer("paris.") == ch09.normalize_answer("Paris")


def test_semantic_entropy_is_below_surface():
    samples = [("Paris", 0.36), ("The city is Paris", 0.24), ("Lyon", 0.20),
               ("It was Marseille", 0.12), ("paris.", 0.08)]
    out = ch09.semantic_entropy(samples)
    assert out["semantic_nats"] < out["surface_nats"]
    assert len(out["clusters"]) == 3                     # three meaning clusters


# --- confidence: calibration -----------------------------------------------

def test_calibration_reduces_ece():
    fixture = ch09.qa_fixture(seed=0)
    assert len(fixture) == 400
    raw = ch09.ece(fixture, "logprob_conf")
    tau = ch09.fit_temperature(fixture, "logprob_conf")
    for it in fixture:
        it["cal_conf"] = ch09.calibrate(it["logprob_conf"], tau)
    assert ch09.ece(fixture, "cal_conf") < raw
    assert ch09.brier(fixture, "cal_conf") <= ch09.brier(fixture, "logprob_conf")


def test_qa_fixture_is_reproducible():
    assert ch09.qa_fixture(seed=0) == ch09.qa_fixture(seed=0)


# --- abstention: risk-coverage, CRC, split conformal -----------------------

def test_risk_curve_and_relation():
    fixture = ch09.qa_fixture(seed=0)
    curve = ch09.risk_curve(fixture, "logprob_conf")
    for row in curve:
        # marginal error = coverage * selective risk
        assert abs(row["marginal_error"] - row["coverage"] * row["selective_risk"]) < 1e-9
        assert 0.0 <= row["coverage"] <= 1.0


def test_crc_controls_marginal_error():
    fixture = ch09.qa_fixture(seed=0)
    for alpha in (0.10, 0.20):
        t = ch09.crc_threshold(fixture, "logprob_conf", alpha)
        answered = [it for it in fixture if it["logprob_conf"] >= t]
        marginal = sum(1 - it["correct"] for it in answered) / len(fixture)
        assert marginal <= alpha


def test_split_conformal_meets_coverage():
    cal, test = ch09.multiclass_fixture()
    out = ch09.split_conformal(cal, test, 0.10)
    assert out["set_coverage"] >= 0.90              # guaranteed 1 - alpha
    assert 0.0 <= out["singleton_coverage"] <= out["set_coverage"] + 1e-9


# --- streaming -------------------------------------------------------------

def test_stream_incremental_handles_split_multibyte_and_stop():
    # the four UTF-8 bytes of an emoji arrive as separate pieces
    pieces = [b"A", b"\xf0", b"\x9f", b"\x99", b"\x82"]
    assert ch09.stream_incremental(pieces) == "A\U0001f642"
    # a stop straddling two chunks is caught before leaking
    assert ch09.stream_incremental([b"ans", b"wer STO", b"P tail"], "STOP") == "answer "


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
