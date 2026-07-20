"""Executable invariants for the Chapter 5 teaching code.

Imports the tangled module ``code/ch05/_generated.py`` (produced from the
chapter's ``# @save`` cells by ``scripts/tangle.py``) and checks the properties
the chapter claims: that the joint law recovers the exponents of a known
generator and that its compute-optimal split satisfies ``C = 6ND``; that the
residual-bootstrap forecast returns an ordered interval; that the data pipeline
extracts, filters with named reasons, catches a planted near-duplicate via
MinHash/LSH while preserving documents that merely share boilerplate, removes an
evaluation plant, and selects exact source quotas without silent backfill; that
fertility orders a multibyte language above English; and that recursive
self-consumption collapses a distribution while a real anchor preserves it.

The module is loaded under a unique name (``ch05_generated``) rather than the
bare ``sys.path`` pattern because several chapters each ship a ``_generated``
module; a plain import would collide inside one pytest process.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch05_generated", ROOT / "code" / "ch05" / "_generated.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ch05 = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("ch05_generated", ch05)
_SPEC.loader.exec_module(ch05)

ScalingLaw = ch05.ScalingLaw
fit_scaling_law = ch05.fit_scaling_law
extrapolation_interval = ch05.extrapolation_interval
Document = ch05.Document
extract_records = ch05.extract_records
filter_documents = ch05.filter_documents
near_deduplicate = ch05.near_deduplicate
decontaminate = ch05.decontaminate
mix_documents = ch05.mix_documents
measure_fertility = ch05.measure_fertility
recursive_generations = ch05.recursive_generations


def _synthetic_ladder():
    """A ladder generated from a known law so the fit has a ground truth."""
    generator = ScalingLaw(floor=2.0, param_coeff=1.0, data_coeff=1.0,
                           param_exponent=0.34, data_exponent=0.28)
    params, tokens, losses = [], [], []
    rng = np.random.default_rng(0)
    for n in (2.5e7, 1.0e8, 4.0e8, 1.6e9):
        for d in (3.0e8, 1.2e9, 6.0e9, 2.4e10):
            params.append(n)
            tokens.append(d)
            losses.append(float(generator.loss(n, d)) + rng.normal(0, 0.002))
    return params, tokens, losses


def test_fit_recovers_exponents_and_compute_optimum_obeys_6nd() -> None:
    params, tokens, losses = _synthetic_ladder()
    law = fit_scaling_law(params, tokens, losses)
    # A joint fit at small scale is only loosely determined, so we ask for the
    # right ballpark on the exponents and a tight fit on the losses themselves.
    assert abs(law.param_exponent - 0.34) < 0.1
    assert abs(law.data_exponent - 0.28) < 0.1
    predicted = [float(law.loss(n, d)) for n, d in zip(params, tokens)]
    rmse = np.sqrt(np.mean([(p - o) ** 2 for p, o in zip(predicted, losses)]))
    assert rmse < 0.01
    compute = 9.0e20
    opt_n, opt_d, opt_loss = law.compute_optimal(compute)
    assert abs(6 * opt_n * opt_d / compute - 1) < 1e-9
    assert opt_n > 0 and opt_d > 0 and opt_loss > law.floor


def test_scaling_law_loss_falls_with_more_resources() -> None:
    law = ScalingLaw(2.0, 1.0, 1.0, 0.34, 0.28)
    assert law.loss(2e8, 1e9) < law.loss(1e8, 1e9)   # more params helps
    assert law.loss(1e8, 2e9) < law.loss(1e8, 1e9)   # more tokens helps


def test_extrapolation_reports_an_ordered_interval() -> None:
    params, tokens, losses = _synthetic_ladder()
    result = extrapolation_interval(params, tokens, losses, multiplier=100.0, samples=40, seed=1)
    assert result["p05"] <= result["p95"]
    assert result["valid"] >= 20
    assert result["target_compute"] == pytest.approx(
        max(6 * p * d for p, d in zip(params, tokens)) * 100.0
    )
    assert result["opt_params"] > 0 and result["opt_tokens"] > 0


def _raw(url, source, rights, text, lang="en"):
    return {"url": url, "source": source, "rights": rights, "lang": lang, "text": text}


def test_extraction_and_filter_leave_a_reason_ledger() -> None:
    body = "A durable checked record with plain readable prose about surveys and dates. " * 6
    raw = [
        _raw("u://a", "reference", "licensed", body),
        _raw("u://b", "reference", "restricted", body),          # rights
        _raw("u://c", "community", "permission", "Short."),      # too-short
        _raw("u://d", "web", "licensed", "\n".join(["BUY NOW"] * 40)),  # repetition
    ]
    documents = extract_records(raw)
    assert len(documents) == 4
    assert len({d.doc_id for d in documents}) == 4
    kept, rejected = filter_documents(documents)
    reasons = {r["reason"] for r in rejected}
    assert {"rights", "too-short", "repetition"} <= reasons
    assert [d.source for d in kept] == ["reference"]


def test_minhash_catches_near_duplicate_but_keeps_distinct_boilerplate() -> None:
    nav = "Home About Login Cart Help"
    body = "Evidence line {i} describes a measured system and a recorded source."
    shared = "\n".join(body.format(i=i) for i in range(120))
    other = "\n".join(f"Botany record {i} explains roots leaves and seeds." for i in range(120))
    docs = [
        Document("a", "u://a", "reference", "en", "licensed", nav + "\n" + shared),
        Document("b", "u://b", "reference", "en", "licensed", nav + "\n" + shared + "\nA tiny footer edit."),
        Document("c", "u://c", "library", "en", "public-domain", nav + "\n" + other),
    ]
    kept, clusters = near_deduplicate(docs)
    assert len(kept) == 2                                    # a and b collapse
    assert any(len(members) == 2 for members in clusters)
    assert "c" in {d.doc_id for d in kept}                   # distinct doc survives despite shared nav


def test_decontamination_and_mixture_are_exact() -> None:
    body = "A durable checked record with plain readable prose about surveys. " * 8
    docs = [
        Document("r1", "u://r1", "reference", "en", "licensed", body),
        Document("r2", "u://r2", "reference", "en", "licensed", "the benchmark answer is cobalt river " + body),
        Document("r3", "u://r3", "reference", "en", "licensed", body),
        Document("l1", "u://l1", "library", "en", "public-domain", body),
        Document("l2", "u://l2", "library", "en", "public-domain", body),
        Document("c1", "u://c1", "community", "en", "permission", body),
        Document("w1", "u://w1", "web", "en", "licensed", body),
    ]
    clean, removed = decontaminate(docs, ["benchmark answer is cobalt river"])
    assert {r["doc_id"] for r in removed} == {"r2"}
    mixture = mix_documents(clean, {"reference": 0.5, "library": 0.25, "community": 0.25, "web": 0.0}, total=4)
    counts = {}
    for doc in mixture:
        counts[doc.source] = counts.get(doc.source, 0) + 1
    assert counts == {"reference": 2, "library": 1, "community": 1}
    assert all(doc.source != "web" for doc in mixture)


def test_fertility_orders_a_multibyte_language_above_english() -> None:
    byte_encode = lambda text: list(text.encode("utf-8"))
    rows = measure_fertility(
        {"English": "measure tokens before a budget", "Arabic": "قس الرموز قبل الميزانية"},
        byte_encode,
    )
    values = {r["language"]: r["tokens_per_char"] for r in rows}
    assert values["Arabic"] > values["English"]


def test_recursive_replacement_collapses_but_a_real_anchor_preserves() -> None:
    seeds = range(120)
    replace = np.mean([recursive_generations(mode="replace", seed=s)[0] for s in seeds], axis=0)
    accumulate = np.mean([recursive_generations(mode="accumulate", seed=s)[0] for s in seeds], axis=0)
    assert replace[-1] < replace[0] * 0.85          # spread collapses under replacement
    assert accumulate[-1] > replace[-1]             # the real anchor holds more spread
