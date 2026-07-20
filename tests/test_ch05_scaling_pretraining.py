"""Executable contracts for Chapter 5 scaling and pretraining data."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CH05 = ROOT / "code" / "ch05"
CH02 = ROOT / "code" / "ch02"
sys.path.insert(0, str(CH05))
sys.path.insert(0, str(CH02))

from chapter2_adapter import BytePairTokenizer  # noqa: E402
from mini_pipeline import (  # noqa: E402
    Document,
    decontaminate,
    extract_wet,
    filter_documents,
    measure_fertility,
    mix_documents,
    near_deduplicate,
)
from scaling_fit import (  # noqa: E402
    ScalingObservation,
    extrapolation_interval,
    fit_scaling_law,
    load_observations,
)


_BUILD_SPEC = importlib.util.spec_from_file_location("ch05_run_build", CH05 / "run_build.py")
assert _BUILD_SPEC is not None and _BUILD_SPEC.loader is not None
_BUILD_MODULE = importlib.util.module_from_spec(_BUILD_SPEC)
_BUILD_SPEC.loader.exec_module(_BUILD_MODULE)
run_build = _BUILD_MODULE.run_build
write_synthetic_wet = _BUILD_MODULE.write_synthetic_wet
CLOZE_TASKS = _BUILD_MODULE.CLOZE_TASKS


def test_scaling_fit_recovers_proxy_exponents_and_compute_optimum() -> None:
    observations = load_observations(CH05 / "fixtures" / "proxy_ladder.csv")
    law = fit_scaling_law(observations)
    assert abs(law.parameter_exponent - 0.34) < 0.04
    assert abs(law.data_exponent - 0.28) < 0.04
    compute = 9.0e20
    parameters, tokens, loss = law.compute_optimal(compute)
    assert abs(6 * parameters * tokens / compute - 1) < 1e-10
    assert parameters > 0 and tokens > 0 and loss > law.irreducible_loss


def test_extrapolation_reports_an_ordered_uncertainty_interval() -> None:
    observations = load_observations(CH05 / "fixtures" / "proxy_ladder.csv")
    result = extrapolation_interval(observations, bootstrap_samples=30, seed=3)
    assert result["prediction_p05"] < result["prediction_p95"]
    assert result["fit_p05"] < result["fit_p95"]
    assert result["target_compute_flops"] == max(row.compute_flops for row in observations) * 10
    assert result["valid_bootstrap_fits"] >= 15


def test_scaling_fit_rejects_nonfinite_observations() -> None:
    observations = load_observations(CH05 / "fixtures" / "proxy_ladder.csv")
    observations[0] = ScalingObservation(float("nan"), observations[0].tokens, observations[0].loss)
    with pytest.raises(ValueError, match="finite and positive"):
        fit_scaling_law(observations)


def test_wet_extraction_and_filtering_leave_a_reason_ledger(tmp_path: Path) -> None:
    path = tmp_path / "fixture.wet"
    records = write_synthetic_wet(path, target_bytes=700_000, body_bytes=16_384)
    documents = extract_wet(path)
    accepted, rejected = filter_documents(documents)
    reasons = Counter(row["reason"] for row in rejected)
    assert len(documents) == records
    assert {"reference", "library", "community"} & {document.source for document in accepted}
    assert reasons["rights-policy"] > 0
    assert reasons["line-repetition"] > 0
    assert reasons["too-short"] > 0


def test_minhash_lsh_clusters_near_duplicates_but_preserves_distinct_text() -> None:
    common = "\n".join(
        f"Evidence line {index} describes a measured system and a recorded source."
        for index in range(180)
    )
    documents = [
        Document("a", "https://x/a", "reference", "en", "licensed", common),
        Document("b", "https://x/b", "reference", "en", "licensed", common + "\nA small revision."),
        Document(
            "c",
            "https://x/c",
            "library",
            "en",
            "public-domain",
            "\n".join(f"Botany record {index} explains roots leaves and seeds." for index in range(180)),
        ),
    ]
    unique, audit = near_deduplicate(documents)
    assert len(unique) == 2
    assert any(row["cluster_size"] == 2 for row in audit)
    assert {document.doc_id for document in unique} & {"a", "b"}
    assert "c" in {document.doc_id for document in unique}


def test_bottom_k_shingles_do_not_confuse_shared_boilerplate_with_content() -> None:
    boilerplate = "\n".join(f"Shared navigation item {index} home account help." for index in range(30))
    left = boilerplate + "\n" + "\n".join(
        f"Astronomy evidence {index} discusses stars galaxies and spectra." for index in range(700)
    )
    right = boilerplate + "\n" + "\n".join(
        f"Botany evidence {index} discusses roots leaves and pollen." for index in range(700)
    )
    documents = [
        Document("left", "u:left", "reference", "en", "licensed", left),
        Document("right", "u:right", "reference", "en", "licensed", right),
    ]
    unique, _ = near_deduplicate(documents)
    assert {document.doc_id for document in unique} == {"left", "right"}


def test_decontamination_mixture_and_fertility_are_explicit() -> None:
    documents = [
        Document("r1", "u:r1", "reference", "en", "licensed", "A long clean record. " * 20),
        Document("r2", "u:r2", "reference", "en", "licensed", "The benchmark answer phrase is hidden here. " * 8),
        Document("l1", "u:l1", "library", "en", "public-domain", "A library record with durable prose. " * 20),
        Document("c1", "u:c1", "community", "en", "permission", "A community record with durable prose. " * 20),
        Document("r3", "u:r3", "reference", "en", "licensed", "Another reference record. " * 20),
        Document("w1", "u:w1", "web", "en", "licensed", "A zero weight web record. " * 20),
    ]
    kept, rejected = decontaminate(documents, ["benchmark answer phrase is hidden here"])
    assert {row["doc_id"] for row in rejected} == {"r2"}
    mixture = mix_documents(
        kept, {"reference": 0.5, "library": 0.25, "community": 0.25}, total_docs=4
    )
    assert Counter(document.source for document in mixture) == {
        "reference": 2,
        "library": 1,
        "community": 1,
    }
    assert all(document.source != "web" for document in mixture)
    with pytest.raises(ValueError, match="infeasible source quotas"):
        mix_documents(
            kept, {"reference": 0.8, "library": 0.1, "community": 0.1, "web": 0.0}, total_docs=4
        )
    tokenizer = BytePairTokenizer.train("Engineering evidence and careful budgets. " * 100, 280)
    fertility = measure_fertility(
        {"English": "Measure tokens before setting a budget.", "Arabic": "قس الرموز قبل تحديد الميزانية."},
        tokenizer.encode,
    )
    values = {row["language"]: row["tokens_per_character"] for row in fertility}
    assert values["Arabic"] > values["English"]


def test_integrated_build_emits_evidence_and_cleaned_data_wins(tmp_path: Path) -> None:
    metrics = run_build(
        tmp_path,
        raw_target_bytes=2_200_000,
        train_steps=35,
        bootstrap_samples=30,
        replicate_seeds=(17, 29),
    )
    raw_loss = metrics["toy_pretraining"]["raw"]["heldout_loss"]
    cleaned_loss = metrics["toy_pretraining"]["cleaned"]["heldout_loss"]
    assert cleaned_loss < raw_loss
    paired = metrics["toy_pretraining"]["paired_raw_minus_cleaned"]
    assert paired["cleaned_wins_every_seed"]
    assert paired["p05"] > 0
    cloze = metrics["toy_pretraining"]["cloze_contract"]
    assert cloze["few_shot_examples"] == 2
    assert cloze["constant_position_baseline_accuracy"] == 0.4
    assert len(set(cloze["correct_answer_positions"])) == 3
    assert metrics["pipeline"]["mixture_sources"] == {"reference": 3, "library": 2, "community": 1}
    assert metrics["experiment_contracts"]["pipeline"]["actual_bytes"] >= 2_200_000
    for filename in (
        "metrics.json",
        "scaling-fit.csv",
        "pipeline-stages.csv",
        "dedup-clusters.csv",
        "document-ledger.csv",
        "training-curves.csv",
        "fertility.csv",
        "samples.txt",
        "scaling-law-fit.svg",
        "data-pipeline.svg",
        "data-quality-training.svg",
    ):
        assert (tmp_path / filename).exists()
    assert not (tmp_path / "synthetic-crawl.wet").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["experiment_contracts"]["training"]["paired_seeds"] == [17, 29]
    assert saved["reproducibility"]["input_hashes"]["tokenizer_sha256"]
    ledger = list(csv.DictReader((tmp_path / "document-ledger.csv").open(encoding="utf-8")))
    assert len(ledger) == metrics["pipeline"]["stages"][0]["documents"]
    assert {row["decision"] for row in ledger} >= {"rejected", "selected", "eligible-not-selected"}
