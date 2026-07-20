"""Executable invariants for Chapter 22's evaluation suite and release gate.

Imports only the tangled module ``code/ch22/_generated.py`` (the chapter's
``# @save`` cells in document order) under a chapter-unique module name, and
drives it against the committed trace and judge-calibration fixtures.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch22_generated", ROOT / "code" / "ch22" / "_generated.py"
)
ch22 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["ch22_generated"] = ch22
_SPEC.loader.exec_module(ch22)

TRACES = ROOT / "code" / "ch22" / "fixtures" / "traces.jsonl"
JUDGE = ROOT / "code" / "ch22" / "fixtures" / "judge_calibration.json"


def test_trace_fixture_expands_to_isolated_trials() -> None:
    rows = ch22.grade_traces(TRACES)
    assert len(rows) == 96
    assert len({row["environment_id"] for row in rows}) == 96


def test_grade_trial_conjunction_blocks_the_unsafe_success() -> None:
    tasks = ch22.load_jsonl(TRACES)
    refund = tasks[0]  # golden refund-en
    unsafe = ch22.grade_trial(refund["runs"]["candidate"][2], refund)
    assert unsafe["state_pass"] is True
    assert unsafe["policy_pass"] is False
    assert unsafe["success"] is False  # one failed component sinks the conjunction


def test_grade_traces_rejects_reused_environment_identity() -> None:
    tasks = ch22.load_jsonl(TRACES)
    corrupt = copy.deepcopy(tasks)
    corrupt[1]["snapshot"] = corrupt[0]["snapshot"]  # collide two tasks' snapshots
    corrupt_path = ROOT / "code" / "ch22" / "fixtures" / "_corrupt_tmp.jsonl"
    import json

    corrupt_path.write_text("\n".join(json.dumps(t) for t in corrupt), encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            ch22.grade_traces(corrupt_path)
    finally:
        corrupt_path.unlink()


def test_cohen_kappa_exposes_the_always_pass_judge() -> None:
    human = ["PASS"] * 90 + ["FAIL"] * 10
    always_pass = ["PASS"] * 100
    assert sum(h == j for h, j in zip(human, always_pass)) / 100 == pytest.approx(0.90)
    assert ch22.cohen_kappa(human, always_pass) == pytest.approx(0.0)


def test_judge_calibration_fixture() -> None:
    report = ch22.judge_report(JUDGE)
    assert report["n"] == 100
    assert report["agreement"] == pytest.approx(0.89)
    assert report["kappa"] == pytest.approx(0.7355769230769231)
    assert report["fail_recall"] == pytest.approx(0.80)
    assert report["position_flip_rate"] == pytest.approx(0.12)


def test_capability_and_reliability_estimators_diverge() -> None:
    assert ch22.pass_at_k_estimate(4, 2, 2) == pytest.approx(5 / 6)
    assert ch22.pass_pow_k_estimate(4, 2, 2) == pytest.approx(1 / 6)
    assert ch22.pass_pow_k_estimate(4, 1, 2) == 0.0
    assert ch22.pass_at_k_estimate(4, 3, 2) == 1.0
    assert ch22.pass_pow_k_estimate(4, 3, 2) == pytest.approx(0.5)


def test_task_cluster_is_the_unit_of_uncertainty() -> None:
    rows = ch22.grade_traces(TRACES)
    baseline = ch22.task_metrics(rows, "baseline")
    candidate = ch22.task_metrics(rows, "candidate")
    result = ch22.paired_cluster_uncertainty(baseline, candidate)
    assert result["point"] == pytest.approx(0.125)
    assert result["low"] == pytest.approx(-0.020833333333333332)
    assert result["high"] == pytest.approx(0.22916666666666666)
    assert result["mde_80"] == pytest.approx(0.18288502039846424)
    assert len(result["task_deltas"]) == 12


def test_primary_failure_categorizes_at_earliest_boundary() -> None:
    rows = ch22.grade_traces(TRACES)
    refund_fails = [
        r for r in rows
        if r["task_id"] == "refund-en" and r["system"] == "candidate" and not r["success"]
    ]
    categories = {ch22.primary_failure(r) for r in refund_fails}
    assert categories == {"policy_violation", "poor_explanation"}
    assert all(ch22.primary_failure(r) is None for r in rows if r["success"])


def test_error_burden_separates_equal_success_unequal_harm() -> None:
    def row(name: str, **flags: bool) -> dict:
        base = {"slice": name, "system": "candidate", "schema_pass": True,
                "state_pass": True, "policy_pass": True, "judge_pass": True}
        base.update(flags)
        base["success"] = all(base[k] for k in
                              ("schema_pass", "state_pass", "policy_pass", "judge_pass"))
        return base

    rows = (
        [row("a") for _ in range(6)] + [row("a", policy_pass=False) for _ in range(2)]
        + [row("b") for _ in range(6)] + [row("b", judge_pass=False) for _ in range(2)]
    )
    burden = ch22.error_burden(rows, "candidate")
    assert burden["a"]["success"] == burden["b"]["success"] == 0.75
    assert burden["a"]["failures"] == {"policy_violation": 2}
    assert burden["b"]["failures"] == {"poor_explanation": 2}


def test_release_report_blocks_the_flattering_candidate() -> None:
    report = ch22.release_report(TRACES, JUDGE)
    assert report["verdict"] == "BLOCK"
    assert "must-not-break regression: refund-en" in report["reasons"]
    assert "paired lower bound exceeds the allowed quality loss" in report["reasons"]
    # every candidate slice clears the floor, yet the release still blocks
    assert min(report["slices"]["candidate"].values()) >= 0.75
    # a favorable process metric does not rescue the verdict
    assert report["trajectory_f1"]["candidate"] > report["trajectory_f1"]["baseline"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
