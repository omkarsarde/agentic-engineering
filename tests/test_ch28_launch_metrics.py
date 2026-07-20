"""Executable invariants for Chapter 28's launch-review artifact.

Imports only the tangled module ``code/ch28/_generated.py`` (the chapter's
``# @save`` cells in document order) under a chapter-unique module name so the
suite never collides with another chapter's generated module.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch28_generated", ROOT / "code" / "ch28" / "_generated.py"
)
ch28 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["ch28_generated"] = ch28  # dataclasses resolve annotations via sys.modules
_SPEC.loader.exec_module(ch28)


def test_logs_expand_to_two_500_row_ledgers() -> None:
    decisions, usage = ch28.expand_logs(ch28.COHORT_LOG)
    assert len(decisions) == 500
    assert len(usage) == 500


def test_reliance_quadrants_are_exhaustive_and_aggregate_looks_fine() -> None:
    overall = ch28.build_launch_review()["reliance"]["overall"]
    assert sum(overall["quadrants"].values()) == overall["n"] == 500
    assert overall["quadrants"] == {
        "correct_accept": 395, "correct_reject": 25, "wrong_accept": 34, "wrong_reject": 46,
    }
    assert overall["appropriate_rate"] == 0.882


def test_new_reviewer_cohort_overreliance_is_hidden_by_the_aggregate() -> None:
    by_cohort = ch28.build_launch_review()["reliance"]["by_cohort"]
    assert by_cohort["new_reviewer"]["overreliance_rate"] == 0.8333
    assert by_cohort["experienced"]["overreliance_rate"] < 0.2
    # the damning cohort rate is far above the aggregate over-rate
    assert by_cohort["new_reviewer"]["overreliance_rate"] > by_cohort["experienced"]["overreliance_rate"]


def test_funnel_is_nested_monotone_and_shows_the_access_slice() -> None:
    funnel = ch28.build_launch_review()["funnel"]
    counts = list(funnel["overall"]["counts"].values())
    assert counts == [500, 460, 420, 380, 345, 295]
    assert all(a >= b for a, b in zip(counts, counts[1:]))  # never increases
    assistive = funnel["by_cohort"]["assistive_technology"]
    assert assistive["conversion_from_previous"]["completed"] == 0.7647


def test_offline_eval_exceeds_realized_end_to_end_value() -> None:
    counts = ch28.build_launch_review()["funnel"]["overall"]["counts"]
    realized = round(counts["outcome_realized"] / counts["exposed"], 4)
    assert realized == 0.69  # far below a representative 0.90 offline eval


def test_three_designs_show_autonomy_raising_cost_per_outcome() -> None:
    rows = {name: ch28.cost_per_outcome(**d) for name, d in ch28.DESIGNS.items()}
    assert rows["deterministic_rules"]["cost_per_outcome"] == 0.26
    assert rows["single_call_classifier"]["cost_per_outcome"] == 0.9767
    assert rows["bounded_agent"]["cost_per_outcome"] == 1.4232
    # every design beats the conventional baseline, yet the agent is dearest per outcome
    assert all(r["cost_per_outcome"] < ch28.BASELINE_COST_PER_OUTCOME for r in rows.values())
    assert rows["bounded_agent"]["cost_per_outcome"] == max(r["cost_per_outcome"] for r in rows.values())


def test_economics_charge_review_labor_and_expose_break_even() -> None:
    econ = ch28.build_launch_review()["economics"]
    assert econ["review_labor"] == 310.0
    assert econ["remediation"] == 136.0
    assert econ["cost_per_outcome"] == 1.4232
    assert econ["cost_per_outcome"] < econ["baseline_cost_per_outcome"]
    assert econ["average_review_seconds"] == 62.0
    assert econ["break_even_review_seconds"] == 74.2
    assert econ["energy_wh_per_outcome"] == 0.2609


def test_gates_are_conjunctive_only_economics_passes() -> None:
    gates = {g["gate"]: g["passed"] for g in ch28.build_launch_review()["gates"]}
    assert gates == {
        "appropriate_reliance": False,
        "cohort_overreliance": False,
        "assistive_completion": False,
        "cost_per_outcome": True,
    }


def test_risk_register_status_is_derived_from_gate_results() -> None:
    report = ch28.build_launch_review()
    entries = report["risk_register"]
    assert len(entries) == 3
    assert all(e.status == "OPEN" for e in entries)
    # a passing economic gate cannot buy back a failing people gate
    assert report["recommendation"] == "HOLD_FULL_LAUNCH"
    assert report["failed_conditions"] == [
        "appropriate_reliance", "cohort_overreliance", "assistive_completion",
    ]


def test_render_appendix_is_the_real_program_output() -> None:
    text = ch28.render_appendix(ch28.build_launch_review())
    assert "HOLD_FULL_LAUNCH" in text
    assert "appropriate_reliance" in text and "FAIL" in text
    assert "PASS" in text  # the cost gate


def test_report_is_reproducible() -> None:
    assert ch28.render_appendix(ch28.build_launch_review()) == ch28.render_appendix(
        ch28.build_launch_review()
    )


def test_trust_objects_carry_labeled_evidence() -> None:
    diff = ch28.PlanDiff(
        affected_object="EXP-1", field="category", from_state="(none)", to_state="Travel",
        permission="write:expense_category",
        evidence=((ch28.EpistemicLabel.UNRESOLVED, "receipt missing"),),
    )
    assert diff.evidence[0][0] is ch28.EpistemicLabel.UNRESOLVED
    receipt = ch28.Receipt("ledger", "op-1", "written", "2026-07-19T00:00:00Z", "confirmed")
    assert receipt.op_id == "op-1"
