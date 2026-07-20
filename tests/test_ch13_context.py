"""Executable invariants for the Chapter 13 context-engineering code.

Imports the tangled module ``code/ch13/_generated.py`` (produced from the
chapter's ``# @save`` cells by ``scripts/tangle.py``) and checks the properties
the chapter claims: the stub routes by keyword prior and is steered by the
nearest demonstration; automatic optimization climbs a dev metric while a
held-out split lags (overfitting); rendering is deterministic, trust-labelled,
and cannot silently drop the query; selection defeats a distraction that a naive
preload does not; prefix reuse ends at the first differing byte with the 7.3x
cost multiplier; and compaction classifies durable state before summarizing,
asserting survival, so the 50-turn grid separates the compaction and prefix
axes.

The module is loaded under a unique name (``ch13_generated``) and registered in
``sys.modules`` before execution, because several chapters each ship a module
called ``_generated`` and frozen dataclasses need their module resolvable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch13_generated", ROOT / "code" / "ch13" / "_generated.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ch13 = importlib.util.module_from_spec(_SPEC)
sys.modules["ch13_generated"] = ch13
_SPEC.loader.exec_module(ch13)

BASELINE = ch13.BASELINE
DEV_SET = ch13.DEV_SET
HELDOUT_SET = ch13.HELDOUT_SET
Demo = ch13.Demo
Prompt = ch13.Prompt
Segment = ch13.Segment
classify = ch13.classify
evaluate = ch13.evaluate
optimize_prompt = ch13.optimize_prompt
render_context = ch13.render_context
select_context = ch13.select_context
common_prefix_bytes = ch13.common_prefix_bytes
cache_cost = ch13.cache_cost
compact_history = ch13.compact_history
assert_survival = ch13.assert_survival
synthetic_history = ch13.synthetic_history
simulate_turns = ch13.simulate_turns
token_count = ch13.token_count


def test_stub_uses_keyword_prior_and_precedence() -> None:
    # default precedence checks status before technical, so a crash described
    # with delivery vocabulary is routed to status until precedence is fixed.
    ambiguous = "The tracking page crashes whenever I open it."
    assert classify(BASELINE, ambiguous) == "status"
    fixed = Prompt(BASELINE.instruction, precedence=("refund", "technical", "status"))
    assert classify(fixed, ambiguous) == "technical"


def test_baseline_failure_table_has_three_named_failures() -> None:
    result = evaluate(BASELINE, DEV_SET)
    assert result["score"] == pytest.approx(9 / 12)
    assert len(result["failures"]) == 3
    # every failure row carries what a debugging table needs
    for row in result["failures"]:
        assert set(row) == {"text", "expected", "actual"}
        assert row["expected"] != row["actual"]


def test_demonstration_steers_and_a_corrupted_label_flips() -> None:
    query = "The billing page throws an error when I submit the form."
    clean = Prompt(BASELINE.instruction,
                   demos=(Demo("The billing page shows an error on submit.", "technical"),))
    corrupted = Prompt(BASELINE.instruction,
                       demos=(Demo("The billing page shows an error on submit.", "status"),))
    assert classify(clean, query) == "technical"
    assert classify(corrupted, query) == "status"


def test_optimizer_climbs_dev_but_overfits_heldout() -> None:
    result = optimize_prompt(DEV_SET, HELDOUT_SET)
    traj = result["trajectory"]
    dev = [row["dev_score"] for row in traj]
    held = [row["heldout_score"] for row in traj]
    # trajectory is non-decreasing on the searched split and reaches 100%
    assert dev == sorted(dev)
    assert dev[-1] == pytest.approx(1.0)
    # the held-out split lags — the searched score is not a generalization estimate
    assert held[-1] < dev[-1]
    # both proposal operators were exercised: an instruction edit and a demo
    winner = result["winner"]
    assert "PRECEDENCE" in winner.instruction
    assert len(winner.demos) >= 1


def test_rendering_is_deterministic_stable_first_and_trust_labelled() -> None:
    segments = [
        Segment("query", "user", "answer this"),
        Segment("system", "developer", "follow policy", stable=True),
    ]
    first = render_context(segments, budget=20)
    assert first == render_context(segments, budget=20)
    assert first.index("<system") < first.index("<query")
    assert 'trust="developer"' in first and 'trust="user"' in first


def test_required_segment_raises_rather_than_vanishing() -> None:
    with pytest.raises(ValueError, match="required query"):
        render_context([Segment("query", "user", "one two three four")], budget=2)


def test_selection_defeats_a_distraction_that_preload_does_not() -> None:
    signal = Segment("retrieved", "tool", "order 41 shipped on 2026-07-18 via courier",
                     tags=("order",), priority=8)
    distractors = [Segment("retrieved", "promo", f"promo note {i}") for i in range(6)]
    query = Segment("query", "user", "Where is order 41?")
    system = Segment("system", "developer", "Route by policy.", stable=True)
    everything = [system, *distractors, signal, query]
    preloaded = render_context(everything, budget=49)
    selected = render_context(select_context(everything, "Where is order 41?"), budget=49)
    assert "order 41 shipped" not in preloaded
    assert "order 41 shipped" in selected


def test_confusion_from_a_similar_wrong_label_demo() -> None:
    query = "Where is order 41? It still has not arrived."
    confused = Prompt(BASELINE.instruction,
                      demos=(Demo("Order 41: refund it, it has not arrived.", "refund"),))
    assert classify(confused, query) == "refund"      # the wrong, competing label
    assert classify(BASELINE, query) == "status"      # restored by removing the demo


def test_prefix_reuse_ends_at_first_differing_byte() -> None:
    assert common_prefix_bytes("STATIC\nA", "STATIC\nB") == len("STATIC\n".encode())
    assert common_prefix_bytes("NOW=11\nS", "NOW=12\nS") == len("NOW=1".encode())


def test_cache_cost_multiplier_is_seven_point_three() -> None:
    assert cache_cost(7000, 300)["multiplier"] == pytest.approx(7.3)


def test_compaction_preserves_every_durable_line_and_shrinks() -> None:
    history = synthetic_history()
    compacted, changed = compact_history(history, budget=500)
    joined = "\n".join(compacted)
    assert changed
    assert token_count(joined) < token_count("\n".join(history))
    durable = [m for m in history if m.startswith(("DECISION ", "OPEN ", "ERROR "))]
    assert all(line in joined for line in durable)


def test_survival_check_rejects_a_lost_decision() -> None:
    with pytest.raises(AssertionError, match="lost durable state"):
        assert_survival(["DECISION D1: do not deploy.", "routine"], ["routine"])


def test_fifty_turn_grid_separates_compaction_and_prefix_axes() -> None:
    off_stable = simulate_turns(False, True)
    on_stable = simulate_turns(True, True)
    on_broken = simulate_turns(True, False)
    # compaction bounds utilization; without it the window overflows
    assert max(r["utilization"] for r in off_stable) > 1.0
    assert max(r["utilization"] for r in on_stable) < 0.9
    assert sum(r["compacted"] for r in on_stable) >= 1
    # a stable prefix preserves reuse; a timestamp-first prefix destroys it,
    # independent of compaction being on
    assert sum(r["reuse"] for r in on_stable[1:]) / (len(on_stable) - 1) > 0.9
    assert sum(r["reuse"] for r in on_broken[1:]) / (len(on_broken) - 1) < 0.05
