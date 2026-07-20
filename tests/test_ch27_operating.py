"""Executable invariants for Chapter 27's operations console.

Imports only the tangled module ``code/ch27/_generated.py`` (the chapter's
``# @save`` cells in document order) under a chapter-unique module name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch27_generated", ROOT / "code" / "ch27" / "_generated.py"
)
ch27 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["ch27_generated"] = ch27  # dataclasses resolve annotations via sys.modules
_SPEC.loader.exec_module(ch27)


# --- Tracing ---------------------------------------------------------------

def test_durable_pause_is_two_linked_traces() -> None:
    spans, first_trace = ch27.emit_linked_run()
    assert len(spans) == 5
    assert len({s.context.trace_id for s in spans}) == 2
    assert sum(len(s.links) for s in spans) == 1
    linked = [f"{link.context.trace_id:032x}" for s in spans for link in s.links]
    assert linked == [first_trace]


def test_trace_cost_is_diagnostic_only() -> None:
    spans, _ = ch27.emit_linked_run()
    assert ch27.trace_cost(spans) == 0.013


def test_replay_suppresses_writes_and_replays_reads() -> None:
    recorded = {"lookup_order": {"order_id": "A-17", "paid_cents": 4999}}
    log: list = []
    assert ch27.replay_execute("lookup_order", {}, False, recorded, log) == recorded["lookup_order"]
    out = ch27.replay_execute("refund_order", {"amount_cents": 4999}, True, recorded, log)
    assert out["effect_suppressed"] is True
    assert log == []  # diagnosing a run repeated no effect


# --- SLIs ------------------------------------------------------------------

def test_slis_use_grounded_outcomes_not_http_status() -> None:
    slis = ch27.compute_slis(ch27.fixture_records())
    assert slis["success_and_grounded_rate"] == 0.5
    assert slis["exactly_one_effect_rate"] == 0.75
    assert slis["p95_ttft_ms"] == 400
    assert slis["cost_per_successful_task"] == 0.028


def test_cost_per_task_divides_by_successes_not_all_runs() -> None:
    records = [
        ch27.RunRecord("1", "a", True, True, 1, 100, 1.0, 1.0),
        ch27.RunRecord("2", "a", True, False, 2, 200, 2.0, 0.0),
    ]
    slis = ch27.compute_slis(records)
    assert slis["cost_per_successful_task"] == 3.0  # total 3.0 over one success


def test_nearest_rank_percentile() -> None:
    assert ch27.nearest_rank([1, 2, 3, 4], 0.95) == 4
    assert ch27.nearest_rank([5], 0.5) == 5


# --- Error budget ----------------------------------------------------------

def test_burn_rate_and_exhaustion() -> None:
    burn = ch27.burn_rate(9_655, 10_000, 0.995)
    assert round(burn, 1) == 6.9
    assert round(ch27.days_to_exhaustion(30, 0.95, burn), 1) == 4.1
    assert ch27.days_to_exhaustion(30, 0.95, 0.0) == float("inf")


def test_autonomy_ladder_and_safety_is_not_budgeted() -> None:
    assert ch27.autonomy_action(0.2) == "normal"
    assert ch27.autonomy_action(2.0) == "open_ticket"
    assert ch27.autonomy_action(6.9) == "require_review_and_page"
    assert ch27.autonomy_action(20.0) == "read_only_and_freeze_rollout"
    assert ch27.autonomy_action(0.1, safety_violation=True) == "stop_new_effects"


def test_multiwindow_needs_both_windows() -> None:
    assert ch27.multiwindow_page(20.0, 20.0) is True
    assert ch27.multiwindow_page(20.0, 2.0) is False   # long high, short recovered
    assert ch27.multiwindow_page(2.0, 20.0) is False   # short spike, long not yet


def test_multiwindow_combined_is_precise() -> None:
    series = [0.12 if 20 <= m < 80 else 0.001 for m in range(130)]

    def burn(end: int, window: int) -> float:
        return ch27.window_burn(series[max(0, end - window + 1): end + 1], 0.995)

    combined = [m for m in range(130) if ch27.multiwindow_page(burn(m, 30), burn(m, 5))]
    short = [m for m in range(130) if burn(m, 5) >= 14.4]
    assert short[0] < combined[0]      # short leads
    assert combined[-1] < 82           # combined self-resolves near the revert


# --- Drift and canary ------------------------------------------------------

def test_drift_by_tenant_exposes_hidden_slice() -> None:
    baseline = [ch27.RunRecord("b1", "A", True, True, 1, 10, 1, 0.9),
                ch27.RunRecord("b2", "B", True, True, 1, 10, 1, 0.9)]
    current = [ch27.RunRecord("c1", "A", True, True, 1, 10, 1, 1.0),
               ch27.RunRecord("c2", "B", False, False, 0, 10, 1, 0.4)]
    assert ch27.drift_by_tenant(baseline, current, 0.2) == {"B": 0.5}


def test_streaming_drift_delay_measures_window_lag() -> None:
    stream = [0.9] * 30 + [0.6] * 60
    fired = ch27.streaming_drift_delay(0.9, stream, window=20, threshold=0.15)
    assert fired == 40  # ten samples after the shift at 30


def test_canary_decision_ships_holds_and_rolls_back() -> None:
    assert ch27.canary_decision(970, 1000, 900, 1000)["decision"] == "rollback"
    assert ch27.canary_decision(970, 1000, 965, 1000)["decision"] == "ship"
    assert ch27.canary_decision(970, 1000, 24, 25)["decision"] == "hold"


# --- Chaos and incidents ---------------------------------------------------

def test_chaos_drill_exactly_once_depends_on_idempotency_key() -> None:
    assert ch27.run_chaos_drill(True) == 1     # invariant holds
    assert ch27.run_chaos_drill(False) == 2    # duplicate effect


def test_apply_effect_once_dedupes_on_key() -> None:
    store: dict = {}
    first, created = ch27.apply_effect_once(store, "k", {"v": 1})
    second, again = ch27.apply_effect_once(store, "k", {"v": 2})
    assert created is True and again is False
    assert second == {"v": 1} and len(store) == 1


def test_incident_timeline_mttd_from_first_harm() -> None:
    events = [
        {"t": 0, "kind": "first_harmful"}, {"t": 47, "kind": "detected"},
        {"t": 52, "kind": "contained"}, {"t": 55, "kind": "last_harmful_effect"},
        {"t": 70, "kind": "technical_recovered"}, {"t": 190, "kind": "business_recovered"},
    ]
    tl = ch27.incident_timeline(events)
    assert tl["mttd_min"] == 47
    assert tl["business_recovery_min"] > tl["technical_recovery_min"]


def test_regression_case_wires_to_both_gates() -> None:
    case = ch27.as_regression_test("INC-1", "no bypass", "replay + assert")
    assert case["case_id"] == "regression::INC-1"
    assert case["gate"] == "release + chaos"


# --- HITL and privacy ------------------------------------------------------

def test_hitl_flags_rubber_stamp_but_not_a_healthy_queue() -> None:
    rubber = [ch27.ApprovalRecord(f"a{i}", "approve", 0.2) for i in range(99)] + \
             [ch27.ApprovalRecord("d", "deny", 0.3)]
    healthy = ([ch27.ApprovalRecord(f"a{i}", "approve", 8.0) for i in range(60)] +
               [ch27.ApprovalRecord(f"d{i}", "deny", 12.0) for i in range(40)])
    assert ch27.hitl_metrics(rubber)["rubber_stamp_signal"] is True
    assert ch27.hitl_metrics(rubber)["approval_rate"] == 0.99
    assert ch27.hitl_metrics(healthy)["rubber_stamp_signal"] is False


def test_littles_law_and_stability() -> None:
    assert ch27.queue_length(40, 0.25) == 10.0
    assert ch27.queue_stable(40, 12, 0.25) is True
    assert ch27.queue_stable(40, 9, 0.25) is False


def test_sanitizer_redacts_secrets_but_keeps_token_counts() -> None:
    attrs = {
        "authorization": "Bearer abc.secret",
        "app.tenant.id": "tenant-raw",
        "gen_ai.input.messages": "private prompt",
        "gen_ai.usage.input_tokens": 20000,
        "error.message": "request used Bearer xyz.123 to call the tool",
    }
    cleaned = ch27.sanitize_attributes(attrs)
    assert cleaned["authorization"] == "[REDACTED]"
    assert cleaned["app.tenant.id"] != "tenant-raw" and len(cleaned["app.tenant.id"]) == 12
    assert cleaned["gen_ai.input.messages"] == "[CONTENT_DISABLED]"
    assert cleaned["gen_ai.usage.input_tokens"] == 20000  # the false-positive we fixed
    assert "xyz.123" not in cleaned["error.message"]


def test_sanitizer_captures_content_when_enabled() -> None:
    cleaned = ch27.sanitize_attributes({"gen_ai.input.messages": "hi"}, capture_content=True)
    assert cleaned["gen_ai.input.messages"] == "hi"


def test_pet_numbers() -> None:
    dp = ch27.dp_epsilon([0.5] * 5)
    assert dp["total_epsilon"] == 2.5
    assert round(dp["max_likelihood_ratio"], 1) == 12.2
    assert ch27.federated_average([(0.9, 100), (0.4, 300)]) == 0.525


# --- Integration -----------------------------------------------------------

def test_operator_snapshot_is_self_contained() -> None:
    snap = ch27.operator_snapshot()
    assert snap["spans"] == 5 and snap["fleet_burn"] == 6.9
    assert snap["runtime_action"] == "require_review_and_page"
    assert snap["canary"] == "rollback"
    assert snap["chaos_exactly_once"] is True
    assert snap["rubber_stamp"] is True
