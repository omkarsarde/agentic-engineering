"""Executable invariants for Chapter 16's agent loop, tools, and patterns.

Imports only the tangled module ``code/ch16/_generated.py`` (the chapter's
``# @save`` cells in document order) under a chapter-unique module name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch16_generated", ROOT / "code" / "ch16" / "_generated.py"
)
ch16 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["ch16_generated"] = ch16  # dataclasses resolve annotations via sys.modules
_SPEC.loader.exec_module(ch16)

RECEIPT = {"order_id": "A-17", "refunded_cents": 4999}
TASK = "Order A-17 arrived damaged. Please make this right."


def test_validate_call_separates_wellformed_from_broken() -> None:
    tools, _ = ch16.make_environment()
    good = ch16.ToolCall("c1", "refund_order", {"order_id": "A-17", "amount_cents": 4999})
    bad_type = ch16.ToolCall("c2", "refund_order", {"order_id": "A-17", "amount_cents": "lots"})
    bad_name = ch16.ToolCall("c3", "delete_database", {})
    unparsed = ch16.ToolCall("c4", "refund_order", None)
    assert ch16.validate_call(good, tools) is None
    assert "must be int" in ch16.validate_call(bad_type, tools)
    assert "unknown tool" in ch16.validate_call(bad_name, tools)
    assert "not valid JSON" in ch16.validate_call(unparsed, tools)


def test_tool_schema_json_advertises_contracts_as_data() -> None:
    tools, _ = ch16.make_environment()
    declarations = {d["function"]["name"]: d for d in ch16.tool_schema_json(tools)}
    refund = declarations["refund_order"]["function"]["parameters"]
    assert refund["properties"]["amount_cents"] == {"type": "integer"}
    assert refund["required"] == ["amount_cents", "order_id"]


def test_parse_tool_calls_roundtrip_and_malformed_arguments() -> None:
    message = ch16.tool_call_message("call_7", "refund_order",
                                     {"order_id": "A-17", "amount_cents": 4999})
    (call,) = ch16.parse_tool_calls(message)
    assert call == ch16.ToolCall("call_7", "refund_order",
                                 {"order_id": "A-17", "amount_cents": 4999})
    broken = {"role": "assistant", "content": None,
              "tool_calls": [{"id": "call_8", "type": "function",
                              "function": {"name": "refund_order",
                                           "arguments": '{"order_id": A-17}'}}]}
    (bad,) = ch16.parse_tool_calls(broken)
    assert bad.arguments is None
    assert ch16.parse_tool_calls(ch16.answer_message("done")) == []


def test_execute_call_returns_typed_failure_kinds() -> None:
    tools, effects = ch16.make_environment()
    ok = ch16.execute_call(
        ch16.ToolCall("c1", "lookup_order", {"order_id": "A-17"}), tools, ch16.allow_all)
    missing = ch16.execute_call(
        ch16.ToolCall("c2", "lookup_order", {"order_id": "Z-99"}), tools, ch16.allow_all)
    unknown = ch16.execute_call(
        ch16.ToolCall("c3", "delete_database", {}), tools, ch16.allow_all)
    over = ch16.execute_call(
        ch16.ToolCall("c4", "refund_order", {"order_id": "A-17", "amount_cents": 9999}),
        tools, ch16.refund_gate)
    assert (ok.kind, missing.kind, unknown.kind, over.kind) == (
        "result", "error", "invalid", "denied")
    assert effects == []  # only the allowed lookup ran; nothing effectful did


def test_missing_gate_lets_the_overrefund_through() -> None:
    tools, effects = ch16.make_environment()
    result = ch16.run_agent(TASK, ch16.overreaching_model(), tools, ch16.allow_all)
    assert result.stop == ch16.Stop.ANSWERED
    assert effects == [{"order_id": "A-17", "refunded_cents": 9999}]


def test_gate_denial_is_information_and_the_model_recovers() -> None:
    tools, effects = ch16.make_environment()
    result = ch16.run_agent(TASK, ch16.overreaching_model(), tools, ch16.refund_gate)
    assert result.stop == ch16.Stop.ANSWERED
    assert effects == [RECEIPT]
    assert any(o.kind == "denied" for o in result.state.observations)


def test_denied_effect_never_reaches_handler() -> None:
    tools, effects = ch16.make_environment()
    insist = ch16.tool_call_message("call_s", "refund_order",
                                    {"order_id": "A-17", "amount_cents": 9999})
    stubborn = ch16.ScriptedModel(turns=[insist], reactions={"denied": insist})
    result = ch16.run_agent(TASK, stubborn, tools, ch16.refund_gate)
    assert result.stop == ch16.Stop.NO_PROGRESS
    assert effects == []


def test_step_budget_terminates_a_wanderer() -> None:
    tools, _ = ch16.make_environment()
    look = ch16.tool_call_message("call_w", "lookup_order", {"order_id": "A-17"})
    wanderer = ch16.ScriptedModel(turns=[look], reactions={"order_id": look})
    result = ch16.run_agent("Investigate.", wanderer, tools, ch16.refund_gate,
                            ch16.Limits(max_turns=3))
    assert result.stop == ch16.Stop.STEP_LIMIT
    assert result.state.turns == 3


def test_token_budget_refuses_to_admit_another_call() -> None:
    tools, _ = ch16.make_environment()
    look = ch16.tool_call_message("call_w", "lookup_order", {"order_id": "A-17"})
    wanderer = ch16.ScriptedModel(turns=[look], reactions={"order_id": look})
    result = ch16.run_agent("Investigate.", wanderer, tools, ch16.refund_gate,
                            ch16.Limits(max_turns=40, max_prompt_tokens=1200))
    assert result.stop == ch16.Stop.TOKEN_LIMIT
    assert max(result.state.prompt_token_log) <= 1200
    # the log only ever grows: the transcript is resent every turn
    assert result.state.prompt_token_log == sorted(result.state.prompt_token_log)


def test_check_run_invariants_hold_on_success_and_no_progress() -> None:
    tools, effects = ch16.make_environment()
    result = ch16.run_agent(TASK, ch16.overreaching_model(), tools, ch16.refund_gate)
    assert all(ch16.check_run(result, effects, ch16.Limits()).values())

    tools2, effects2 = ch16.make_environment()
    insist = ch16.tool_call_message("call_s", "refund_order",
                                    {"order_id": "A-17", "amount_cents": 9999})
    stubborn = ch16.ScriptedModel(turns=[insist], reactions={"denied": insist})
    stalled = ch16.run_agent(TASK, stubborn, tools2, ch16.refund_gate)
    assert all(ch16.check_run(stalled, effects2, ch16.Limits()).values())


def test_answered_is_not_success_for_a_liar() -> None:
    tools, effects = ch16.make_environment()
    liar = ch16.ScriptedModel(turns=[
        ch16.answer_message("Good news - your refund for order A-17 is complete!"),
    ])
    result = ch16.run_agent(TASK, liar, tools, ch16.refund_gate)
    assert result.stop == ch16.Stop.ANSWERED
    assert not ch16.verify_refund(effects, "A-17", 4999)


def test_workflow_patterns_share_outcome_but_not_cost() -> None:
    def drafter() -> ch16.ScriptedModel:
        return ch16.ScriptedModel(turns=[
            ch16.answer_message("We refunded 4999 cents for damaged order A-17."),
        ])

    tools, fx = ch16.make_environment()
    chain_meter = ch16.Meter()
    ch16.run_chain(drafter(), tools, ch16.refund_gate, chain_meter)
    assert fx == [RECEIPT] and chain_meter.model_calls == 1

    tools, fx = ch16.make_environment()
    par_meter = ch16.Meter()
    ch16.run_parallel(drafter(), tools, ch16.refund_gate, par_meter)
    assert fx == [RECEIPT] and par_meter.tool_calls == 3

    tools, fx = ch16.make_environment()
    ev_meter = ch16.Meter()
    sloppy = ch16.ScriptedModel(turns=[
        ch16.answer_message("We are sorry - your refund is on its way."),
        ch16.answer_message("We refunded 4999 cents for damaged order A-17."),
    ])
    ch16.run_evaluator(sloppy, tools, ch16.refund_gate, ev_meter)
    assert fx == [RECEIPT] and ev_meter.model_calls == 2  # one rejected round


def test_router_actually_routes_between_branches() -> None:
    tools, fx = ch16.make_environment()
    billing = ch16.ScriptedModel(turns=[
        ch16.answer_message("billing"),
        ch16.answer_message("We refunded 4999 cents for A-17."),
    ])
    ch16.run_router(TASK, billing, tools, ch16.refund_gate, ch16.Meter())
    assert fx == [RECEIPT]

    tools, fx = ch16.make_environment()
    policy = ch16.ScriptedModel(turns=[ch16.answer_message("policy")])
    reply = ch16.run_router("What is your refund policy?", policy, tools,
                            ch16.refund_gate, ch16.Meter())
    assert fx == []  # the policy branch never touches the refund tool
    assert "5000 cents" in reply


def test_plan_is_not_an_execution_grant() -> None:
    tools, effects = ch16.make_environment()
    plan = [
        {"tool": "lookup_order", "arguments": {"order_id": "A-17"}},
        {"tool": "refund_order", "arguments": {"order_id": "A-17", "amount_cents": 9999}},
    ]
    observations, failed_at = ch16.execute_plan(plan, tools, ch16.refund_gate)
    assert failed_at == 1
    assert observations[1].kind == "denied"
    assert effects == []
    replan = [{"tool": "refund_order",
               "arguments": {"order_id": "A-17", "amount_cents": 4999}}]
    _, done = ch16.execute_plan(replan, tools, ch16.refund_gate)
    assert done == 1
    assert effects == [RECEIPT]
