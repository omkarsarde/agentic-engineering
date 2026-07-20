"""Executable invariants for the Chapter 12 teaching code.

Imports the tangled module ``code/ch12/_generated.py`` (produced from the
chapter's ``# @save`` cells by ``scripts/tangle.py``) and checks the real
properties the chapter claims about the API boundary: that ``chat`` posts the
OpenAI-compatible wire shape through an injected transport; that the scripted
stub replays turns honestly; that nucleus sampling keeps a shape-dependent set;
that the constraint mask renormalizes and its tax is the diverted mass; that the
bounded repair loop reaches application validity, returns a typed failure on
exhaustion, and never accepts a business-invalid record; that the validity
experiment reproduces the 76/92/100 ladder with five schema-valid-but-nonexistent
orders; that a single tool turn correlates by id, rejects bad arguments before
the handler, and demands every id be answered once; and that a streamed proposal
is inert until both close and terminal events arrive.

The module is loaded under a unique name (``ch12_generated``) because several
chapters each ship a module called ``_generated``; a plain import would collide
inside one pytest process.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch12_generated", ROOT / "code" / "ch12" / "_generated.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ch12 = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("ch12_generated", ch12)
_SPEC.loader.exec_module(ch12)

chat = ch12.chat
ScriptedModel = ch12.ScriptedModel
nucleus = ch12.nucleus
constrained_renormalize = ch12.constrained_renormalize
constraint_tax = ch12.constraint_tax
extract = ch12.extract
Extraction = ch12.Extraction
StructuredOutputFailed = ch12.StructuredOutputFailed
IncompleteToolStream = ch12.IncompleteToolStream
assemble_tool_stream = ch12.assemble_tool_stream
assert_all_calls_answered = ch12.assert_all_calls_answered
run_tool_turn = ch12.run_tool_turn
classify_failure = ch12.classify_failure
count_tokens = ch12.count_tokens
render_chat_template = ch12.render_chat_template
schema_response_format = ch12.schema_response_format
REFUND_SCHEMA = ch12.REFUND_SCHEMA
make_tickets = ch12.make_tickets
scripted_candidate = ch12.scripted_candidate
semantic_check = ch12.semantic_check
run_validity_experiment = ch12.run_validity_experiment


LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_order",
        "description": "Read one order by exact id.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "pattern": "^O-[0-9]{4}$"}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
}


def _stub(turns):
    return ScriptedModel(turns)


def test_chat_posts_openai_compatible_shape_through_the_transport() -> None:
    seen: dict = {}

    def transport(url: str, body: dict) -> dict:
        seen.update(url=url, body=body)
        return {
            "id": "req_1",
            "choices": [{"message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"total_tokens": 8},
        }

    result = chat("http://localhost:8000", "m",
                  [{"role": "user", "content": "hi"}], transport=transport)
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["body"]["messages"][0]["role"] == "user"
    assert seen["body"]["temperature"] == 0.0
    assert result["finish_reason"] == "stop"
    assert result["request_id"] == "req_1"


def test_scripted_model_replays_turns_and_records_the_body() -> None:
    stub = _stub([({"role": "assistant", "content": "a"}, "stop"),
                  ({"role": "assistant", "content": "b"}, "stop")])
    first = chat("http://stub", "m", [{"role": "user", "content": "x"}], transport=stub)
    second = chat("http://stub", "m", [{"role": "user", "content": "y"}], transport=stub)
    assert first["message"]["content"] == "a"
    assert second["message"]["content"] == "b"
    # last turn repeats once exhausted
    third = chat("http://stub", "m", [{"role": "user", "content": "z"}], transport=stub)
    assert third["message"]["content"] == "b"
    assert stub.last_body["messages"][0]["content"] == "z"


def test_tool_and_schema_fields_only_appear_when_supplied() -> None:
    stub = _stub([({"role": "assistant", "content": "ok"}, "stop")])
    chat("http://stub", "m", [{"role": "user", "content": "x"}], transport=stub)
    assert "tools" not in stub.last_body and "response_format" not in stub.last_body
    chat("http://stub", "m", [{"role": "user", "content": "x"}], transport=stub,
         tools=[LOOKUP_TOOL], response_format=schema_response_format("s", REFUND_SCHEMA))
    assert stub.last_body["tools"] == [LOOKUP_TOOL]
    assert stub.last_body["response_format"]["json_schema"]["strict"] is True


def test_nucleus_keeps_a_shape_dependent_set() -> None:
    peaked = [0.60, 0.25, 0.10, 0.03, 0.02]
    flat = [1 / 12] * 12
    assert nucleus(peaked, 0.7) == [0, 1]           # two tokens reach 0.85 >= 0.7
    assert len(nucleus(flat, 0.7)) == 9             # nine of twelve reach 0.75 >= 0.7
    assert nucleus(peaked, 0.5) == [0]              # one token already covers 0.6


def test_constraint_mask_renormalizes_and_tax_is_diverted_mass() -> None:
    probs = [0.45, 0.08, 0.25, 0.12, 0.10]
    legal = {1}
    after = constrained_renormalize(probs, legal)
    assert after[1] == pytest.approx(1.0)
    assert sum(after) == pytest.approx(1.0)
    assert constraint_tax(probs, legal) == pytest.approx(0.45)  # top token 0 is illegal
    assert constraint_tax(probs, {0}) == 0.0                    # top token is legal
    with pytest.raises(ValueError):
        constrained_renormalize(probs, set())                   # empty support


def test_bounded_repair_reaches_application_validity() -> None:
    good = {"ticket_id": "T-011", "action": "refund", "order_id": "O-0011",
            "amount_cents": 1307, "currency": "USD"}
    bad = {**good, "note": "stray"}
    stub = _stub([({"role": "assistant", "content": json.dumps(bad)}, "stop"),
                  ({"role": "assistant", "content": json.dumps(good)}, "stop")])
    result = extract(partial(chat, "http://stub", "m", transport=stub),
                     [{"role": "user", "content": "extract"}], REFUND_SCHEMA,
                     semantic_check=semantic_check)
    assert isinstance(result, Extraction)
    assert result.attempts == 2
    assert result.value == good
    assert result.total_tokens > 0


def test_bounded_repair_returns_typed_failure_on_exhaustion() -> None:
    stub = _stub([({"role": "assistant", "content": "not json"}, "stop")])
    with pytest.raises(StructuredOutputFailed):
        extract(partial(chat, "http://stub", "m", transport=stub),
                [{"role": "user", "content": "extract"}], REFUND_SCHEMA, max_attempts=2)


def test_repair_never_accepts_a_business_invalid_record() -> None:
    # schema-perfect but nonexistent order; every attempt repeats it -> must fail
    nonexistent = {"ticket_id": "T-010", "action": "refund", "order_id": "O-9010",
                   "amount_cents": 1270, "currency": "USD"}
    stub = _stub([({"role": "assistant", "content": json.dumps(nonexistent)}, "stop")])
    with pytest.raises(StructuredOutputFailed):
        extract(partial(chat, "http://stub", "m", transport=stub),
                [{"role": "user", "content": "extract"}], REFUND_SCHEMA,
                max_attempts=2, semantic_check=semantic_check)


def test_validity_experiment_reproduces_the_ladder() -> None:
    report = run_validity_experiment()
    assert report["validity"][1] == pytest.approx(0.76)
    assert report["validity"][2] == pytest.approx(0.92)
    assert report["validity"][3] == pytest.approx(1.0)
    assert report["business_invalid"] == 5
    assert report["repair_rate"] == pytest.approx(0.08)
    assert report["tokens_per_success"] > 0


def test_prose_wrapped_level1_object_still_parses_but_lowercase_enum_fails() -> None:
    tickets = make_tickets()
    # ticket index 7 is wrapped in prose but otherwise valid -> schema-valid
    wrapped = scripted_candidate(tickets[6], level=1)
    assert wrapped.startswith("Here is the extraction:")
    validator = ch12.Draft202012Validator(REFUND_SCHEMA)
    assert not list(validator.iter_errors(ch12.parse_object(wrapped)))
    # ticket index 4 lower-cases the currency -> schema-invalid
    lowered = scripted_candidate(tickets[3], level=1)
    assert list(validator.iter_errors(ch12.parse_object(lowered)))


def test_single_tool_turn_correlates_by_id() -> None:
    proposal = {"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_9", "type": "function",
        "function": {"name": "lookup_order", "arguments": json.dumps({"order_id": "O-0009"})}}]}
    stub = _stub([(proposal, "tool_calls"),
                  ({"role": "assistant", "content": "shipped"}, "stop")])
    result = run_tool_turn(partial(chat, "http://stub", "m", transport=stub),
                           [{"role": "user", "content": "status?"}], LOOKUP_TOOL,
                           lambda order_id: {"order_id": order_id, "status": "shipped"})
    assert result["finish_reason"] == "stop"
    assert result["message"]["content"] == "shipped"


def test_invalid_tool_arguments_never_reach_the_handler() -> None:
    called = False

    def handler(order_id):
        nonlocal called
        called = True
        return {"order_id": order_id}

    proposal = {"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_bad", "type": "function",
        "function": {"name": "lookup_order", "arguments": json.dumps({"order_id": 9})}}]}
    stub = _stub([(proposal, "tool_calls")])
    with pytest.raises(ValueError, match="tool arguments failed validation"):
        run_tool_turn(partial(chat, "http://stub", "m", transport=stub),
                      [{"role": "user", "content": "status?"}], LOOKUP_TOOL, handler)
    assert not called


def test_every_call_id_must_be_answered_exactly_once() -> None:
    calls = [{"id": "call_1"}, {"id": "call_2"}]
    with pytest.raises(ValueError):
        assert_all_calls_answered(calls, [{"role": "tool", "tool_call_id": "call_1"}])
    with pytest.raises(ValueError):
        assert_all_calls_answered(
            calls, [{"role": "tool", "tool_call_id": "call_1"},
                    {"role": "tool", "tool_call_id": "call_1"}])
    assert_all_calls_answered(
        calls, [{"role": "tool", "tool_call_id": "call_2"},
                {"role": "tool", "tool_call_id": "call_1"}])  # order-insensitive


def test_stream_is_inert_until_close_and_terminal_events() -> None:
    chunks = [
        {"tool_call": {"index": 0, "id": "c1", "name": "lookup_order", "arguments": '{"order_'}},
        {"tool_call": {"index": 0, "arguments": 'id":"O-0001"}', "closed": True}},
    ]
    with pytest.raises(IncompleteToolStream):
        assemble_tool_stream(chunks)                       # closed but not terminal
    with pytest.raises(IncompleteToolStream):
        assemble_tool_stream([chunks[0], {"terminal": True}])  # terminal but unclosed
    done = assemble_tool_stream([*chunks, {"terminal": True}])
    assert done[0]["arguments"] == {"order_id": "O-0001"}
    assert done[0]["id"] == "c1"


def test_two_stream_indices_are_correlated_by_id() -> None:
    chunks = [
        {"tool_call": {"index": 0, "id": "c1", "name": "lookup_order",
                       "arguments": '{"order_id":"O-0001"}', "closed": True}},
        {"tool_call": {"index": 1, "id": "c2", "name": "lookup_order",
                       "arguments": '{"order_id":"O-0002"}', "closed": True}},
        {"terminal": True},
    ]
    proposals = assemble_tool_stream(chunks)
    assert [p["id"] for p in proposals] == ["c1", "c2"]
    assert proposals[1]["arguments"] == {"order_id": "O-0002"}


def test_classify_failure_routes_by_symptom_class() -> None:
    class TimeoutException(Exception):
        pass

    assert classify_failure(error=TimeoutException()) == "code_retry"
    assert classify_failure(error=ValueError()) == "typed_failure"
    assert classify_failure(status=429) == "code_retry"
    assert classify_failure(status=400) == "typed_failure"
    assert classify_failure(finish_reason="length") == "model_retry_without_effect"
    assert classify_failure(finish_reason="content_filter") == "typed_refusal"
    assert classify_failure(finish_reason="stop") == "accept"


def test_template_choice_changes_the_token_count() -> None:
    convo = [{"role": "user", "content": "Where is O-0042?"}]
    marker_a = {"roles": {"user": {"open": "<|u|>", "close": "<|/u|>"}},
                "sep": "", "assistant_open": "<|a|>"}
    marker_b = {"roles": {"user": {"open": "U: ", "close": "\n"}},
                "sep": "", "assistant_open": "A: "}
    a = count_tokens(render_chat_template(convo, marker_a))
    b = count_tokens(render_chat_template(convo, marker_b))
    assert a != b  # identical messages, different prompt tokens
