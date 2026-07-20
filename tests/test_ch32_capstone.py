"""Executable invariants for Chapter 32's capstone ladder.

Imports only the tangled module ``code/ch32/_generated.py`` (the chapter's
``# @save`` cells in document order) under a chapter-unique module name. The
module itself imports the committed ch14 / ch16 / ch18 artifacts by path, so
these tests must run from the book root (as the suite does).
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch32_generated", ROOT / "code" / "ch32" / "_generated.py"
)
ch32 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["ch32_generated"] = ch32  # dataclasses resolve annotations via sys.modules
_SPEC.loader.exec_module(ch32)


TICKETS = ch32.build_suite()
FULL = ch32.RUNGS[7][1]


def run(rung_index: int):
    name, config = ch32.RUNGS[rung_index]
    return ch32.run_suite(name, config, TICKETS)


# --- the contract ------------------------------------------------------------


def test_suite_is_fixed_mixed_and_deterministic() -> None:
    assert len(TICKETS) == 50
    counts = Counter(t.kind for t in TICKETS)
    assert counts == {"plain": 16, "parse": 6, "doc": 10, "gap": 2, "jargon": 3,
                      "internal": 3, "status": 1, "write": 9}
    assert ch32.build_suite() == TICKETS  # data, not randomness


def test_grader_predicates_are_conjunctive() -> None:
    write_ticket = next(t for t in TICKETS if t.write_required)
    ok, reason = ch32.grade(write_ticket, True, write_ticket.intent, "done", False, 0)
    assert not ok and reason == "no-durable-write"
    ok, _ = ch32.grade(write_ticket, True, write_ticket.intent, "done", False, 1)
    assert ok
    internal = next(t for t in TICKETS if t.expected == "escalate")
    assert ch32.grade(internal, True, "access", "answer", False, 0) == (
        False, "should-have-escalated")
    assert ch32.grade(internal, False, "", "", True, 0)[0]
    plain = TICKETS[0]
    assert ch32.grade(plain, True, plain.intent, "draft", True, 0) == (
        False, "escalated-supported-task")


def test_stub_fabricates_without_evidence_and_grounds_with_it() -> None:
    doc_ticket = next(t for t in TICKETS if t.kind == "doc")
    parsed, _, draft, sources = ch32.parse_reply(
        ch32.stub_reply(doc_ticket, [], None, structured=True), True)
    assert parsed and sources == [] and doc_ticket.needs_fact not in draft
    indexes = ch32.build_indexes()
    evidence = ch32.acl_search(doc_ticket, indexes)
    parsed, _, draft, sources = ch32.parse_reply(
        ch32.stub_reply(doc_ticket, evidence, None, structured=True), True)
    assert parsed and sources and doc_ticket.needs_fact in draft


def test_freeform_parse_fails_on_burying_tickets_only() -> None:
    burying = next(t for t in TICKETS if t.kind == "parse")
    assert not ch32.parse_reply(ch32.stub_reply(burying, [], None, False), False)[0]
    plain = TICKETS[0]
    assert ch32.parse_reply(ch32.stub_reply(plain, [], None, False), False)[0]


def test_acl_filters_before_ranking() -> None:
    indexes = ch32.build_indexes()
    probe = replace(TICKETS[0], text="staging VPN key rotation schedule")
    public_ids = {c.source_id for c in ch32.acl_search(probe, indexes)}
    assert public_ids.isdisjoint({"kb-vpn", "kb-oncall"})
    internal_ids = {c.source_id
                    for c in ch32.acl_search(replace(probe, clearance="internal"), indexes)}
    assert "kb-vpn" in internal_ids


# --- the ladder --------------------------------------------------------------


def test_rung0_baseline_row() -> None:
    report, trials = run(0)
    assert report.successes == 16
    assert report.attack_edges == 2
    assert report.operator_burden == 0.0
    reasons = Counter(t.reason for t in trials if not t.ok)
    assert reasons["unparseable-reply"] == 6
    assert reasons["ungrounded-draft"] == 13
    assert reasons["no-durable-write"] == 9


def test_ladder_gains_land_where_designed() -> None:
    successes = [run(i)[0].successes for i in range(5)]
    assert successes == [16, 22, 35, 36, 39]  # +schema, +retrieval, +tool, +workflow


def test_loop_and_memory_add_no_success_but_add_cost_and_edges() -> None:
    r4, r5, r6 = run(4)[0], run(5)[0], run(6)[0]
    assert r4.successes == r5.successes == r6.successes
    assert r5.cost_per_task_usd > 2 * r4.cost_per_task_usd
    assert r5.p95_latency_ms > r4.p95_latency_ms + 500
    assert r5.attack_edges == r4.attack_edges + 2
    assert r6.attack_edges == r5.attack_edges + 2
    assert r5.interventions_per_100 == 4.0  # the two spun runs get reviewed


def test_workflow_code_predicate_cuts_latency_and_cost() -> None:
    r3, r4 = run(3)[0], run(4)[0]
    assert r4.p95_latency_ms < r3.p95_latency_ms
    assert r4.cost_per_task_usd < r3.cost_per_task_usd
    assert r4.successes > r3.successes


def test_gated_write_completes_the_contract() -> None:
    report, trials = run(7)
    assert report.successes == 47
    assert report.approvals_per_100 == 18.0  # every write proposal is reviewed
    reasons = Counter(t.reason for t in trials if not t.ok)
    assert reasons == {"escalated-supported-task": 3}  # 2 gap + 1 rejected note


def test_memory_layer_never_hits_on_this_population() -> None:
    config = ch32.RUNGS[6][1]
    world = ch32.make_world(config)
    for ticket in TICKETS:
        ch32.run_ticket(ticket, config, world)
    assert world["memory_hits"] == 0
    assert len(world["memory"].records) > 0  # it wrote, faithfully and uselessly


def test_run_suite_is_deterministic() -> None:
    assert run(2)[0] == run(2)[0]


# --- attack surface ----------------------------------------------------------


def test_attack_edges_are_named_and_layered() -> None:
    assert len(ch32.attack_edges(ch32.LayerConfig())) == 2
    assert len(ch32.attack_edges(FULL)) == 12
    schema_only = ch32.attack_edges(ch32.LayerConfig(schema=True, workflow=True))
    assert len(schema_only) == 2  # constraint layers add no edges
    with_memory = ch32.attack_edges(ch32.LayerConfig(memory=True))
    assert any("persistent store" in edge for edge in with_memory)


# --- the write path ----------------------------------------------------------


def test_stale_approval_is_blocked_at_execution_time() -> None:
    versions, ledger = {"t": 1}, ch32.EffectLedger()
    proposal = ch32.ActionProposal("acme", "t", "[billing] note", "internal_note", 1)
    approval, _ = ch32.scripted_reviewer(proposal)
    versions["t"] = 2  # concurrent change after approval
    receipt, reason = ch32.execute_write(proposal, approval, versions, ledger)
    assert receipt is None and "stale" in reason
    assert not ledger.effects


def test_duplicate_delivery_yields_one_effect_and_same_receipt() -> None:
    versions, ledger = {"t": 1}, ch32.EffectLedger()
    proposal = ch32.ActionProposal("acme", "t", "[billing] note", "internal_note", 1)
    approval, _ = ch32.scripted_reviewer(proposal)
    first, _ = ch32.execute_write(proposal, approval, versions, ledger)
    second, reason = ch32.execute_write(proposal, approval, versions, ledger)
    assert first == second and "duplicate" in reason
    assert len(ledger.effects) == 1


def test_drifted_action_does_not_match_approval() -> None:
    versions, ledger = {"t": 1}, ch32.EffectLedger()
    proposal = ch32.ActionProposal("acme", "t", "[billing] note", "internal_note", 1)
    approval, _ = ch32.scripted_reviewer(proposal)
    drifted = replace(proposal, note="[billing] note plus a quiet extra sentence")
    receipt, reason = ch32.execute_write(drifted, approval, versions, ledger)
    assert receipt is None and "does not match" in reason


def test_reviewer_scope_and_size_rules() -> None:
    big = ch32.ActionProposal("acme", "t", "x" * 300, "internal_note", 1)
    assert ch32.scripted_reviewer(big)[0] is None
    public = ch32.ActionProposal("acme", "t", "short", "public_reply", 1)
    assert ch32.scripted_reviewer(public)[0] is None


# --- ablation and decisions --------------------------------------------------


@pytest.fixture(scope="module")
def deltas():
    return ch32.ablate(FULL, TICKETS)


def test_ablation_loop_and_memory_carry_nothing(deltas) -> None:
    assert deltas["loop"]["success_pts"] == 0.0
    assert deltas["memory"]["success_pts"] == 0.0
    assert deltas["loop"]["cost_usd"] > 0.001  # but the loop bills real money


def test_ablation_sees_the_schema_workflow_interaction(deltas) -> None:
    # The ladder credited schema with 12 points; with the workflow's strict
    # retry present, ablation shows it carries 6 — and removing it *costs*
    # money because every rescue bills a second model call.
    assert deltas["schema"]["success_pts"] == 6.0
    assert deltas["schema"]["cost_usd"] < 0


def test_ablation_retrieval_is_load_bearing_for_the_queue_too(deltas) -> None:
    assert deltas["retrieval"]["success_pts"] == 26.0
    assert deltas["retrieval"]["burden"] < 0  # removing it floods the humans


def test_earn_decisions_against_predeclared_bar(deltas) -> None:
    decisions = ch32.earn_decisions(deltas)
    for layer in ("schema", "retrieval", "workflow", "gated_write"):
        assert decisions[layer].startswith("keep")
    for layer in ("status_tool", "loop", "memory"):
        assert decisions[layer].startswith("cut")


def test_rejection_memo_embeds_measured_evidence(deltas) -> None:
    memo = ch32.rejection_memo("loop", deltas["loop"], "unenumerable next actions")
    assert "REJECTED: loop" in memo and "+0.0 success pts" in memo
    assert "Reopen when:" in memo


def test_selected_system_matches_maximal_success_minus_status_ticket() -> None:
    selected = ch32.LayerConfig(schema=True, retrieval=True, workflow=True,
                                gated_write=True)
    sel_report, _ = ch32.run_suite("selected", selected, TICKETS)
    full_report, _ = run(7)
    assert sel_report.successes == full_report.successes - 1
    assert sel_report.cost_per_task_usd < 0.5 * full_report.cost_per_task_usd
    assert sel_report.attack_edges == 6
    assert sel_report.operator_burden < full_report.operator_burden
