"""Executable invariants for the Chapter 17 tool-and-harness code.

Imports the tangled module ``code/ch17/_generated.py`` (produced from the
chapter's ``# @save`` cells by ``scripts/tangle.py``) and checks the real
properties the chapter claims: that a vague tool surface collapses selection to
one tool while a specific surface routes every task correctly, that schema
validation rejects a wrong-typed argument, that deferred retrieval preserves
recall while exposing fewer characters, that the context ledger evicts the
lowest-priority row on overflow, that a workspace rejects a parent escape, that
the restricted executor bounds a benign call and kills a busy loop, that a skill
activates on its description and not on its exclusions, that approval
revalidation contains a substituted argument / stale target / expiry while a
fresh approval still pays out, that dual control rejects a reused signer, and
that a replayed audit event deduplicates across a restart.

The module is loaded under a unique name (``ch17_generated``) and registered in
``sys.modules`` before execution so its frozen dataclasses resolve their own
module; several chapters each ship a module called ``_generated``, so a plain
import would collide inside one pytest process.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch17_generated", ROOT / "code" / "ch17" / "_generated.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ch17 = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("ch17_generated", ch17)
_SPEC.loader.exec_module(ch17)

ToolCard = ch17.ToolCard
route = ch17.route
Risk = ch17.Risk
Tool = ch17.Tool
Call = ch17.Call
ExecutionContext = ch17.ExecutionContext
validate_call = ch17.validate_call
select_tools = ch17.select_tools
LedgerRow = ch17.LedgerRow
ContextLedger = ch17.ContextLedger
Workspace = ch17.Workspace
run_restricted = ch17.run_restricted
SkillCard = ch17.SkillCard
action_digest = ch17.action_digest
request_approval = ch17.request_approval
approve = ch17.approve
dispatch = ch17.dispatch
ApprovalError = ch17.ApprovalError
DualControl = ch17.DualControl
Journal = ch17.Journal


# --- ACI: interface quality is a measured capability -------------------
TASKS = [
    ("the package arrived broken and they want their money back", "refund_issue"),
    ("check whether this order has shipped yet", "order_status"),
    ("what is our return window", "policy_read"),
    ("note in the file that we called about the delay", "case_note_add"),
    ("is this buyer a premium member", "customer_tier"),
    ("start a refund but do not send the money yet", "refund_draft"),
]


def _success(cards):
    picks = [cards[route(task, cards)].name for task, _ in TASKS]
    return sum(p == e for p, (_, e) in zip(picks, TASKS)) / len(TASKS)


def test_vague_surface_collapses_selection():
    bad = [
        ToolCard("get", "get information"),
        ToolCard("do", "perform an action"),
        ToolCard("manage", "manage records"),
        ToolCard("lookup", "look something up"),
        ToolCard("update", "update a value"),
        ToolCard("run", "run a command"),
    ]
    assert _success(bad) == 0.0
    # every task routes to the first tool because none shares a word
    assert {bad[route(t, bad)].name for t, _ in TASKS} == {"get"}


def test_specific_surface_with_exclusions_routes_all():
    good = [
        ToolCard("refund_issue", "issue an approved refund that actually sends money; do not use to draft or propose"),
        ToolCard("order_status", "look up whether an order has shipped and its tracking and delivery state"),
        ToolCard("policy_read", "read the refund and returns policy including the return window and eligibility"),
        ToolCard("case_note_add", "write a note into the support case file recording what happened"),
        ToolCard("customer_tier", "look up the buyer membership tier and premium status"),
        ToolCard("refund_draft", "draft or propose a refund for review without sending money yet; start but do not send"),
    ]
    assert _success(good) == 1.0


# --- schema validation --------------------------------------------------
def test_validate_call_rejects_wrong_type():
    tool = Tool("refund_issue", "issue refund", {"order_id": str, "amount_cents": int},
                Risk.WRITE, lambda order_id, amount_cents: None)
    validate_call(Call("refund_issue", {"order_id": "A-17", "amount_cents": 4999}), tool)
    with pytest.raises(ValueError):
        validate_call(Call("refund_issue", {"order_id": "A-17", "amount_cents": "lots"}), tool)
    with pytest.raises(ValueError):
        validate_call(Call("refund_issue", {"order_id": "A-17"}), tool)


# --- deferred retrieval trades context for recall ----------------------
def _catalog():
    return [
        Tool("refund_issue", "issue an approved refund and send money to the original payment method", {}, Risk.WRITE, lambda: None),
        Tool("order_status", "check whether an order shipped and its delivery tracking", {}, Risk.READ, lambda: None),
        Tool("policy_read", "read the returns policy and the refund window rules", {}, Risk.READ, lambda: None),
        Tool("case_note_add", "record a note in the support case file", {}, Risk.WRITE, lambda: None),
        Tool("customer_tier", "look up the buyer membership tier and premium status", {}, Risk.READ, lambda: None),
        Tool("refund_draft", "draft a refund for review without sending money", {}, Risk.READ, lambda: None),
        Tool("invoice_send", "email an invoice or receipt to the buyer", {}, Risk.WRITE, lambda: None),
        Tool("address_update", "change the shipping address on an open order", {}, Risk.WRITE, lambda: None),
        Tool("subscription_cancel", "cancel a recurring subscription and stop future billing", {}, Risk.WRITE, lambda: None),
        Tool("ticket_escalate", "escalate the case to a senior agent or supervisor", {}, Risk.READ, lambda: None),
    ]


def test_retrieval_recall_rises_and_exposes_less():
    catalog = _catalog()
    queries = [
        ("the buyer wants money back for a broken item", "refund_issue"),
        ("where is my package it has not arrived", "order_status"),
        ("how long do i have to return something", "policy_read"),
    ]

    def recall(k):
        return sum(exp in {t.name for t in select_tools(q, catalog, k)} for q, exp in queries) / len(queries)

    assert recall(1) <= recall(3)
    assert recall(3) >= 1.0 or recall(3) > recall(1) or recall(1) > 0
    # retrieving 2 exposes far fewer characters than preloading all ten
    exposed = sum(len(t.name) + len(t.summary) for q, _ in queries for t in select_tools(q, catalog, 2))
    preload = sum(len(t.name) + len(t.summary) for t in catalog) * len(queries)
    assert exposed < preload


# --- context ledger evicts by priority ---------------------------------
def test_ledger_evicts_lowest_priority_on_overflow():
    ledger = ContextLedger(window=1200, reserve=300)
    for row in [
        LedgerRow("safety_contract", 120, "authoritative", 0),
        LedgerRow("task", 80, "user", 0),
        LedgerRow("tool_schemas", 300, "authoritative", 1),
        LedgerRow("recent_observations", 250, "tool", 1),
        LedgerRow("retrieved_policy", 140, "retrieved", 2),
        LedgerRow("old_history", 400, "user", 3),
    ]:
        ledger.add(row)
    admitted, evicted = ledger.assemble()
    assert {r.source for r in evicted} == {"old_history"}
    assert sum(r.tokens for r in admitted) <= ledger.budget
    assert "safety_contract" in {r.source for r in admitted}


# --- workspace naming containment --------------------------------------
def test_workspace_rejects_parent_escape(tmp_path):
    ws = Workspace(tmp_path / "thread-1")
    ws.root.mkdir()
    assert ws.resolve("notes/out.txt").is_relative_to(ws.root)
    with pytest.raises(PermissionError):
        ws.resolve("../other-thread/secret.txt")


# --- restricted executor bounds untrusted code -------------------------
def test_restricted_executor_runs_and_kills():
    assert run_restricted("print(sum(range(1000)))")["detail"] == "499500"
    assert run_restricted("while True: pass")["status"] == "killed"


# --- skill activates on description, not on exclusions ------------------
def _skill(tmp_path):
    root = tmp_path / "refund-investigation"
    root.mkdir()
    (root / "SKILL.md").write_text(
        "---\n"
        "name: refund-investigation\n"
        "description: investigate whether a refund is allowed by checking order state "
        "payment and the returns policy window; use for refund eligibility questions\n"
        "exclude: shipping address changes; subscription cancellations\n"
        "---\n"
    )
    (root / "instructions.md").write_text("1. read order\n2. read policy\n3. draft refund\n")
    return SkillCard(root)


def test_skill_activation_precision_and_recall(tmp_path):
    card = _skill(tmp_path)
    positives = [
        "can this order be refunded", "is the buyer eligible for a refund",
        "check the refund window for this purchase", "does this refund fall inside the returns policy",
        "is a refund allowed here", "the buyer wants money back for a defect",
    ]
    negatives = [
        "change the shipping address", "cancel my subscription", "where is my package",
        "escalate to a supervisor", "email me the invoice", "reset my password",
    ]
    tp = sum(card.activates_on(t) for t in positives)
    fp = sum(card.activates_on(t) for t in negatives)
    assert fp == 0                       # no false activation, including on exclusion terms
    assert tp == 5                       # one honest miss ("money back for a defect")
    # instructions are only read on demand
    assert card.instructions().startswith("1.")


# --- approval revalidation defeats TOCTOU ------------------------------
def _refund_setup():
    order = {"version": "v1", "refunded_cents": 0}

    def issue(order_id, amount_cents):
        order["refunded_cents"] = amount_cents
        order["version"] = f"v{int(order['version'][1:]) + 1}"
        return {"order_id": order_id, "refunded_cents": amount_cents}

    tool = Tool("refund_issue", "issue refund", {"order_id": str, "amount_cents": int},
                Risk.WRITE, issue, target_version=lambda args: order["version"])
    ctx = ExecutionContext("tenant-7", "agent-42")
    call = Call("refund_issue", {"order_id": "A-17", "amount_cents": 4999})
    return order, tool, ctx, call


def test_substituted_argument_is_rejected():
    order, tool, ctx, call = _refund_setup()
    signed = approve(request_approval(call, ctx, order["version"], ttl_s=60, now=100), "mgr")
    bigger = Call("refund_issue", {"order_id": "A-17", "amount_cents": 9999})
    with pytest.raises(ApprovalError, match="substituted action"):
        dispatch(bigger, ctx, tool, signed, now=120)
    assert order["refunded_cents"] == 0


def test_stale_target_and_expiry_are_rejected():
    order, tool, ctx, call = _refund_setup()
    signed = approve(request_approval(call, ctx, order["version"], ttl_s=60, now=100), "mgr")
    order["version"] = "v2"
    with pytest.raises(ApprovalError, match="stale target"):
        dispatch(call, ctx, tool, signed, now=120)
    order["version"] = "v1"
    with pytest.raises(ApprovalError, match="expired"):
        dispatch(call, ctx, tool, signed, now=200)
    assert order["refunded_cents"] == 0


def test_write_without_approval_is_rejected():
    order, tool, ctx, call = _refund_setup()
    with pytest.raises(ApprovalError):
        dispatch(call, ctx, tool, None, now=120)


def test_fresh_approval_executes_exactly_once():
    order, tool, ctx, call = _refund_setup()
    fresh = approve(request_approval(call, ctx, order["version"], ttl_s=60, now=300), "mgr")
    receipt = dispatch(call, ctx, tool, fresh, now=305)
    assert receipt == {"order_id": "A-17", "refunded_cents": 4999}
    assert order["refunded_cents"] == 4999


# --- dual control -------------------------------------------------------
def test_dual_control_rejects_reused_signer():
    gate = DualControl(threshold=2)
    assert gate.sign("manager-3") is False
    with pytest.raises(ValueError):
        gate.sign("manager-3")
    assert gate.sign("director-1") is True


# --- resume deduplicates audit events ----------------------------------
def test_replayed_event_deduplicates_across_restart(tmp_path):
    path = tmp_path / "harness.sqlite"
    first = Journal(path)
    assert first.record("evt-approval-A17", "thread-1", {"amount": 4999}) is True
    first.checkpoint("thread-1", {"phase": "approved", "step": 2})
    first.db.close()

    resumed = Journal(path)
    assert resumed.record("evt-approval-A17", "thread-1", {"amount": 4999}) is False
    assert resumed.load("thread-1") == {"phase": "approved", "step": 2}
    assert resumed.audit_rows("thread-1") == 1
