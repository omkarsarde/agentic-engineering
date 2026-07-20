"""Executable security claims for Chapter 24's containment boundary.

Imports only the tangled module ``code/ch24/_generated.py`` (the chapter's
``# @save`` cells in document order) under a chapter-unique module name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch24_generated", ROOT / "code" / "ch24" / "_generated.py"
)
ch24 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["ch24_generated"] = ch24  # dataclasses resolve annotations via sys.modules
_SPEC.loader.exec_module(ch24)


def _matrix() -> dict[str, dict[str, str]]:
    return {name: {cfg: ch24.run_attack(name, cfg) for cfg in ch24.CONFIGS}
            for name in ch24.ATTACKS}


# --- the attack phase: the naive agent is fully exploited -------------------

def test_naive_agent_is_exploited_by_both_injections() -> None:
    world = ch24.fresh_world()
    ctx = ch24.assemble_context(ch24.direct, ch24.retrieve("refund order"))
    ch24.naive_execute(ch24.compromised_model(ctx), world)
    assert ch24.attack_succeeded(world)  # direct injection moved money

    world = ch24.fresh_world()
    ctx = ch24.assemble_context(ch24.benign, ch24.retrieve("order A-17 note"))
    ch24.naive_execute(ch24.compromised_model(ctx), world)
    assert ch24.attack_succeeded(world)  # indirect injection exfiltrated a URL


# --- the re-attack suite: which layer stops which attack --------------------

def test_matrix_naive_leaks_every_attack() -> None:
    matrix = _matrix()
    assert all(matrix[name]["naive"] == "achieved" for name in ch24.ATTACKS)


def test_gate_contains_every_instruction_attack() -> None:
    matrix = _matrix()
    instruction = [n for n, a in ch24.ATTACKS.items() if a["kind"] == "instruction"]
    for name in instruction:
        assert matrix[name]["gate"] != "achieved"
        assert matrix[name]["full"] != "achieved"


def test_detector_miss_does_not_break_gate_containment() -> None:
    matrix = _matrix()
    # the detector lets at least one instruction attack through ...
    assert matrix["indirect-transfer"]["detector"] == "achieved"
    # ... yet the gate contains it regardless of the detector.
    assert matrix["indirect-transfer"]["gate"] != "achieved"


def test_source_separation_contains_indirect_upstream_of_the_gate() -> None:
    matrix = _matrix()
    assert matrix["indirect-transfer"]["full"] == "source-sep"
    assert matrix["indirect-exfil"]["full"] == "source-sep"
    # a direct injection lives in the trusted query, so the gate catches it.
    assert matrix["direct-transfer"]["full"] == "gate"


def test_knowledge_poison_survives_authorization_controls() -> None:
    # Authorization is not integrity: an authorized-but-false-premise refund
    # is not contained by the gate or by source separation.
    matrix = _matrix()
    assert matrix["knowledge-poison"]["gate"] == "achieved"
    assert matrix["knowledge-poison"]["full"] == "achieved"


# --- the policy enforcement point -------------------------------------------

def test_irreversible_action_requires_external_approval() -> None:
    action = ch24.Action("wire_transfer", {"amount": 500, "to": "x"})
    world = ch24.fresh_world()
    gate = ch24.EnforcementPoint(ch24.PolicyEngine(), ch24.AuditLog())
    decision = gate.execute(action, ch24.agent_principal(), world)
    assert decision.effect == "review"
    assert world["balance"] == 1000


def test_bound_approval_allows_the_exact_action() -> None:
    actor = ch24.agent_principal()
    action = ch24.Action("refund", {"order_id": "A-17", "amount": 9999})
    approval = ch24.Approval("reviewer-9", ch24.action_digest(action, actor), ch24.POLICY_VERSION)
    world = ch24.fresh_world()
    gate = ch24.EnforcementPoint(ch24.PolicyEngine(), ch24.AuditLog())
    assert gate.execute(action, actor, world, approval).effect == "allow"
    assert world["refunds"] == ["A-17"]


def test_substituted_action_is_rejected() -> None:
    actor = ch24.agent_principal()
    approved = ch24.Action("refund", {"order_id": "A-17", "amount": 9999})
    swapped = ch24.Action("refund", {"order_id": "A-99", "amount": 9999})
    token = ch24.Approval("reviewer-9", ch24.action_digest(approved, actor), ch24.POLICY_VERSION)
    world = ch24.fresh_world()
    gate = ch24.EnforcementPoint(ch24.PolicyEngine(), ch24.AuditLog())
    assert gate.execute(swapped, actor, world, token).effect == "deny"
    assert world["refunds"] == []


def test_self_issued_approval_is_rejected() -> None:
    actor = ch24.agent_principal()
    action = ch24.Action("wire_transfer", {"amount": 500, "to": "x"})
    self_approval = ch24.Approval(actor.subject, ch24.action_digest(action, actor), ch24.POLICY_VERSION)
    gate = ch24.EnforcementPoint(ch24.PolicyEngine(), ch24.AuditLog())
    assert gate.execute(action, actor, ch24.fresh_world(), self_approval).effect == "deny"


def test_egress_is_allowlisted() -> None:
    gate = ch24.EnforcementPoint(ch24.PolicyEngine(), ch24.AuditLog())
    world = ch24.fresh_world()
    blocked = gate.execute(ch24.Action("fetch_url", {"url": "https://evil.example/c"}),
                           ch24.agent_principal(), world)
    assert blocked.effect == "deny"
    assert world["fetched"] == []
    allowed = gate.execute(ch24.Action("fetch_url", {"url": "https://help.example/status"}),
                           ch24.agent_principal(), world)
    assert allowed.effect == "allow"


def test_missing_scope_is_denied() -> None:
    weak = ch24.Principal("agent", "tenant-7", frozenset({"order:read"}))
    gate = ch24.EnforcementPoint(ch24.PolicyEngine(), ch24.AuditLog())
    decision = gate.execute(ch24.Action("wire_transfer", {"amount": 1, "to": "x"}),
                            weak, ch24.fresh_world())
    assert decision.effect == "deny"


def test_audit_chain_detects_tampering() -> None:
    audit = ch24.AuditLog()
    gate = ch24.EnforcementPoint(ch24.PolicyEngine(), audit)
    gate.execute(ch24.Action("fetch_url", {"url": "https://evil.example/c"}),
                 ch24.agent_principal(), ch24.fresh_world())
    assert audit.verify()
    audit.entries[0]["decision"]["reason"] = "rewritten"
    assert not audit.verify()


# --- source separation, sandbox, identity -----------------------------------

def test_poisoned_document_cannot_inject_a_step() -> None:
    plan = ch24.privileged_planner(ch24.benign)
    slots = ch24.quarantined_extractor(ch24.retrieve("order A-17 note status"))
    assert not any(step.tool == "wire_transfer" for step in plan)
    assert slots["order_status"].label == "untrusted"


def test_sandbox_blocks_egress_and_file_reads() -> None:
    assert ch24.restricted_exec("import socket")[0] is False
    assert ch24.restricted_exec("result = open('/etc/passwd').read()")[0] is False
    ok, detail = ch24.restricted_exec("result = sum(range(10))")
    assert ok and detail == "45"


def test_scoped_handler_denies_cross_tenant_access() -> None:
    actor = ch24.agent_principal()
    cross = ch24.Action("order_lookup", {"order_id": "Z-1", "tenant_id": "tenant-9"})
    assert ch24.scoped_handler(cross, actor) == "deny: cross-tenant access"
    same = ch24.Action("order_lookup", {"order_id": "Z-1", "tenant_id": "tenant-7"})
    assert ch24.scoped_handler(same, actor)["tenant_id"] == "tenant-7"


# --- the model as an asset --------------------------------------------------

def test_membership_inference_recovers_training_members() -> None:
    from sklearn.neural_network import MLPClassifier

    Xm, ym, Xn, yn = ch24.make_split()
    model = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=600,
                          alpha=1e-5, random_state=0).fit(Xm, ym)
    loss_m = ch24.per_example_loss(model, Xm, ym)
    loss_n = ch24.per_example_loss(model, Xn, yn)
    advantage, _threshold = ch24.attack_advantage(loss_m, loss_n)
    assert advantage > 0.3               # memorization leaks membership
    assert loss_m.mean() < loss_n.mean()  # members sit at lower loss
