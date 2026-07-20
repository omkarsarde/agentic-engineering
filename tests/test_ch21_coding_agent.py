"""Executable invariants for the Chapter 21 agent-applications teaching code.

Imports the tangled module ``code/ch21/_generated.py`` (produced from the
chapter's ``# @save`` cells by ``scripts/tangle.py``) and checks the properties
the chapter claims: that the coding agent's executor owns its invariants (path
containment, read-only tests, unambiguous anchors, syntax admission), that the
scored suite produces the exact resolved / proposal / test-run / rejection
counts shown, that the repair budget is a semantic knob, that a stale line-range
edit is caught by its digest, that the citation audit catches an unsupported but
cited claim, that a stale UI reference is refused, that a policy-gated refund
verifies against final state, and that the unit and join-fan-out gates fire.

The module is loaded under a unique name (``ch21_generated``) rather than the
bare ``sys.path`` pattern because several chapters each ship a module called
``_generated``; a plain import would collide inside one pytest process, and the
name must be registered in ``sys.modules`` before exec so the module's frozen
dataclasses can resolve their own module namespace.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch21_generated", ROOT / "code" / "ch21" / "_generated.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ch21 = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("ch21_generated", ch21)
_SPEC.loader.exec_module(ch21)

Edit = ch21.Edit
EditRejected = ch21.EditRejected
StaleObservation = ch21.StaleObservation
UnitError = ch21.UnitError
apply_edit = ch21.apply_edit
apply_guarded_edit = ch21.apply_guarded_edit
autonomy_regime = ch21.autonomy_regime
citation_audit = ch21.citation_audit
digest = ch21.digest
reconcile_join = ch21.reconcile_join
refund = ch21.refund
repository_map = ch21.repository_map
resolve = ch21.resolve
run_suite = ch21.run_suite
to_dollars = ch21.to_dollars
tree_digest = ch21.tree_digest
Account = ch21.Account
EvidenceRecord = ch21.EvidenceRecord
UINode = ch21.UINode

TASKS = json.loads((ROOT / "data" / "ch21" / "tasks.json").read_text(encoding="utf-8"))


# --- coding agent: the scored suite and its executor invariants ---------------

def test_suite_produces_the_shown_counts():
    report = run_suite(TASKS)
    assert report["tasks"] == 6
    assert report["resolved"] == 6
    assert report["proposals"] == 8
    assert report["test_runs"] == 7          # one rejected edit never reaches the runner
    assert report["rejected_edits"] == 1
    json.dumps(report)                        # the event report is JSON-serializable


def test_repair_budget_is_semantic():
    assert run_suite(TASKS, max_proposals=1)["resolved"] == 4
    assert run_suite(TASKS, max_proposals=2)["resolved"] == 6


def test_workspace_escape_is_rejected_before_write():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        with pytest.raises(EditRejected):
            apply_edit(root, Edit("../escape.py", "a", "b", "malformed"))


def test_tests_are_read_only():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        (root / "test_app.py").write_text("value = 1\n", encoding="utf-8")
        with pytest.raises(EditRejected):
            apply_edit(root, Edit("test_app.py", "1", "2", "game the oracle"))


def test_ambiguous_anchor_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        (root / "app.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
        with pytest.raises(EditRejected):
            apply_edit(root, Edit("app.py", "x = 1", "x = 2", "ambiguous"))


def test_syntax_break_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        (root / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        with pytest.raises(EditRejected):
            apply_edit(root, Edit("app.py", "return 1", "return (", "break it"))


def test_repository_map_exposes_structure_not_source():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        (root / "app.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
        mapped = repository_map(root)
        assert mapped[0]["symbols"] == ["answer"]
        assert "source" not in mapped[0]


# --- edit formats: the digest catches stale drift -----------------------------

def test_guarded_edit_rejects_stale_digest():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        (root / "m.py").write_text("y = 1\n", encoding="utf-8")
        stale = digest("something the proposer saw earlier")
        with pytest.raises(EditRejected):
            apply_guarded_edit(root, "m.py", stale, 0, 1, ["y = 2"])


def test_guarded_edit_applies_on_fresh_digest():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        src = "y = 1\n"
        (root / "m.py").write_text(src, encoding="utf-8")
        receipt = apply_guarded_edit(root, "m.py", digest(src), 0, 1, ["y = 2"])
        assert "y = 2" in (root / "m.py").read_text(encoding="utf-8")
        assert receipt.startswith("---")


# --- deep research: the bidirectional citation audit --------------------------

def test_citation_audit_flags_cited_but_unsupported():
    claims = {"c1": "supported claim", "c2": "cited-but-context claim", "c3": "uncited claim"}
    records = [
        EvidenceRecord("c1", "s1", "span", "supports", "peer-reviewed", "2026-07-10"),
        EvidenceRecord("c2", "s2", "span", "context", "marketing", "2026-07-11"),
    ]
    findings = citation_audit(claims, records)
    joined = " | ".join(findings)
    assert "c1" not in joined                 # genuinely supported
    assert "c2: cited but UNSUPPORTED" in joined
    assert "c3: NO CITATION" in joined


# --- computer use: grounding fails safe ---------------------------------------

def test_resolve_refuses_ambiguous_and_missing():
    tree = [UINode("button", "Submit", "b1"), UINode("button", "Submit", "b9")]
    with pytest.raises(StaleObservation):
        resolve(tree, "button", "Submit")     # two matches
    with pytest.raises(StaleObservation):
        resolve(tree, "button", "Cancel")     # zero matches
    assert resolve([UINode("button", "Ok", "b2")], "button", "Ok") == "b2"


def test_tree_digest_changes_when_ui_changes():
    v1 = [UINode("button", "Submit", "b1")]
    v2 = v1 + [UINode("button", "Submit", "b9")]
    assert tree_digest(v1) != tree_digest(v2)


# --- conversational: policy gate + final-state verification --------------------

def test_refund_commits_within_policy_and_verifies():
    policy = {"max_days": 30, "max_amount": 100.0}
    result = refund(Account("c", paid=80.0, days_since_purchase=12), 80.0, policy)
    assert result["verdict"] == "committed"
    assert result["verified"] is True
    assert result["refunded"] == 80.0


def test_refund_escalates_outside_policy():
    policy = {"max_days": 30, "max_amount": 100.0}
    stale = refund(Account("c", paid=80.0, days_since_purchase=45), 80.0, policy)
    over = refund(Account("c", paid=80.0, days_since_purchase=1), 500.0, policy)
    assert stale["verdict"] == "escalate" and "window" in stale["packet"]["reason"]
    assert over["verdict"] == "escalate" and "limit" in over["packet"]["reason"]


# --- data rigor: unit and join-fan-out gates ----------------------------------

def test_unit_normalization_and_unknown_unit():
    assert to_dollars(3.0, "thousands") == 3000.0
    assert to_dollars(1200.0, "dollars") == 1200.0
    with pytest.raises(UnitError):
        to_dollars(1.0, "euros")


def test_join_fanout_is_caught_but_clean_join_passes():
    with pytest.raises(UnitError):
        reconcile_join(5, 7, key_unique=False)
    reconcile_join(5, 5, key_unique=True)     # unique key, no fan-out -> no raise


# --- the autonomy regime map --------------------------------------------------

def test_autonomy_regime_quadrants():
    assert autonomy_regime(0.9, 0.9) == "iterate freely"
    assert autonomy_regime(0.3, 0.9) == "stage for review"
    assert autonomy_regime(0.9, 0.2) == "verify, then commit"
    assert autonomy_regime(0.2, 0.2) == "advisory (human commits)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
