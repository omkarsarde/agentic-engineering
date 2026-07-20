"""Executable invariants for the Chapter 18 memory teaching code.

Imports the tangled module ``code/ch18/_generated.py`` (produced from the
chapter's ``# @save`` cells by ``scripts/tangle.py``) and checks the real
properties the chapter claims: the write gate refuses unsafe candidates; a
semantic contradiction supersedes while retaining history (Berlin beats Munich)
and stays separable under two clocks; retrieval enforces scope before it ranks,
abstains as a first-class outcome, and forgets on a time-to-live; the
self-editing agent classifies each turn ADD/UPDATE/NOOP/DELETE; experience
raises a task score across episodes through a held-out promotion gate that
blocks an over-general lesson; deletion propagates to derived summaries only
after invalidation; and the LongMemEval-style probes score the three designs as
the chapter reports.

The module is loaded under a unique name (``ch18_generated``) rather than the
bare ``sys.path`` pattern because several chapters each ship a module called
``_generated``; a plain import would collide inside one pytest process, and the
frozen dataclasses need the module registered before execution.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch18_generated", ROOT / "code" / "ch18" / "_generated.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ch18 = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("ch18_generated", ch18)
_SPEC.loader.exec_module(ch18)

Candidate = ch18.Candidate
DeletionManifest = ch18.DeletionManifest
Kind = ch18.Kind
Lesson = ch18.Lesson
LessonStore = ch18.LessonStore
MemoryAgent = ch18.MemoryAgent
MemoryStore = ch18.MemoryStore
Op = ch18.Op
ScriptedExtractor = ch18.ScriptedExtractor
Scope = ch18.Scope
Source = ch18.Source
Status = ch18.Status
Ticket = ch18.Ticket
base_policy = ch18.base_policy
reflect = ch18.reflect

MINA = Scope("acme", user_id="mina")


def _fact(value: str, when: int, source: Source = Source.USER) -> Candidate:
    return Candidate("home city", value, Kind.SEMANTIC, MINA, source, f"e{when}", when)


# --- write gate ------------------------------------------------------------
def test_write_gate_rejects_unsafe_sources() -> None:
    store = MemoryStore()
    # missing provenance
    assert store.write(Candidate("k", "v", Kind.SEMANTIC, MINA, Source.USER, "", 1))[0] is None
    # retrieved documents cannot become durable memory
    doc = Candidate("k", "hello", Kind.SEMANTIC, MINA, Source.RETRIEVED_DOCUMENT, "d1", 1)
    ok, reason = store.write(doc)
    assert ok is None and "retrieved text" in reason
    # model may not self-author a procedure
    proc = Candidate("k", "do x", Kind.PROCEDURAL, MINA, Source.MODEL_INFERENCE, "m1", 1)
    assert store.write(proc)[0] is None
    # out-of-range confidence
    bad = Candidate("k", "v", Kind.SEMANTIC, MINA, Source.USER, "e1", 1, confidence=1.4)
    assert store.write(bad)[0] is None


def test_write_gate_accepts_valid_user_fact() -> None:
    store = MemoryStore()
    record, reason = store.write(_fact("Munich", 10))
    assert record is not None and record.status is Status.ACTIVE and reason == "accepted"


# --- supersession and bitemporal truth -------------------------------------
def test_decide_op_add_noop_update() -> None:
    store = MemoryStore()
    assert store.decide_op(_fact("Munich", 10)) is Op.ADD
    store.write(_fact("Munich", 10))
    assert store.decide_op(_fact("Munich", 14)) is Op.NOOP
    assert store.decide_op(_fact("Berlin", 20)) is Op.UPDATE


def test_supersession_retains_history() -> None:
    store = MemoryStore()
    store.write(_fact("Munich", 10))
    store.write(_fact("Berlin", 20))
    old = next(r for r in store.records.values() if r.value == "Munich")
    assert old.status is Status.SUPERSEDED and old.valid_to == 20
    assert store.retrieve("home city", MINA).value == "Berlin"
    assert store.retrieve("home city", MINA, as_of=15).value == "Munich"
    assert store.retrieve("home city", MINA, as_of=25).value == "Berlin"


def test_bitemporal_true_versus_believed() -> None:
    store = MemoryStore()
    store.write(_fact("Munich", 10), now=10)
    store.write(_fact("Berlin", 18), now=30)  # learned late
    # valid-time truth at t=20 is Berlin; what we believed at t=20 is Munich
    assert store.retrieve("home city", MINA, as_of=20).value == "Berlin"
    believed = max((r for r in store.records.values()
                    if r.recorded_at <= 20 and r.valid_from <= 20), key=lambda r: r.valid_from)
    assert believed.value == "Munich"


# --- scoped retrieval, abstention, forgetting ------------------------------
def test_scope_is_enforced_before_ranking() -> None:
    store = MemoryStore()
    raj = Scope("globex", user_id="raj")
    store.write(_fact("Berlin", 10))
    store.write(Candidate("home city", "Berlin", Kind.SEMANTIC, raj, Source.USER, "b1", 10))
    assert len(store.index["berlin"]) == 2  # both tenants share the posting
    assert store.retrieve("home city", raj).value == "Berlin"
    assert store.retrieve("home city", Scope("zzz", user_id="mina")) is None


def test_retrieval_abstains_on_unknown() -> None:
    store = MemoryStore()
    store.write(_fact("Berlin", 10))
    assert store.retrieve("what is my favorite color?", MINA) is None


def test_ttl_forgetting_removes_from_index() -> None:
    store = MemoryStore()
    store.write(Candidate("otp", "verifying", Kind.WORKING, MINA, Source.VERIFIED_TOOL, "o1", 100, ttl=5))
    assert store.retrieve("otp", MINA, now=103) is not None
    assert store.expire(now=106)  # returns the expired id(s)
    assert store.retrieve("otp", MINA, now=106) is None


# --- self-editing agent ----------------------------------------------------
def test_agent_edits_reduce_to_four_ops() -> None:
    agent = MemoryAgent(MemoryStore(), ScriptedExtractor(MINA))
    ops = []
    for t, turn in [(10, "I live in Munich and I prefer aisle seats."),
                    (14, "Still living in Munich."),
                    (20, "I just moved to Berlin!"),
                    (24, "please forget where I live.")]:
        ops.extend(call.op for call in agent.observe(turn, t))
    assert Op.ADD in ops and Op.UPDATE in ops and Op.NOOP in ops and Op.DELETE in ops
    assert agent.answer("home city") == "I don't have that in memory."
    assert agent.answer("seat preference") == "aisle"


# --- experiential learning -------------------------------------------------
def _triage_sets():
    train = [Ticket("c1", frozenset({"refund"}), "BILLING"),
             Ticket("c2", frozenset({"refund", "fraud"}), "FRAUD"),
             Ticket("c3", frozenset({"crash"}), "ENGINEERING"),
             Ticket("c4", frozenset({"crash", "security"}), "SECURITY")]
    held = [Ticket("h1", frozenset({"refund", "fraud", "eu"}), "FRAUD"),
            Ticket("h2", frozenset({"crash", "security", "x"}), "SECURITY")]
    neg = [Ticket("n1", frozenset({"refund"}), "BILLING"),
           Ticket("n2", frozenset({"crash"}), "ENGINEERING")]
    return train, held, neg


def test_lessons_improve_score_across_episodes() -> None:
    train, held, neg = _triage_sets()
    store = LessonStore()
    scores = []
    for _ in range(4):
        correct, promoted = 0, False
        for ticket in train:
            if store.decide(ticket) == ticket.gold:
                correct += 1
            elif not promoted and not any(l.pattern == ticket.keywords for l in store.lessons):
                ok, _ = store.promote(reflect(ticket), held, neg)
                promoted = ok
        scores.append(correct / len(train))
    assert scores[0] < scores[-1]  # experience raised the score
    assert scores[-1] == 1.0


def test_promotion_gate_blocks_over_general_lesson() -> None:
    _, held, neg = _triage_sets()
    store = LessonStore()
    over_general = Lesson(pattern=frozenset({"refund"}), action="FRAUD")
    promoted, report = store.promote(over_general, held, neg)
    assert promoted is False and "misfire" in report


# --- deletion propagation --------------------------------------------------
def test_deletion_fails_until_derived_invalidated() -> None:
    store = MemoryStore()
    e1, _ = store.write(Candidate("visit", "3rd", Kind.EPISODIC, MINA, Source.VERIFIED_TOOL, "v1", 10))
    e2, _ = store.write(Candidate("visit", "9th", Kind.EPISODIC, MINA, Source.VERIFIED_TOOL, "v2", 12))
    summary, reason = store.consolidate(
        [e1, e2], Candidate("care summary", "frequent visitor", Kind.SEMANTIC,
                            MINA, Source.VERIFIED_TOOL, "c1", 13, confidence=0.8))
    assert reason == "consolidated" and set(summary.parents) == {e1.record_id, e2.record_id}

    first = store.delete_subject(MINA)
    assert first.complete() is False
    assert first.targets["derived_summaries"] == "OPEN"
    assert store.retrieve("care summary", MINA) is not None  # still leaks after pass 1

    assert summary.record_id in store.invalidate_derived()
    assert store.delete_subject(MINA).complete() is True
    assert store.retrieve("care summary", MINA) is None


def test_consolidate_requires_two_verified_episodes() -> None:
    store = MemoryStore()
    e1, _ = store.write(Candidate("visit", "3rd", Kind.EPISODIC, MINA, Source.VERIFIED_TOOL, "v1", 10))
    summary, reason = store.consolidate(
        [e1], Candidate("s", "x", Kind.SEMANTIC, MINA, Source.VERIFIED_TOOL, "c1", 13))
    assert summary is None and "two verified" in reason


# --- evaluation probes -----------------------------------------------------
def test_probe_suite_scores_three_designs() -> None:
    governed = ch18.probe_governed()
    transcript = ch18.probe_transcript_only()
    nomem = ch18.probe_no_memory()
    assert governed == {"update": 1, "temporal": 1, "abstention": 1, "isolation": 1, "deletion": 1}
    assert transcript["isolation"] == 0 and transcript["update"] == 1
    assert nomem["update"] == 0 and nomem["abstention"] == 1
