"""Executable invariants for the Chapter 15 agentic-retrieval build.

Imports the tangled module ``code/ch15/_generated.py`` (produced from the
chapter's ``# @save`` cells by ``scripts/tangle.py``) and checks the claims the
chapter makes with numbers: that one-shot RAG cannot answer the three-hop
question until it over-retrieves while the agentic loop reaches it with a few
targeted searches; that the loop's accuracy saturates at a search budget of
three; that relevance and support are graded apart (the CRAG signal); that the
permission filter runs before ranking and at graph construction; and that an
indirect injection carried by a retrieved document fires against a naive reader
but is blocked once an integrity gate admits only trusted evidence.

The module is loaded under a unique name (``ch15_generated``) because several
chapters each ship a module called ``_generated``; a bare import would collide
inside one pytest process.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch15_generated", ROOT / "code" / "ch15" / "_generated.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ch15 = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("ch15_generated", ch15)
_SPEC.loader.exec_module(ch15)

corpus = ch15.corpus
search = ch15.search
extract_facts = ch15.extract_facts
one_shot = ch15.one_shot
agentic = ch15.agentic
assess = ch15.assess
build_graph = ch15.build_graph
local_search = ch15.local_search
global_search = ch15.global_search
naive_agent = ch15.naive_agent
hardened_agent = ch15.hardened_agent
grep = ch15.grep
token_count = ch15.token_count
questions = ch15.questions
score = ch15.score

ENG = frozenset({"engineering"})
REGION_Q = "Which region stores telemetry for Project Falcon?"


def test_extractor_reads_chain_and_ignores_distractors() -> None:
    facts = {d.id: extract_facts(d) for d in corpus()}
    assert ("Falcon", "depends_on", "Atlas") in facts["d01"]
    assert ("Atlas", "telemetry_store", "Aurora") in facts["d02"]
    assert ("Aurora", "region", "us-east-2") in facts["d03"]
    # Vocabulary-sharing distractors state no extractable relation.
    for did in ("d05", "d06", "d07", "d08"):
        assert facts[did] == ()


def test_permission_filter_precedes_ranking() -> None:
    docs = corpus()
    # The finance-only budget record is never returned to an engineering caller,
    # even when the query is an exact semantic match at large k.
    assert not any(d.id == "d09" for d in search(docs, "Falcon annual budget", ENG, k=9))


def test_one_shot_misses_multi_hop_until_it_over_retrieves() -> None:
    docs = corpus()
    for k in range(1, 7):
        assert one_shot(docs, REGION_Q, ENG, "Falcon", "region", k=k).answer is None
    deep = one_shot(docs, REGION_Q, ENG, "Falcon", "region", k=7)
    assert deep.answer == "us-east-2"
    assert deep.docs_read == 7


def test_agentic_loop_solves_multi_hop_with_targeted_searches() -> None:
    docs = corpus()
    result = agentic(docs, REGION_Q, ENG, "Falcon", "region")
    assert result.answer == "us-east-2"
    assert result.searches == 3
    # It never reads the distractors one-shot needed a deep k to skip.
    assert "search(Aurora) -> ['d02', 'd03']" in result.trace


def test_frontier_agentic_dominates_on_context_volume() -> None:
    docs = corpus()
    qs = questions()
    one_shot_acc = sum(
        score(one_shot(docs, t, g, s, goal, k=3).answer, exp)
        for _, t, g, s, goal, exp in qs
    ) / len(qs)
    ag_correct = ag_docs = 0
    for _, t, g, s, goal, exp in qs:
        r = agentic(docs, t, g, s, goal)
        ag_correct += score(r.answer, exp)
        ag_docs += r.docs_read
    assert one_shot_acc < 1.0            # one-shot at k=3 misses the multi-hops
    assert ag_correct == len(qs)         # agentic answers every question
    assert ag_docs / len(qs) < 5.83      # on less context than one-shot needs for 1.0


def test_budget_knee_is_three() -> None:
    docs = corpus()
    qs = questions()

    def accuracy(budget: int) -> float:
        return sum(
            score(agentic(docs, t, g, s, goal, budget=budget).answer, exp)
            for _, t, g, s, goal, exp in qs
        ) / len(qs)

    assert accuracy(1) < accuracy(2) < accuracy(3) == 1.0
    assert accuracy(4) == accuracy(3)    # no gain past the knee


def test_required_abstention_is_scored_correct() -> None:
    docs = corpus()
    budget_q = next(q for q in questions() if q[0] == "q6")
    _, text, groups, start, goal, expected = budget_q
    assert expected is None
    assert agentic(docs, text, groups, start, goal).answer is None
    assert score(agentic(docs, text, groups, start, goal).answer, expected)


def test_crag_separates_relevance_from_support() -> None:
    docs = corpus()
    # Passages relevant to "Falcon region" but supporting no such claim -> reformulate.
    for d in search(docs, "Which region is Project Falcon in?", ENG, k=3):
        v = assess(d, "Which region is Project Falcon in?", "Falcon", "region")
        assert v.relevant and not v.supported and v.action == "reformulate"
    # The passage that actually states Aurora's region -> cite.
    d03 = next(d for d in corpus() if d.id == "d03")
    v = assess(d03, "What region does Aurora run in?", "Aurora", "region")
    assert v.supported and v.action == "cite"


def test_graph_is_permission_scoped_and_multi_hop() -> None:
    docs = corpus()
    graph = build_graph(docs, ENG)
    answer, path = local_search(graph, "Falcon", "region")
    assert answer == "us-east-2"
    assert [step[3] for step in path] == ["d01", "d02", "d03"]  # provenance trail
    assert set(global_search(graph)) == {
        "depends_on", "owner", "telemetry_store", "region", "release_gate"
    }
    finance = build_graph(docs, frozenset({"finance"}))
    assert [e[1] for e in finance] == ["budget"]  # only the finance edge


def test_code_search_prefers_exact_symbols() -> None:
    repo = {
        "billing.py": "raise PaymentTimeout('gateway slow')",
        "retry.py": "# retry on PaymentTimeout",
        "notes.md": "the payment provider is slow to respond",
    }
    # An exact identifier resolves precisely and cheaply; a paraphrase does not.
    assert grep(repo, "PaymentTimeout") == ["billing.py", "retry.py"]
    assert grep(repo, "provider slow") == []


def test_long_context_costs_more_tokens_than_rag() -> None:
    docs = corpus()
    authorized = [d for d in docs if "engineering" in d.groups and d.trusted]
    long_ctx = sum(token_count(d.text) for d in authorized)
    rag = sum(token_count(d.text) for d in search(docs, REGION_Q, ENG, k=3) if d.trusted)
    assert rag < long_ctx


def test_indirect_injection_fires_naively_then_is_blocked() -> None:
    docs = corpus()
    attack_q = "What region does Aurora run in?"
    executed, naive_facts = naive_agent(docs, attack_q, ENG)
    # The naive reader obeys the injected directives and ingests the poison.
    assert any(did == "d10" for did, _ in executed)
    assert ("attacker-zone", "d10") in naive_facts

    blocked, hardened_facts = hardened_agent(docs, attack_q, ENG)
    assert blocked == ["d10"]
    # Only the verified region survives the integrity gate.
    assert hardened_facts == [("us-east-2", "d03")]
