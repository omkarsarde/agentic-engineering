"""Executable invariants for the Chapter 20 multi-agent teaching code.

Imports the tangled module ``code/ch20/_generated.py`` (produced from the
chapter's ``# @save`` cells by ``scripts/tangle.py``) and checks the real
properties the chapter claims: the three-condition split gate, the cost-matched
A/B (same answer, a measured token multiplier above one, a latency speedup above
one, and the team spending more tokens than the lean single agent), that the
term-by-term cost breakdown matches the orchestrator's total, that a worker with
an unauthorized tool is rejected, that reports are immutable single-writer
artifacts, that a missing-provenance fault is contained and attributed while a
disabled gate launders it, that the single agent fails closed under budget, and
that delta-utility flips sign with the product's latency weight.

The module is loaded under a unique name (``ch20_generated``) and registered in
``sys.modules`` before execution so the dataclasses' string annotations resolve
and so the bare ``_generated`` name never collides across chapters in one pytest
process.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch20_generated", ROOT / "code" / "ch20" / "_generated.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ch20 = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("ch20_generated", ch20)
_SPEC.loader.exec_module(ch20)

CORPUS = ch20.CORPUS
OBJECTIVE = ch20.OBJECTIVE
QUERY = ch20.QUERY
Worker = ch20.Worker
Workspace = ch20.Workspace
Orchestrator = ch20.Orchestrator
TaskBrief = ch20.TaskBrief
should_split = ch20.should_split
single_agent = ch20.single_agent
delta_utility = ch20.delta_utility
run_experiment = ch20.run_experiment
cost_breakdown = ch20.cost_breakdown
majority_vote_accuracy = ch20.majority_vote_accuracy


def test_split_requires_all_three_conditions() -> None:
    assert should_split(decomposable=True, useful_boundary=True, measured_failure=True)
    assert not should_split(decomposable=False, useful_boundary=True, measured_failure=True)
    assert not should_split(decomposable=True, useful_boundary=False, measured_failure=True)
    assert not should_split(decomposable=True, useful_boundary=True, measured_failure=False)


def test_cost_matched_same_answer_and_ratios() -> None:
    r = run_experiment(CORPUS)
    assert r["same_answer"] is True
    assert r["team"]["status"] == "completed"
    assert r["single"]["status"] == "completed"
    assert r["token_multiplier"] > 1.0          # coordination costs tokens
    assert r["latency_speedup"] > 1.0           # parallelism buys latency


def test_team_spends_more_tokens_than_single() -> None:
    r = run_experiment(CORPUS)
    assert r["team"]["tokens"] > r["single"]["tokens"]


def test_cost_breakdown_matches_orchestrator_total() -> None:
    b = cost_breakdown(OBJECTIVE, CORPUS)
    r = run_experiment(CORPUS)
    assert b["team"] == r["team"]["tokens"]
    assert b["single"] == r["single"]["tokens"]
    assert b["plan"] + b["workers"] + b["merge"] == b["team"]


def test_missing_provenance_is_contained_and_attributed() -> None:
    fault = run_experiment(CORPUS, poison_worker="worker-2")["fault"]
    assert fault["status"] == "contained"
    assert fault["answer"] is None
    assert fault["failure"]["category"] == "inter-agent misalignment"
    assert fault["failure"]["agent"] == "worker-2"
    assert fault["failure"]["step"] == "handoff"
    assert fault["failure"]["reason"] == "missing provenance"


def test_disabled_gate_launders_unsupported_finding(tmp_path) -> None:
    ws = Workspace(tmp_path / "nogate")
    res = Orchestrator().run(OBJECTIVE, CORPUS, ws, poison_worker="worker-2", verify=False)
    assert res.status == "completed"     # no gate -> the poisoned finding reaches the answer
    assert res.answer is not None


def test_worker_rejects_unauthorized_tool() -> None:
    brief = TaskBrief("worker-x", "americas", QUERY, ("search", "write"), "WorkerReport/v1", 200)
    with pytest.raises(PermissionError):
        Worker().run(brief, CORPUS["americas"])


def test_reports_are_immutable_single_writer_artifacts(tmp_path) -> None:
    ws = Workspace(tmp_path / "ws")
    res = Orchestrator().run(OBJECTIVE, CORPUS, ws)
    assert res.status == "completed"
    files = list((tmp_path / "ws").glob("*.json"))
    assert len(files) == len(CORPUS)
    # A second write to the same worker path is refused (immutable handoff).
    report = Worker().run(TaskBrief("worker-1", "americas", QUERY, ("search",), "WorkerReport/v1", 200),
                          CORPUS["americas"])
    with pytest.raises(FileExistsError):
        ws.write(report)


def test_single_agent_fails_closed_below_budget() -> None:
    res = single_agent(OBJECTIVE, CORPUS, token_budget=5)
    assert res.status == "budget_exhausted"
    assert res.answer is None


def test_delta_utility_flips_with_latency_weight() -> None:
    r = run_experiment(CORPUS)
    d_cost = r["team"]["tokens"] - r["single"]["tokens"]
    d_latency = r["team"]["latency"] - r["single"]["latency"]
    latency_bound = delta_utility(d_quality=0.0, d_cost=d_cost, d_latency=d_latency, d_risk=1.0,
                                  w_cost=0.1, w_latency=1.0, w_risk=0.5)
    latency_free = delta_utility(d_quality=0.0, d_cost=d_cost, d_latency=d_latency, d_risk=1.0,
                                 w_cost=0.1, w_latency=0.1, w_risk=0.5)
    assert latency_bound > 0          # a latency-bound product should build the team
    assert latency_free < 0           # a latency-insensitive product should not


def test_speedup_grows_with_team_size_multiplier_stays_bounded() -> None:
    def make_corpus(n: int) -> dict:
        base = list(CORPUS.values())
        return {f"r{i}": [dict(doc, id=f"r{i}-{doc['id']}") for doc in base[i % len(base)]]
                for i in range(n)}

    small = run_experiment(make_corpus(2))
    large = run_experiment(make_corpus(8))
    # More parallel workers widen the latency win (max stays flat, sum grows) ...
    assert large["latency_speedup"] > small["latency_speedup"]
    # ... while the token multiplier stays a bounded, roughly flat coordination tax.
    assert small["token_multiplier"] > 1.0
    assert large["token_multiplier"] > 1.0


def test_ensemble_helps_only_when_uncorrelated() -> None:
    p = 0.65
    independent = majority_vote_accuracy(5, p, correlation=0.0)
    correlated = majority_vote_accuracy(5, p, correlation=1.0)
    assert independent > p + 0.05        # aggregating independent voters beats one voter
    assert abs(correlated - p) < 0.05    # fully correlated voters give no gain
