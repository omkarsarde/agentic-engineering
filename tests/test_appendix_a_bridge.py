"""Tests for the Appendix A bridge, importing the tangled ``code/appa/_generated.py``.

The appendix teaches probability, decision theory, causal inference, optimization,
search, RL vocabulary, control, and distributed systems through small executed
demos. These tests pin the load-bearing numbers each demo prints, so the prose
and the tangled module cannot drift apart.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE = Path(__file__).parents[1] / "code" / "appa" / "_generated.py"
_spec = importlib.util.spec_from_file_location("appa_generated", MODULE)
appa = importlib.util.module_from_spec(_spec)
sys.modules["appa_generated"] = appa  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(appa)


# --- probability and information theory ---

def test_base_rate_posterior() -> None:
    assert round(appa.posterior(0.02, 0.90, 0.05), 3) == 0.269


def test_entropy_and_kl_asymmetry() -> None:
    peaked, uniform = [0.90, 0.05, 0.05], [1 / 3, 1 / 3, 1 / 3]
    assert round(appa.entropy(uniform), 3) == 1.585
    assert appa.entropy(peaked) < appa.entropy(uniform)
    forward = appa.kl_divergence(peaked, uniform)
    reverse = appa.kl_divergence(uniform, peaked)
    assert forward > 0 and reverse > 0
    assert abs(forward - reverse) > 0.1  # KL is not symmetric


# --- decision theory ---

def test_asymmetric_loss_prefers_escalation() -> None:
    assert appa.best_expected_loss(0.2, 10, 1) == 0.8  # min(0.8, 2.0)


def test_value_of_information_and_posteriors() -> None:
    result = appa.value_of_signal(0.2, 0.85, 0.9, 10.0, 1.0)
    assert round(result["voi"], 3) == 0.42
    assert round(result["p_positive"], 3) == 0.25
    assert round(result["posterior_positive"], 2) == 0.68
    assert round(result["posterior_negative"], 2) == 0.04


def test_proper_scores_reward_honesty() -> None:
    true_p = 0.3

    def expected(loss, report):
        return true_p * loss(report, 1) + (1 - true_p) * loss(report, 0)

    honest_brier = expected(appa.brier, 0.3)
    honest_log = expected(appa.log_loss, 0.3)
    for report in (0.5, 0.1):
        assert expected(appa.brier, report) > honest_brier
        assert expected(appa.log_loss, report) > honest_log


# --- causal inference ---

def test_confounding_flips_the_sign() -> None:
    logs = appa.simulate_routing(20_000, seed=0)
    naive = appa.naive_effect(logs)
    adjusted = appa.backdoor_effect(logs)
    assert naive < -0.15  # confounded: agent looks worse
    assert 0.07 < adjusted < 0.11  # deconfounded recovers the true +0.10


def test_ips_large_weight_exceeds_reward_range() -> None:
    logged = [(1, 1.0, 0.2), (0, 1.0, 0.8), (1, 0.0, 0.8), (0, 0.0, 0.2), (1, 1.0, 0.8)]
    assert round(appa.ips_value(logged), 3) == 1.25


# --- optimization ---

def test_pareto_removes_dominated_configuration() -> None:
    frontier = appa.pareto_names(appa.POINTS)
    assert "medium/direct" not in frontier
    assert "small/direct" in frontier
    assert "large/ensemble" in frontier
    assert len(frontier) == 5


# --- search ---

def test_astar_optimal_and_cheaper_than_uninformed() -> None:
    informed_path, informed_seen = appa.astar(appa.manhattan)
    _, blind_seen = appa.astar(lambda a, b: 0)
    assert len(informed_path) - 1 == 8  # optimal path length
    assert len(informed_seen) == 15
    assert len(blind_seen) == 18
    assert len(informed_seen) < len(blind_seen)


# --- MDP / info gain / bandit ---

def test_value_iteration_propagates_reward_backward() -> None:
    history = appa.value_iteration(3)
    assert history[-1] == [0.81, 0.9, 1.0]


def test_clarifying_question_information_gain() -> None:
    result = appa.clarifying_info_gain(0.5, 0.9, 0.2)
    assert round(result["p_yes"], 2) == 0.55
    assert round(result["info_gain_bits"], 2) == 0.40
    assert 0 < result["info_gain_bits"] < 1  # never fully resolves the intent


def test_greedy_bandit_has_higher_regret_than_ucb() -> None:
    ucb = appa.bandit_regret("ucb", seed=1)
    greedy = appa.bandit_regret("greedy", seed=1)
    assert ucb < greedy


# --- control ---

def test_control_loop_stability_regimes() -> None:
    stable = appa.error_dynamics(0.5, 0)
    diverging = appa.error_dynamics(1.3, 1)
    assert abs(stable[-1]) < 0.01  # decays to the setpoint
    assert abs(diverging[-1]) > 10  # grows without bound under lag + high gain


# --- distributed systems ---

def test_delivery_stream_is_at_least_once() -> None:
    deliveries = appa.fixture_deliveries()
    assert len(deliveries) == 24
    assert len({task.key for task in deliveries}) == 20


def test_break_it_lab_counts() -> None:
    report = appa.build_report()
    assert report["naive"]["effects"] == 24
    assert report["local_ledger"]["effects"] == 20
    assert report["ambiguous_window"]["calls"] == 21
    assert report["ambiguous_window"]["effects"] == 21
    assert report["provider_key"]["calls"] == 21
    assert report["provider_key"]["effects"] == 20
