"""Executable invariants for Chapter 23's agent-RL training stack.

Imports only the tangled module ``code/ch23/_generated.py`` (the chapter's
``# @save`` cells in document order) under a chapter-unique module name.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch23_generated", ROOT / "code" / "ch23" / "_generated.py"
)
ch23 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["ch23_generated"] = ch23  # dataclasses resolve annotations via sys.modules
_SPEC.loader.exec_module(ch23)


def make_bc_policy() -> "ch23.SoftmaxPolicy":
    policy = ch23.SoftmaxPolicy(ch23.ACTIONS)
    ch23.fit(policy, ch23.expert_examples(ch23.EXPERT_TASKS))
    return policy


def test_softmax_normalizes_and_survives_large_logits() -> None:
    probs = ch23.softmax([1000.0, 1000.0, 998.0, 990.0])
    assert abs(sum(probs) - 1.0) < 1e-12
    assert probs[0] == probs[1] > probs[2] > probs[3] > 0.0


def test_update_is_a_score_function_step() -> None:
    policy = ch23.SoftmaxPolicy(("a", "b"))
    before = policy.probabilities("s")[0]
    policy.update("s", "a", 0.5)
    up = policy.probabilities("s")[0]
    policy.update("s", "a", -1.0)
    down = policy.probabilities("s")[0]
    assert up > before
    assert down < up


def test_sample_records_the_behavior_log_probability() -> None:
    policy = ch23.SoftmaxPolicy(("a", "b", "c"))
    policy.update("s", "b", 1.0)
    action, logp = policy.sample("s", random.Random(0))
    assert math.isclose(logp, policy.logp("s", action))


def test_gym_safe_paths_for_low_and_high_amounts() -> None:
    for amount, actions in ((20, ["lookup", "refund", "finish"]),
                            (80, ["lookup", "request_approval", "refund", "finish"])):
        gym = ch23.RefundGym()
        gym.reset(ch23.Task("t", amount))
        for action in actions:
            gym.step(action)
        assert gym.done and gym.refunded
        assert not gym.violation and gym.effects == 1


def test_gym_shortcut_is_permitted_but_flagged() -> None:
    gym = ch23.RefundGym()
    gym.reset(ch23.Task("t", 80))
    for action in ("lookup", "refund", "finish"):
        gym.step(action)
    assert gym.done and gym.refunded
    assert gym.violation


def test_gym_duplicate_refund_is_detectable() -> None:
    gym = ch23.RefundGym()
    gym.reset(ch23.Task("t", 20))
    for action in ("lookup", "refund", "refund", "finish"):
        gym.step(action)
    assert gym.effects == 2


def test_gym_invalid_actions_land_in_recover_and_terminal_raises() -> None:
    gym = ch23.RefundGym()
    assert gym.reset(ch23.Task("t", 20)) == "start"
    assert gym.step("refund") == "recover"
    gym.step("finish")
    try:
        gym.step("lookup")
    except RuntimeError:
        pass
    else:
        raise AssertionError("step after terminal state must raise")


def test_gym_reset_isolates_episodes() -> None:
    gym = ch23.RefundGym()
    gym.reset(ch23.Task("one", 20))
    for action in ("lookup", "refund", "finish"):
        gym.step(action)
    assert gym.refunded
    gym.reset(ch23.Task("two", 80))
    assert not gym.refunded and not gym.violation and gym.effects == 0


def test_expert_demonstrations_never_visit_recover() -> None:
    examples = ch23.expert_examples(ch23.EXPERT_TASKS)
    states = {state for state, _ in examples}
    assert "recover" not in states
    assert {"start", "known_low", "known_high", "approved", "refunded"} <= states


def test_behavior_cloning_learns_covered_states_but_not_recover() -> None:
    policy = make_bc_policy()
    assert policy.greedy("known_high") == "request_approval"
    assert policy.probabilities("recover") == [0.25, 0.25, 0.25, 0.25]
    rows = ch23.evaluate(policy, ch23.EVAL_TASKS)
    assert sum(row["success"] for row in rows) == 2
    failing = [row for row in rows if not row["success"]]
    assert all(row["actions"] == ["finish"] for row in failing)


def test_one_dagger_round_closes_the_recovery_gap() -> None:
    policy = make_bc_policy()
    labels = ch23.dagger_labels(policy, ch23.TRAIN_TASKS[2:])
    assert labels and all(label == ("recover", "lookup") for label in labels)
    ch23.fit(policy, ch23.expert_examples(ch23.EXPERT_TASKS) + labels)
    assert sum(row["success"] for row in ch23.evaluate(policy, ch23.EVAL_TASKS)) == 4


def test_grouped_advantages_center_within_each_anchor_group() -> None:
    policy = make_bc_policy()
    rng = random.Random(11)
    batch = [ch23.rollout(policy, ch23.Task("same", 90, "recover"), rng) for _ in range(12)]
    advantages = ch23.grouped_advantages(batch, ch23.outcome_score)
    per_group: dict[tuple[str, str], float] = {}
    for index, trajectory in enumerate(batch):
        for turn, transition in enumerate(trajectory.transitions):
            key = (trajectory.task_id, transition.state)
            per_group[key] = per_group.get(key, 0.0) + advantages[(index, turn)]
    for total in per_group.values():
        assert abs(total) < 1e-6


def test_clipped_update_has_zero_gradient_outside_the_fence() -> None:
    policy = ch23.SoftmaxPolicy(ch23.ACTIONS)
    stale = ch23.Transition("start", "lookup", math.log(0.05))  # ratio = 5
    before = list(policy.probabilities("start"))
    assert ch23.clipped_update(policy, stale, advantage=1.0) is False
    assert policy.probabilities("start") == before


def test_clipped_update_applies_inside_the_fence() -> None:
    policy = ch23.SoftmaxPolicy(ch23.ACTIONS)
    fresh = ch23.Transition("start", "lookup", policy.logp("start", "lookup"))
    before = policy.probabilities("start")[ch23.ACTIONS.index("lookup")]
    assert ch23.clipped_update(policy, fresh, advantage=1.0) is True
    assert policy.probabilities("start")[ch23.ACTIONS.index("lookup")] > before


def test_outcome_only_reward_trains_the_unauthorized_shortcut() -> None:
    policy = make_bc_policy()
    ch23.train_group_relative(policy, ch23.TRAIN_TASKS, ch23.outcome_score,
                              random.Random(7))
    rows = {row["task"]: row for row in ch23.evaluate(policy, ch23.EVAL_TASKS)}
    assert rows["eval-high"]["success"] and rows["eval-high"]["violation"]
    assert rows["eval-high"]["actions"] == ["lookup", "refund", "finish"]


def test_guarded_reward_closes_the_gap_without_violations() -> None:
    policy = make_bc_policy()
    history, totals = ch23.train_group_relative(
        policy, ch23.TRAIN_TASKS, ch23.guarded_score, random.Random(7))
    rows = ch23.evaluate(policy, ch23.EVAL_TASKS)
    assert sum(row["success"] for row in rows) == 4
    assert not any(row["violation"] for row in rows)
    assert totals["vetoed_rollouts"] > 0
    assert totals["vetoed_updates"] == 0
    assert history["clipped"][1] > history["clipped"][0]  # second pass clips more


def test_training_is_deterministic_under_a_fixed_seed() -> None:
    runs = []
    for _ in range(2):
        policy = make_bc_policy()
        runs.append(ch23.train_group_relative(policy, ch23.TRAIN_TASKS,
                                              ch23.guarded_score, random.Random(7)))
    assert runs[0] == runs[1]


def test_scorers_disagree_only_on_permission_semantics() -> None:
    gym = ch23.RefundGym()
    gym.reset(ch23.Task("t", 80))
    trajectory = ch23.Trajectory("t")
    state = "start"
    for action in ("lookup", "refund", "finish"):
        trajectory.transitions.append(ch23.Transition(state, action, 0.0))
        state = gym.step(action)
    trajectory.success = gym.done and gym.refunded
    trajectory.violation = gym.violation
    assert ch23.outcome_score(trajectory).eligible is True
    guarded = ch23.guarded_score(trajectory)
    assert guarded.eligible is False
    assert guarded.reward < 0 < ch23.outcome_score(trajectory).reward


def test_throughput_model_puts_the_budget_in_the_environment() -> None:
    totals = {"transitions": 3444, "updates": 6556}
    phases = ch23.modelled_phase_seconds(totals)
    assert phases["environment"] > phases["sampling"] > phases["update"]
    speeds = [ch23.transitions_per_second(totals, n) for n in (1, 4, 16, 64)]
    assert speeds == sorted(speeds)  # parallelism monotonically helps
    ceiling = totals["transitions"] / (phases["sampling"] + phases["update"])
    assert all(speed < ceiling for speed in speeds)
