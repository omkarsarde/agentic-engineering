import json
from pathlib import Path
import random
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch23"))

from agent_gym import (
    RefundGym,
    TabularPolicy,
    Task,
    Trajectory,
    behavior_clone,
    evaluate,
    grouped_advantages,
    rollout,
    run_experiment,
    score_trajectory,
)


class AgentGymTest(unittest.TestCase):
    def test_integrated_experiment_closes_the_state_coverage_gap(self):
        result = run_experiment()
        self.assertEqual(result["bc"].get("successes"), 2)
        self.assertEqual(result["bc_plus_rl"].get("successes"), 4)
        self.assertEqual(result["training"].get("vetoed_updates"), 0)
        self.assertEqual(result["training"].get("max_policy_lag"), 0)
        self.assertEqual(result["training"].get("vetoed_rollouts"), 22)
        units = result["training"].get("work_units")
        self.assertEqual(units, {"environment": 30682, "policy_sampling": 12267, "update": 7774})
        self.assertGreater(units["environment"], units["policy_sampling"])
        self.assertGreater(units["policy_sampling"], units["update"])
        json.dumps(result)

    def test_default_experiment_is_reproducible(self):
        self.assertEqual(run_experiment(), run_experiment())

    def test_behavior_cloning_does_not_cover_recovery_state(self):
        policy = TabularPolicy()
        behavior_clone(policy, [Task("low", 20), Task("high", 80)])
        result = evaluate(policy, [Task("recover", 20, "recover")])
        self.assertEqual(result["successes"], 0)
        self.assertEqual(result["actions"]["recover"], ["finish"])

    def test_successful_policy_violation_is_vetoed(self):
        gym = RefundGym()
        gym.reset(Task("shortcut", 80))
        for action in ["lookup", "refund", "finish"]:
            gym.step(action)
        trajectory = Trajectory("shortcut", success=gym.refunded, policy_violation=gym.violation)
        score = score_trajectory(trajectory)
        self.assertTrue(trajectory.success)
        self.assertFalse(score.eligible)
        self.assertLess(score.reward, 0)

    def test_grouped_advantage_centers_each_repeated_state(self):
        policy = TabularPolicy()
        trajectories = [rollout(policy, Task("same", 20, "recover"), random.Random(seed)) for seed in range(12)]
        advantages = grouped_advantages(trajectories)
        recovery_visits = [
            advantages[(index, turn)]
            for index, trajectory in enumerate(trajectories)
            for turn, transition in enumerate(trajectory.transitions)
            if transition.observation == "recover"
        ]
        self.assertAlmostEqual(sum(recovery_visits), 0.0, places=5)

    def test_reset_removes_prior_effects(self):
        gym = RefundGym()
        gym.reset(Task("one", 20))
        for action in ["lookup", "refund", "finish"]:
            gym.step(action)
        self.assertTrue(gym.refunded)
        gym.reset(Task("two", 80))
        self.assertFalse(gym.refunded)
        self.assertEqual(gym.effects, 0)


if __name__ == "__main__":
    unittest.main()
