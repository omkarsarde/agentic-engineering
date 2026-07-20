from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch31"))

from embodied_eval import (
    MiniTabletopEnv,
    ReversibilityGate,
    ScriptedPolicy,
    SimplerEnvAdapter,
    detokenize_action,
    endpoint_success,
    plot_horizon,
    roundtrip_error,
    run_episode,
    run_suite,
    tokenize_action,
)


class EmbodiedEvalTest(unittest.TestCase):
    def test_action_token_roundtrip_respects_half_bin_error(self):
        action = np.linspace(-0.93, 0.91, 7)
        tokens = tokenize_action(action)
        self.assertTrue(np.allclose(detokenize_action(tokens), action, atol=1 / 255))
        self.assertLessEqual(roundtrip_error(action), 1 / 255)

    def test_action_tokenizer_rejects_wrong_contract(self):
        with self.assertRaises(ValueError):
            tokenize_action(np.zeros(6))
        with self.assertRaises(ValueError):
            tokenize_action(np.array([0, 0, 0, 0, 0, 0, 1.1]))

    def test_replanning_beats_open_loop_at_long_horizon(self):
        open_loop = endpoint_success(24, 0.012, False)
        replanned = endpoint_success(24, 0.012, True)
        self.assertLess(open_loop, 0.1)
        self.assertGreater(replanned, 0.95)

    def test_replanning_does_not_rescue_a_bad_model(self):
        self.assertLess(endpoint_success(24, 0.20, True), 0.2)

    def test_scripted_fallback_is_reproducible_and_final_state_graded(self):
        first = run_suite(MiniTabletopEnv(), ScriptedPolicy(), episodes=10)
        second = run_suite(MiniTabletopEnv(), ScriptedPolicy(), episodes=10)
        self.assertEqual(first["successes"], 10)
        self.assertEqual(first["grader"], "environment final-state predicate")
        self.assertEqual(first["mean_steps"], second["mean_steps"])

    def test_simpler_adapter_uses_environment_done_not_dense_reward(self):
        class FakeEnv:
            calls = 0

            def step(self, action):
                self.calls += 1
                return {"frame": self.calls}, 1.0, self.calls == 2, False, {}

        adapter = SimplerEnvAdapter.__new__(SimplerEnvAdapter)
        adapter.env = FakeEnv()
        adapter.image_from_obs = lambda env, obs: np.array([obs["frame"]])
        first = adapter.step(np.zeros(7))
        second = adapter.step(np.zeros(7))
        self.assertEqual(first[1:], (False, False))
        self.assertEqual(second[1:], (True, True))

    def test_denied_contact_action_stops_before_success(self):
        report = run_episode(MiniTabletopEnv(), ScriptedPolicy(), 0, gate=ReversibilityGate())
        self.assertFalse(report["success"])
        self.assertEqual(report["termination"], "gate_denied")

    def test_approved_contact_action_is_counted_and_completes(self):
        gate = ReversibilityGate()
        report = run_episode(
            MiniTabletopEnv(), ScriptedPolicy(), 0, gate=gate, approve=lambda proposal: True)
        self.assertTrue(report["success"])
        self.assertGreaterEqual(gate.hits, 1)

    def test_numeric_figure_is_generated_from_the_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "horizon.svg"
            plot_horizon(path)
            text = path.read_text(encoding="utf-8")
            self.assertIn("Planning horizon", text)
            self.assertIn("observe + replan", text)


if __name__ == "__main__":
    unittest.main()
