"""Focused invariants for the deterministic Chapter 7 build."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch07"))

from posttrain_lab import run_experiment, write_svg  # noqa: E402


class PostTrainingLabTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_experiment()

    def test_pipeline_improves_specification_utility(self) -> None:
        stages = self.result["stages"]
        self.assertEqual(stages["base"]["spec_top1"], 0.0)
        self.assertEqual(stages["sft"]["spec_top1"], 1.0)
        self.assertGreater(stages["dpo"]["expected_utility"], stages["sft"]["expected_utility"])

    def test_biases_are_visible(self) -> None:
        self.assertGreater(self.result["left_choice_rate"], .58)
        self.assertGreater(self.result["longer_choice_rate"], .63)
        self.assertGreater(self.result["reward_weights"]["length"], .8)
        self.assertGreater(
            self.result["stages"]["dpo"]["normalized_length"],
            self.result["stages"]["sft"]["normalized_length"],
        )

    def test_reinforce_step_raises_rewarded_action_probability(self) -> None:
        check = self.result["reinforce_probability"]
        self.assertGreater(check["after"], check["before"])

    def test_svg_is_generated_from_the_same_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "figure.svg"
            write_svg(self.result, output)
            text = output.read_text(encoding="utf-8")
            self.assertIn(f'{self.result["stages"]["dpo"]["expected_utility"]:.3f}', text)
            self.assertIn("Expected normalized length", text)


if __name__ == "__main__":
    unittest.main()
