"""Focused invariants for the Chapter 8 reasoning and RLVR build."""

from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch08"))

from hf_generate_adapter import extract_final_integer  # noqa: E402
from reason_rl import grpo_advantages, run_experiment  # noqa: E402
sys.modules.pop("render_metrics", None)
from render_metrics import write_svg  # noqa: E402


class ReasonRlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_experiment()

    def test_coverage_and_verifier_ceiling(self) -> None:
        curve = self.result["ttc"]
        coverage = [row["coverage"] for row in curve]
        self.assertEqual(coverage, sorted(coverage))
        self.assertGreater(coverage[-1], .99)
        self.assertLess(curve[-1]["noisy_bon"], .8)
        self.assertLess(curve[-1]["plurality"], curve[-1]["coverage"])

    def test_group_advantages(self) -> None:
        advantages = grpo_advantages([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(advantages), 0.0)
        self.assertGreater(advantages[0], 0.0)
        self.assertEqual(grpo_advantages([.5, .5, .5]), [0.0, 0.0, 0.0])

    def test_exact_rlvr_learns_and_loses_entropy(self) -> None:
        history = self.result["training"]["exact"]
        self.assertGreater(history[-1]["true_accuracy"], .95)
        self.assertLess(history[-1]["normalized_entropy"], history[0]["normalized_entropy"])

    def test_proxy_run_exhibits_goodhart_failure(self) -> None:
        history = self.result["training"]["proxy"]
        self.assertGreater(max(row["true_accuracy"] for row in history), .6)
        self.assertLess(history[-1]["true_accuracy"], .1)
        self.assertGreater(history[-1]["normalized_objective"], .98)
        self.assertGreater(history[-1]["mean_tokens"], 25)

    def test_adapter_extraction_and_numeric_svg(self) -> None:
        self.assertEqual(extract_final_integer("work... final: -1,234"), -1234)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.svg"
            write_svg(path)
            text = path.read_text(encoding="utf-8")
            self.assertIn("Inference accuracy vs generated tokens", text)
            self.assertIn("proxy objective / maximum", text)


if __name__ == "__main__":
    unittest.main()
