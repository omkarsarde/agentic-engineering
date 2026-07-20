"""Focused tests for the Chapter 22 evaluation artifact."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch22"))

import eval_harness as harness


class EvalHarnessTests(unittest.TestCase):
    def test_trace_fixture_expands_to_isolated_trials(self) -> None:
        rows = harness.grade_traces()
        self.assertEqual(len(rows), 96)
        self.assertEqual(len({row["environment_id"] for row in rows}), 96)

    def test_unsafe_success_does_not_pass(self) -> None:
        rows = harness.grade_traces()
        row = next(
            item
            for item in rows
            if item["task_id"] == "refund-en"
            and item["system"] == "candidate"
            and item["trial"] == 2
        )
        self.assertTrue(row["state_pass"])
        self.assertFalse(row["policy_pass"])
        self.assertFalse(row["success"])

    def test_judge_calibration_fixture(self) -> None:
        report = harness.judge_report()
        self.assertEqual(report["n"], 100)
        self.assertAlmostEqual(report["agreement"], 0.89)
        self.assertAlmostEqual(report["kappa"], 0.7355769230769231)
        self.assertAlmostEqual(report["fail_recall"], 0.80)
        self.assertAlmostEqual(report["position_flip_rate"], 0.12)

    def test_capability_and_reliability_estimators(self) -> None:
        self.assertAlmostEqual(harness.pass_at_k_estimate(4, 2, 2), 5 / 6)
        self.assertAlmostEqual(harness.pass_pow_k_estimate(4, 2, 2), 1 / 6)
        self.assertEqual(harness.pass_pow_k_estimate(4, 1, 2), 0.0)

    def test_task_cluster_is_the_unit_of_uncertainty(self) -> None:
        rows = harness.grade_traces()
        baseline = harness.task_metrics(rows, "baseline")
        candidate = harness.task_metrics(rows, "candidate")
        result = harness.paired_cluster_uncertainty(baseline, candidate)
        self.assertAlmostEqual(result["point"], 0.125)
        self.assertAlmostEqual(result["low"], -0.020833333333333332)
        self.assertAlmostEqual(result["high"], 0.22916666666666666)
        self.assertEqual(len(result["task_deltas"]), 12)

    def test_slices_and_golden_case_both_survive_aggregation(self) -> None:
        report = harness.release_report()
        self.assertEqual(report["slices"]["candidate"]["mixed-language"], 0.8125)
        self.assertEqual(report["verdict"], "BLOCK")
        self.assertIn("must-not-break regression: refund-en", report["reasons"])

    def test_trajectory_measure_is_separate_from_outcome(self) -> None:
        report = harness.release_report()
        self.assertGreater(
            report["trajectory_f1"]["candidate"],
            report["trajectory_f1"]["baseline"],
        )
        self.assertEqual(report["verdict"], "BLOCK")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_plot_functions_return_source_backed_figures(self) -> None:
        reliability = harness.plot_reliability()
        release = harness.plot_release(harness.release_report())
        self.assertEqual(len(reliability.axes), 2)
        self.assertEqual(len(release.axes), 2)


if __name__ == "__main__":
    unittest.main()
