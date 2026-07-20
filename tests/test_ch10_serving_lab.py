from __future__ import annotations

import math
import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch10"))

import openai_compatible_adapter  # noqa: E402
sys.modules.pop("render_metrics", None)
import render_metrics  # noqa: E402
import serving_lab as lab  # noqa: E402


class ServingLabTest(unittest.TestCase):
    def test_continuous_batching_improves_fixture_goodput(self) -> None:
        metrics = lab.run_experiment()["scheduling"]
        self.assertAlmostEqual(metrics["static"]["utilization"], sum(lab.REQUEST_LENGTHS) / (4 * 32))
        self.assertGreater(metrics["continuous"]["utilization"], metrics["static"]["utilization"])
        self.assertGreater(metrics["continuous_goodput"]["requests_per_step"],
                           metrics["static_goodput"]["requests_per_step"])

    def test_prefix_pages_share_only_under_complete_identity(self) -> None:
        probe = lab.prefix_cache_probe()
        self.assertTrue(probe["same_identity_reuses"])
        self.assertFalse(probe["cross_tenant_reuses"])
        self.assertEqual(probe["blocks_with_safe_sharing_two_branches"], 4)
        self.assertEqual(probe["tail_capacity_wasted_per_branch"], 3)

    def test_cache_breakeven_handles_economic_edges(self) -> None:
        self.assertAlmostEqual(lab.cache_breakeven(1.0, 1.25, 0.10), 5 / 18)
        self.assertEqual(lab.cache_breakeven(1.0, 1.25, 1.0), math.inf)

    def test_speculation_preserves_target_but_can_lose_under_contention(self) -> None:
        probe = lab.speculation_probe()
        self.assertLess(probe["target_total_variation"], 0.01)
        self.assertAlmostEqual(probe["acceptance_probability"], 0.9)
        self.assertGreater(probe["speedup_low_load"], 1)
        self.assertLess(probe["speedup_saturated"], 1)

    def test_constraints_and_quantization_expose_distortion(self) -> None:
        constrained = lab.constrained_distribution([0.60, 0.25, 0.15], {1, 2})
        self.assertAlmostEqual(sum(constrained["distribution"]), 1)
        self.assertAlmostEqual(constrained["removed_mass"], 0.6)
        rows = lab.quantization_probe()
        self.assertLess(rows[0]["rmse"], rows[-1]["rmse"])
        self.assertGreater(rows[0]["effective_bits_per_value"], rows[-1]["effective_bits_per_value"])

    def test_disaggregation_budget_adapter_and_svg(self) -> None:
        disaggregation = lab.disaggregation_probe()
        self.assertFalse(disaggregation[0]["disaggregate"])
        self.assertTrue(disaggregation[1]["disaggregate"])
        budgets = lab.reasoning_budget_probe()
        self.assertLess(budgets[-1]["synthetic_score"], budgets[-2]["synthetic_score"])
        self.assertTrue(callable(openai_compatible_adapter.profile_request))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "serving.svg"
            render_metrics.render(output)
            svg = output.read_text(encoding="utf-8")
            self.assertIn("Scheduling under one arrival burst", svg)
            self.assertIn("All panels are analytical or synthetic", svg)


if __name__ == "__main__":
    unittest.main()
