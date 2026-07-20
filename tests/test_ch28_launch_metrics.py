from copy import deepcopy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch28"))

from launch_metrics import (
    build_launch_review,
    economics_report,
    expand_logs,
    load_fixture,
    validate_fixture,
)


class LaunchMetricsTest(unittest.TestCase):
    def test_integrated_review_holds_full_launch(self):
        report = build_launch_review()
        self.assertEqual(report["recommendation"], "HOLD_FULL_LAUNCH")
        self.assertEqual(
            report["failed_conditions"],
            ["appropriate_reliance", "cohort_overreliance", "assistive_completion"],
        )

    def test_fixture_expands_to_two_500_row_logs(self):
        decisions, usage = expand_logs(load_fixture())
        self.assertEqual(len(decisions), 500)
        self.assertEqual(len(usage), 500)

    def test_reliance_quadrants_are_exhaustive(self):
        report = build_launch_review()["reliance"]["overall"]
        self.assertEqual(sum(report["quadrants"].values()), report["n"])
        self.assertEqual(report["quadrants"], {
            "correct_accept": 395,
            "correct_reject": 25,
            "wrong_accept": 34,
            "wrong_reject": 46,
        })

    def test_new_reviewers_show_heavy_overreliance(self):
        cohort = build_launch_review()["reliance"]["by_cohort"]["new_reviewer"]
        self.assertEqual(cohort["overreliance_rate"], 0.8333)
        self.assertLess(cohort["reject_when_wrong"], 0.2)

    def test_funnel_is_nested_and_accessibility_slice_is_visible(self):
        report = build_launch_review()["funnel"]
        counts = list(report["overall"]["counts"].values())
        self.assertEqual(counts, [500, 460, 420, 380, 345, 295])
        assistive = report["by_cohort"]["assistive_technology"]
        self.assertEqual(assistive["conversion_from_previous"]["completed"], 0.7647)

    def test_unit_economics_include_review_and_remediation(self):
        economics = build_launch_review()["economics"]
        self.assertEqual(economics["cost_components"]["review_labor"], 310.0)
        self.assertEqual(economics["cost_components"]["remediation"], 136.0)
        self.assertLess(economics["cost_per_outcome"], economics["baseline_cost_per_outcome"])
        self.assertLess(economics["average_review_seconds"], economics["break_even_review_seconds"])

    def test_report_is_reproducible(self):
        self.assertEqual(build_launch_review(), build_launch_review())

    def test_increasing_funnel_is_rejected(self):
        fixture = deepcopy(load_fixture())
        fixture["cohorts"]["experienced"]["funnel"]["completed"] = 191
        with self.assertRaisesRegex(ValueError, "never increase"):
            validate_fixture(fixture)


if __name__ == "__main__":
    unittest.main()
