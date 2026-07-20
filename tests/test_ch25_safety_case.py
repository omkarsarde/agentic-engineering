from copy import deepcopy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch25"))

from safety_case import (
    build_safety_case,
    diagnostic_evidence,
    evaluate_claim,
    interpretability_diagnostics,
    load_case,
    validate_case,
)


class SafetyCaseTest(unittest.TestCase):
    def test_integrated_case_fails_closed(self):
        report = build_safety_case()
        self.assertEqual(report["decision"], "BLOCK")
        self.assertEqual(report["summary"]["supported"], 1)
        self.assertEqual(report["summary"]["gaps"], 2)
        self.assertEqual(report["summary"]["blocking_claims"], ["G2", "G3"])

    def test_report_is_reproducible(self):
        self.assertEqual(build_safety_case(), build_safety_case())

    def test_lens_signal_strengthens_with_depth(self):
        margins = interpretability_diagnostics()["lens_margin"]
        self.assertGreater(margins[-1], margins[0] + 1.0)

    def test_probe_beats_shuffled_label_control(self):
        probe = interpretability_diagnostics()["probe"]
        self.assertGreaterEqual(probe["accuracy"], 0.9)
        self.assertGreaterEqual(probe["selectivity"], 0.25)

    def test_steering_has_risk_and_utility_effects(self):
        doses = interpretability_diagnostics()["steering"]
        self.assertGreater(doses[0]["risk_rate"], doses[-1]["risk_rate"])
        self.assertGreater(doses[0]["utility_proxy"], doses[-1]["utility_proxy"])

    def test_diagnostics_alone_do_not_validate_a_control(self):
        evidence = diagnostic_evidence(interpretability_diagnostics())
        by_id = {item["id"]: item for item in evidence}
        claim = {
            "id": "control-only-in-name",
            "severity": "critical",
            "argument": "control",
            "controls": [{"id": "C1", "evidence": ["E-PROBE", "E-STEER"]}],
        }
        result = evaluate_claim(claim, by_id)
        self.assertEqual(result["status"], "gap")
        self.assertIn("supported only by internal diagnostics", result["reasons"][0])

    def test_inability_claim_requires_adequate_elicitation(self):
        case = load_case()
        evidence = case["evidence"] + diagnostic_evidence(interpretability_diagnostics())
        by_id = {item["id"]: item for item in evidence}
        claim = deepcopy(case["claims"][1])
        self.assertEqual(evaluate_claim(claim, by_id)["status"], "gap")
        claim["capability"]["elicitation_coverage"] = 0.9
        claim["capability"]["sandbagging_check"] = "pass"
        self.assertEqual(evaluate_claim(claim, by_id)["status"], "supported")

    def test_dangling_evidence_is_rejected(self):
        case = load_case()
        case["claims"][0]["controls"][0]["evidence"] = ["missing"]
        evidence = case["evidence"] + diagnostic_evidence(interpretability_diagnostics())
        with self.assertRaisesRegex(ValueError, "missing evidence"):
            validate_case(case, evidence)


if __name__ == "__main__":
    unittest.main()
