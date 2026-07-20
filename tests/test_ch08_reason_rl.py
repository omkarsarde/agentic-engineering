"""Invariants for the Chapter 8 test-time-compute and RLVR build.

Imports the tangled teaching module by file path under a unique name so the
consolidated suite never collides with another chapter's ``_generated``.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "ch08_generated", ROOT / "code" / "ch08" / "_generated.py")
ch08 = importlib.util.module_from_spec(_spec)
sys.modules["ch08_generated"] = ch08  # let frozen dataclasses resolve their module
_spec.loader.exec_module(ch08)


class SolverTest(unittest.TestCase):
    def test_gold_trace_and_verifier(self):
        prob = ch08.Problem(3, (("+", 4), ("-", 1)))
        self.assertEqual(prob.gold_trace, (3, 7, 6))
        self.assertEqual(prob.gold, 6)

    def test_per_attempt_accuracy_matches_closed_form(self):
        # The solver's success rate is reliability**steps, the chapter's knob.
        for steps, reliability in ch08.DIFFICULTY.values():
            measured = ch08.measure_p(steps, reliability, trials=6000)
            self.assertAlmostEqual(measured, reliability ** steps, delta=0.03)

    def test_first_error_locates_slip(self):
        # A fully correct trace has no first error; a corrupted one does.
        good = [(1, True), (2, True)]
        bad = [(1, True), (9, False), (8, False)]
        self.assertIsNone(ch08.first_error(good))
        self.assertEqual(ch08.first_error(bad), 1)


class CoverageTest(unittest.TestCase):
    def test_pass_at_k_estimator(self):
        # No correct attempt -> zero; all correct -> one; monotone in c.
        self.assertEqual(ch08.pass_at_k(8, 0, 4), 0.0)
        self.assertEqual(ch08.pass_at_k(8, 8, 4), 1.0)
        self.assertLess(ch08.pass_at_k(64, 2, 4), ch08.pass_at_k(64, 8, 4))

    def test_coverage_rises_and_matches_theory(self):
        cov = ch08.coverage_curve(*ch08.DIFFICULTY["medium"])
        ks = sorted(cov)
        self.assertEqual([cov[k] for k in ks], sorted(cov[k] for k in ks))
        p = ch08.DIFFICULTY["medium"][1] ** ch08.DIFFICULTY["medium"][0]
        self.assertAlmostEqual(cov[4], 1 - (1 - p) ** 4, delta=0.03)
        self.assertGreater(cov[16], 0.99)

    def test_cot_beats_direct(self):
        direct, cot, sc = ch08.cot_family(*ch08.DIFFICULTY["medium"])
        self.assertGreater(cot, direct)          # serial computation helps
        self.assertGreater(sc[21], sc[1])        # self-consistency climbs


class SelectionTest(unittest.TestCase):
    def test_noisy_verifier_saturates_below_coverage(self):
        cov, plur, bon = ch08.selection_curve(*ch08.DIFFICULTY["medium"])
        self.assertGreater(cov[64], 0.99)
        self.assertLess(bon[64], cov[64])        # selection < coverage
        ceiling = ch08.precision_ceiling(
            ch08.DIFFICULTY["medium"][1] ** ch08.DIFFICULTY["medium"][0], 0.15)
        self.assertAlmostEqual(bon[64], ceiling, delta=0.06)

    def test_plurality_locks_onto_trap(self):
        cov, plur, _ = ch08.selection_curve(4, 0.62, 0.9)
        self.assertGreater(cov[64], 0.99)        # coverage still reaches one
        self.assertLess(plur[64], 0.05)          # plurality confidently wrong

    def test_precision_ceiling_formula(self):
        self.assertAlmostEqual(ch08.precision_ceiling(0.5, 0.0), 1.0)
        self.assertAlmostEqual(ch08.precision_ceiling(0.5, 0.5), 0.5)


class SearchTest(unittest.TestCase):
    def test_beam_beats_greedy_and_equal_budget_sampling(self):
        res = ch08.compare_search(*ch08.DIFFICULTY["hard"])
        self.assertGreater(res["beam"], res["greedy"])
        self.assertGreater(res["beam"], res["indep"])
        # equal-budget comparison: expansions within 2x of each other
        self.assertLess(res["indep_exp"], 2 * res["beam_exp"])

    def test_prm_catches_first_error_early(self):
        prm = ch08.prm_vs_orm_checks(*ch08.DIFFICULTY["hard"])
        self.assertEqual(prm["caught"], prm["wrong"])
        self.assertLess(prm["avg_prm_checks"], prm["steps"])  # early pruning


class ReflectionTest(unittest.TestCase):
    def test_intrinsic_flat_but_destructive(self):
        intrinsic, informed = ch08.self_correction(*ch08.DIFFICULTY["medium"])
        # net change is small, but many correct answers were destroyed
        self.assertLess(abs(intrinsic["after"] - intrinsic["before"]), 0.05)
        self.assertGreater(intrinsic["r2w"], 100)

    def test_verifier_informed_never_destroys(self):
        _, informed = ch08.self_correction(*ch08.DIFFICULTY["medium"])
        self.assertEqual(informed["r2w"], 0)     # verified prefix is locked
        self.assertGreater(informed["after"], informed["before"])


class AllocationTest(unittest.TestCase):
    def test_knee_moves_with_difficulty(self):
        easy = ch08.coverage_curve(*ch08.DIFFICULTY["easy"], problems=400)
        hard = ch08.coverage_curve(*ch08.DIFFICULTY["hard"], problems=400)
        self.assertLess(ch08.knee(easy), ch08.knee(hard))


class GrpoTest(unittest.TestCase):
    def test_group_advantages_center_and_collapse(self):
        adv = ch08.grpo_advantages([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(adv), 0.0, places=6)
        self.assertGreater(adv[0], 0.0)          # a success beats its group
        tied = ch08.grpo_advantages([1.0, 1.0, 1.0])   # a tied group teaches nothing
        self.assertTrue(all(abs(a) < 1e-6 for a in tied))

    def test_exact_rlvr_learns_and_loses_entropy(self):
        history = ch08.train_grpo("exact")
        self.assertGreater(history[-1]["true_acc"], 0.85)
        self.assertLess(history[-1]["entropy"], history[0]["entropy"])

    def test_proxy_run_shows_goodhart(self):
        history = ch08.train_grpo("proxy")
        peak = max(h["true_acc"] for h in history)
        self.assertGreater(peak, 0.8)            # reward first helps
        self.assertLess(history[-1]["true_acc"], 0.15)   # then collapses
        self.assertGreater(history[-1]["objective"], 0.98)  # measure keeps rising
        self.assertGreater(history[-1]["tokens"], 25)    # length exploit

    def test_rlvr_raises_pass_at_1(self):
        base = ch08.policy_coverage([0.3, 0.5, 0.0, 0.0, -1.5])
        trained = ch08.policy_coverage(ch08.train_grpo("exact")[-1]["logits"])
        self.assertGreater(trained[1], base[1])  # pass@1 rises with RLVR


if __name__ == "__main__":
    unittest.main()
