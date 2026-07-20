"""Mechanism tests for the Chapter 11 customization lab."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch11"))

from customization_lab import LoRA, NF4, make_fixture, nf4_quantize, run_lab, softmax, ties_merge  # noqa: E402


class CustomizationLabTests(unittest.TestCase):
    def test_nf4_has_exact_zero_and_preserves_shape(self) -> None:
        self.assertIn(0.0, NF4)
        weights = np.linspace(-2.0, 2.0, 48).reshape(4, 12)
        restored, rmse = nf4_quantize(weights, group_size=16)
        self.assertEqual(restored.shape, weights.shape)
        self.assertGreater(rmse, 0.0)

    def test_softmax_rows_are_probabilities(self) -> None:
        probs = softmax(np.array([[1000.0, 999.0], [-1000.0, -999.0]]))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        self.assertTrue(np.all(probs >= 0.0))

    def test_lora_freezes_the_base(self) -> None:
        fixture = make_fixture()
        base, _ = nf4_quantize(fixture["base"])
        frozen = base.copy()
        adapter = LoRA(base, seed=3)
        x, labels = fixture["sets"]["train"]
        for _ in range(10):
            adapter.sft_step(x, labels, learning_rate=0.1)
        np.testing.assert_array_equal(base, frozen)
        self.assertGreater(float(np.linalg.norm(adapter.delta)), 0.0)

    def test_ties_discards_small_and_sign_conflicting_updates(self) -> None:
        first = np.array([[4.0, 0.1, -3.0, 0.2]])
        second = np.array([[2.0, -0.2, 1.0, 0.1]])
        merged = ties_merge([first, second], [0.5, 0.5], density=0.5)
        self.assertGreater(merged[0, 0], 0.0)
        self.assertLess(merged[0, 2], 0.0)
        self.assertEqual(merged[0, 1], 0.0)

    def test_integrated_release_gate_and_merge_ablation(self) -> None:
        metrics = run_lab()
        self.assertTrue(metrics["gate"]["passed"])
        self.assertLess(metrics["losses"]["sft_last"], metrics["losses"]["sft_first"])
        self.assertLess(metrics["losses"]["kd_last"], metrics["losses"]["kd_first"])
        base = metrics["scores"]["base_nf4"]["task_test"]
        distilled = metrics["scores"]["logit_kd"]["task_test"]
        self.assertGreater(distilled, base)
        equal = [row for row in metrics["merge_sweep"] if row["weight_a"] == 0.5]
        by_method = {row["method"]: row for row in equal}
        self.assertGreater(by_method["ties"]["task_mean"], by_method["linear"]["task_mean"])


if __name__ == "__main__":
    unittest.main()
