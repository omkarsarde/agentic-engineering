from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "ch09"))

import hf_inference_adapter  # noqa: E402
import inference_lab as lab  # noqa: E402
sys.modules.pop("render_metrics", None)
import render_metrics  # noqa: E402


class InferenceLabTest(unittest.TestCase):
    def test_temperature_and_truncation_invariants(self) -> None:
        logits = [3.0, 2.0, 1.0]
        self.assertEqual(lab.softmax(logits, 0.0), [1.0, 0.0, 0.0])
        self.assertEqual(sorted(range(3), key=lab.softmax(logits, 2.0).__getitem__, reverse=True), [0, 1, 2])
        self.assertEqual(sum(p > 0 for p in lab.truncate([0.7, 0.2, 0.1], "top_k", 2)), 2)
        self.assertGreaterEqual(sum(lab.truncate([0.7, 0.2, 0.1], "top_p", 0.0)), 0.999999)
        self.assertEqual(sum(p > 0 for p in lab.truncate([0.7, 0.2, 0.1], "min_p", 0.3)), 1)

    def test_processors_are_sign_aware(self) -> None:
        processed = lab.process_logits([2.0, -2.0], [1, 1], repetition=2.0)
        self.assertEqual(processed, [1.0, -4.0])

    def test_request_anatomy_and_kv_accounting(self) -> None:
        rows = lab.request_profiles()
        self.assertEqual(lab.kv_bytes(128), 16 * 2**20)
        self.assertEqual([row["kv_mib"] for row in rows], [16.0, 64.0, 256.0])
        self.assertTrue(all(row["prefill_bound"] == "compute" for row in rows))
        self.assertTrue(all(row["decode_bound"] == "bandwidth" for row in rows))
        self.assertLess(rows[0]["ttft_ms"], rows[-1]["ttft_ms"])

    def test_seed_control_is_not_batch_invariance(self) -> None:
        metrics = lab.run_experiment()["determinism"]
        self.assertTrue(metrics["seed_replay_exact"])
        self.assertEqual(metrics["greedy_flips"], 1)
        self.assertGreater(metrics["max_probability_drift"], 0.2)

    def test_calibration_conformal_and_abstention(self) -> None:
        metrics = lab.run_experiment()
        self.assertLess(metrics["calibration"]["calibrated"]["ece"], metrics["calibration"]["raw"]["ece"])
        self.assertEqual(metrics["crc"]["threshold"], 0.85)
        self.assertLessEqual(metrics["crc"]["marginal_error"], metrics["crc"]["alpha"])
        self.assertGreater(metrics["crc"]["selective_risk"], metrics["crc"]["alpha"])
        self.assertGreaterEqual(metrics["split_conformal"]["set_coverage"], 0.9)

    def test_semantics_streaming_adapter_and_svg(self) -> None:
        entropy = lab.semantic_entropy_probe()
        self.assertLess(entropy["semantic_entropy_nats"], entropy["surface_entropy_nats"])
        text = lab.stream_until([b"A\xf0", b"\x9f\x99", b"\x82ST", b"OPtail"], "STOP")
        self.assertEqual(text, "A🙂")
        self.assertTrue(callable(hf_inference_adapter.profile_request))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metrics.svg"
            render_metrics.render(output)
            svg = output.read_text(encoding="utf-8")
            self.assertIn("One-request roofline", svg)
            self.assertIn("CRC α=0.10", svg)


if __name__ == "__main__":
    unittest.main()
