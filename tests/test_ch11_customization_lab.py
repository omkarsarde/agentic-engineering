"""Mechanism tests for Chapter 11's customization lab (tangled from the chapter).

The chapter tangles its teaching code into ``code/ch11/_generated.py``; these
tests import that module under a chapter-unique name and exercise the LoRA
adapter, NF4 quantizer, distillation loss, task-vector merge, and release gate
on deliberately tiny training runs so the suite stays fast and offline.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "ch11_generated", ROOT / "code" / "ch11" / "_generated.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ch11_generated"] = module
    spec.loader.exec_module(module)
    return module


ch11 = _load()


def _tiny_base(seed: int = 1):
    torch.manual_seed(seed)
    model = ch11.small_gpt(d_model=32, n_layers=1, n_heads=2)
    ch11.fit(model, ch11.make_stream(["E", "A"], 300, seed=10), steps=80, lr=3e-3, seed=seed)
    return model


class CustomizationTests(unittest.TestCase):
    def test_nf4_has_exact_zero_and_preserves_shape(self) -> None:
        self.assertIn(0.0, ch11.NF4.tolist())
        w = torch.linspace(-2.0, 2.0, 64).reshape(8, 8)
        q = ch11.nf4_quantize(w, group_size=32)
        self.assertEqual(q.shape, w.shape)
        self.assertGreater(float((q - w).norm()), 0.0)
        self.assertLess(float((q - w).norm() / w.norm()), 0.2)

    def test_lora_starts_as_identity_and_freezes_base(self) -> None:
        base = _tiny_base()
        x = torch.tensor([[ch11.STOI[c] for c in "E1234="]])
        before, _, _ = base(x)
        lora = ch11.attach_lora(copy.deepcopy(base), rank=2, alpha=4)
        after, _, _ = lora(x)
        torch.testing.assert_close(before, after)  # B=0 => identical first pass

        frozen = {n: p.detach().clone() for n, p in lora.named_parameters() if ".linear." in n}
        trainable = ch11.lora_parameters(lora)
        self.assertLess(sum(p.numel() for p in trainable),
                        sum(p.numel() for p in base.parameters()))
        ch11.fit(lora, ch11.make_stream("F", 200, seed=20), steps=40, lr=5e-3, seed=2)
        for n, p in lora.named_parameters():
            if ".linear." in n:
                torch.testing.assert_close(p.detach(), frozen[n])  # base never moved
        delta_norm = sum(float(m.delta.detach().norm()) for m in lora.modules()
                         if isinstance(m, ch11.LoRALinear))
        self.assertGreater(delta_norm, 0.0)  # but the adapter did

    def test_ties_discards_small_and_sign_conflicting_coordinates(self) -> None:
        a = torch.tensor([[5.0, 0.05, -4.0, 3.0]])
        b = torch.tensor([[6.0, 0.03, 2.0, 4.0]])
        merged = ch11.ties_merge(a, b, 0.5, 0.5, density=0.5)
        self.assertGreater(float(merged[0, 0]), 0.0)   # both large positive, kept
        self.assertEqual(float(merged[0, 1]), 0.0)     # small in both, trimmed away
        self.assertLess(float(merged[0, 2]), 0.0)      # sign conflict: elected -1 wins

    def test_release_gate_blocks_regression_and_ships_clean_gain(self) -> None:
        base = {"S": 0.4, "E": 1.0, "A": 1.0}
        forgetful = {"S": 1.0, "E": 0.2, "A": 0.3}
        clean = {"S": 1.0, "E": 1.0, "A": 1.0}
        self.assertFalse(ch11.release_gate(base, forgetful, "S", "EA")["ship"])
        self.assertTrue(ch11.release_gate(base, clean, "S", "EA")["ship"])

    def test_distillation_and_merge_helpers_run(self) -> None:
        teacher = _tiny_base()
        ch11.fit(teacher, ch11.make_stream("F", 300, seed=20), steps=60, lr=3e-3, seed=2)
        torch.manual_seed(7)
        student = ch11.small_gpt(d_model=16, n_layers=1, n_heads=2)
        losses = ch11.distill_logits(teacher=teacher, student=student,
                                     stream=ch11.make_stream("F", 60, seed=31),
                                     steps=30, lr=2e-3)
        self.assertLess(losses[-1], losses[0])  # the blended KD loss falls

        completions = ch11.teacher_completions(teacher, "F", 8, seed=3)
        self.assertTrue((completions == ch11.EQ).any())

        la = ch11.attach_lora(copy.deepcopy(teacher), rank=2)
        ch11.lora_parameters(la)
        ch11.fit(la, ch11.make_stream("F", 200, seed=20), steps=20, lr=5e-3, seed=2)
        lb = ch11.attach_lora(copy.deepcopy(teacher), rank=2)
        ch11.lora_parameters(lb)
        ch11.fit(lb, ch11.make_stream("S", 200, seed=25), steps=20, lr=5e-3, seed=3)
        merged = ch11.merged_model(teacher, ch11.collect_deltas(la),
                                   ch11.collect_deltas(lb), 0.5, 0.5, "linear")
        self.assertEqual(float(ch11.accuracy(merged, "E", n=10)) >= 0.0, True)


if __name__ == "__main__":
    unittest.main()
