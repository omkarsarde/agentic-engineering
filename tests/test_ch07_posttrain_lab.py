"""Invariants for the Chapter 7 post-training build (tangled from the chapter).

The chapter tangles its teaching code into ``code/ch07/_generated.py``; these
tests import that module under a unique name and re-run the small SFT, reward,
and DPO experiments to check the mechanisms the chapter claims: the response
mask, the template ABI, the length bias entering the reward model, and DPO
moving the preference against a frozen reference.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ch07 = _load("ch07_generated", "code/ch07/_generated.py")


class TemplateAndSpecTest(unittest.TestCase):
    def test_rule_trace_finds_governed_rows(self) -> None:
        self.assertEqual(ch07.rows_for_rule("refund.identity.v3"), [2, 3, 7])
        self.assertEqual(ch07.rows_for_rule("nonexistent"), [])

    def test_special_tokens_are_distinct_and_above_vocab(self) -> None:
        specials = set(ch07.SPECIAL.values())
        self.assertEqual(len(specials), 4)
        self.assertTrue(all(sid >= ch07.VOCAB_BASE for sid in specials))
        self.assertEqual(ch07.VOCAB, ch07.VOCAB_BASE + 4)

    def test_render_chat_round_trips_content(self) -> None:
        ids = ch07.render_chat([{"role": "user", "content": "can i get a refund"}],
                               add_generation_prompt=True)
        self.assertEqual(ids[0], ch07.SPECIAL["user"])
        self.assertEqual(ids[-1], ch07.SPECIAL["assistant"])
        content = ch07.tokenizer.decode([t for t in ids if t < ch07.VOCAB_BASE], errors="replace")
        self.assertEqual(content, "can i get a refund")

    def test_response_mask_covers_only_the_assistant_span(self) -> None:
        inp, tgt, mask = ch07.build_example("can i get a refund",
                                            "please share your order id and i will check eligibility.")
        self.assertEqual(len(inp), len(tgt))
        self.assertEqual(len(mask), len(tgt))
        self.assertGreater(sum(mask), 0)
        # the first target position (predicting from the user's role token) is masked out
        self.assertEqual(mask[0], 0)
        # the final target is the end token and must be trained (learn to stop)
        self.assertEqual(tgt[-1], ch07.SPECIAL["end"])
        self.assertEqual(mask[-1], 1)


class PackingTest(unittest.TestCase):
    def test_block_diagonal_isolates_examples(self) -> None:
        segments = [0, 0, 0, 1, 1]
        mask = ch07.block_diagonal_mask(segments)
        # a query in example 1 sees none of example 0
        self.assertTrue(all(mask[3][k] for k in range(3)))
        # within an example, causal order holds: position 4 sees 3 and 4, not the future
        self.assertFalse(mask[4][3])
        self.assertFalse(mask[4][4])


class SupervisedFineTuningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch
        torch.manual_seed(1)
        cls.torch = torch
        cls.base = ch07.ch02.TinyGPT(ch07.make_config())
        cls.sft = ch07.ch02.TinyGPT(ch07.make_config())
        cls.sft.load_state_dict(cls.base.state_dict())
        cls.history = ch07.sft_train(cls.sft, ch07.DEMOS, mask_prompt=True, seed=0)

    def test_sft_loss_drops_and_reproduces_demonstrations(self) -> None:
        self.assertLess(self.history[-1], 0.1)
        self.assertGreater(self.history[0], 1.0)
        reproduced = sum(ch07.chat(self.sft, u)[0].strip() == a.strip() for u, a in ch07.DEMOS)
        self.assertGreaterEqual(reproduced, 9)

    def test_masking_prevents_learning_the_user_turn(self) -> None:
        unmasked = ch07.ch02.TinyGPT(ch07.make_config())
        unmasked.load_state_dict(self.base.state_dict())
        ch07.sft_train(unmasked, ch07.DEMOS, mask_prompt=False, seed=0)
        # both learn assistant tokens; only the unmasked model learns user tokens
        self.assertAlmostEqual(ch07.span_loss(self.sft, ch07.DEMOS, "assistant"),
                               ch07.span_loss(unmasked, ch07.DEMOS, "assistant"), delta=0.2)
        self.assertGreater(ch07.span_loss(self.sft, ch07.DEMOS, "user"), 2.0)
        self.assertLess(ch07.span_loss(unmasked, ch07.DEMOS, "user"), 1.0)


class PreferenceAndRewardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = ch07.collect_preferences()
        cls.weights, cls.history = ch07.train_reward_model(cls.rows)

    def test_collection_exposes_position_and_length_bias(self) -> None:
        left = sum(r[3] for r in self.rows) / len(self.rows)
        longer = sum(ch07.features(p, c)[2] > ch07.features(p, r)[2]
                     for p, c, r, _ in self.rows) / len(self.rows)
        self.assertGreater(left, 0.5)
        self.assertGreater(longer, 0.55)

    def test_reward_model_learns_the_length_bias(self) -> None:
        self.assertLess(self.history[-1], self.history[0])
        # spec wants a negative length weight; the biased annotator teaches a positive one
        self.assertGreater(self.weights[2], 0.8)
        self.assertLess(ch07.SPEC_WEIGHTS[2], 0)
        test = ch07.collect_preferences(count=200, seed=99)
        acc = sum(ch07.dot(ch07.features(p, c), self.weights)
                  > ch07.dot(ch07.features(p, r), self.weights) for p, c, r, _ in test) / len(test)
        self.assertGreater(acc, 0.75)


class DirectPreferenceOptimizationTest(unittest.TestCase):
    def test_dpo_moves_preference_against_a_frozen_reference(self) -> None:
        import torch
        torch.manual_seed(1)
        base = ch07.ch02.TinyGPT(ch07.make_config())
        sft = ch07.ch02.TinyGPT(ch07.make_config())
        sft.load_state_dict(base.state_dict())
        ch07.sft_train(sft, ch07.DEMOS, seed=0)

        pairs = ch07.dpo_pairs()
        self.assertEqual(len(pairs), len(ch07.PROMPTS))
        reference = ch07.ch02.TinyGPT(ch07.make_config()); reference.load_state_dict(sft.state_dict())
        policy = ch07.ch02.TinyGPT(ch07.make_config()); policy.load_state_dict(sft.state_dict())

        def margin(p, w, l):
            return ((ch07.seq_logp(policy, p, w) - ch07.seq_logp(reference, p, w))
                    - (ch07.seq_logp(policy, p, l) - ch07.seq_logp(reference, p, l))).item()

        with torch.no_grad():
            before = [margin(p, w, l) for p, w, l in pairs]
        self.assertTrue(all(abs(m) < 1e-5 for m in before))  # policy == reference at start

        history = ch07.train_dpo(policy, reference, pairs)
        self.assertLess(history[-1], history[0])
        with torch.no_grad():
            after = [margin(p, w, l) for p, w, l in pairs]
        # DPO raises the chosen (longer) answer's implicit reward in every pair
        self.assertTrue(all(m > 0 for m in after))


if __name__ == "__main__":
    unittest.main()
