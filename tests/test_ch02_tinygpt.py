"""Executable invariants for the Chapter 2 teaching code.

Imports the tangled module ``code/ch02/_generated.py`` (produced from the
chapter's ``# @save`` cells by ``scripts/tangle.py``) and checks the real
properties the chapter claims: the byte-pair tokenizer round-trips with no
unknown-token path, batches are shifted by exactly one, attention weights are
causal probability rows, the causal mask blocks future information bit-for-bit,
weight tying is storage identity, cached decoding matches the uncached
function, cache storage grows by the derived slice, and the parameter/FLOP
accounting matches the built model.

The module is loaded under a unique name (``ch02_generated``) rather than the
bare ``sys.path`` pattern because several chapters each ship a module called
``_generated``; a plain import would collide inside one pytest process.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch02_generated", ROOT / "code" / "ch02" / "_generated.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ch02 = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("ch02_generated", ch02)
_SPEC.loader.exec_module(ch02)

BytePairTokenizer = ch02.BytePairTokenizer
GPTConfig = ch02.GPTConfig
TinyGPT = ch02.TinyGPT
attention_weights = ch02.attention_weights
flops_per_token = ch02.flops_per_token
generate = ch02.generate
kv_cache_bytes = ch02.kv_cache_bytes
lr_schedule = ch02.lr_schedule
next_token_batch = ch02.next_token_batch
param_breakdown = ch02.param_breakdown
parameter_count = ch02.parameter_count
scaled_dot_product_attention = ch02.scaled_dot_product_attention


def fixture_model() -> TinyGPT:
    torch.manual_seed(3)
    return TinyGPT(
        GPTConfig(vocab_size=280, block_size=20, d_model=32, n_heads=4, n_layers=2)
    ).eval()


def test_bpe_round_trips_and_has_no_unknown_token_path() -> None:
    tokenizer = BytePairTokenizer.train("repeat repeat smaller pieces", 272)
    for text in ("", "repeat", "unseen 🧪 snow 雪", "line one\nline two"):
        ids = tokenizer.encode(text)
        assert all(0 <= token < tokenizer.vocab_size for token in ids)
        assert tokenizer.decode(ids) == text
    # Unfamiliar text falls back to raw bytes, never to an unknown token.
    assert all(token < 256 for token in tokenizer.encode("🧪"))


def test_bpe_merge_order_is_deterministic() -> None:
    first = BytePairTokenizer.train("banana bandana banana", 270)
    second = BytePairTokenizer.train("banana bandana banana", 270)
    assert first.merges == second.merges


def test_next_token_batch_targets_are_shifted_by_exactly_one() -> None:
    data = torch.arange(200, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)
    inputs, targets = next_token_batch(data, block_size=16, batch_size=4, generator=generator)
    assert inputs.shape == targets.shape == (4, 16)
    assert torch.equal(inputs + 1, targets)  # the stream is arange, so +1 is the shift


def test_attention_weights_are_probability_rows_over_values() -> None:
    torch.manual_seed(0)
    query, key, value = torch.randn(3, 5, 8), torch.randn(3, 5, 8), torch.randn(3, 5, 8)
    output, weights = scaled_dot_product_attention(query, key, value)
    torch.testing.assert_close(weights.sum(-1), torch.ones(3, 5))
    assert (weights >= 0).all()
    torch.testing.assert_close(output, weights @ value)
    # A masked pair receives exactly zero weight.
    mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    _, masked = scaled_dot_product_attention(query, key, value, mask=mask)
    assert masked.masked_select(mask).eq(0).all()
    torch.testing.assert_close(masked.sum(-1), torch.ones(3, 5))


def test_causal_mask_blocks_future_information_exactly() -> None:
    model = fixture_model()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    changed_future = torch.tensor([[1, 2, 3, 90, 91]])
    first_logits, _, _ = model(first)
    changed_logits, _, _ = model(changed_future)
    torch.testing.assert_close(first_logits[:, :3], changed_logits[:, :3], rtol=0, atol=0)


def test_random_initial_loss_is_near_log_vocab_size() -> None:
    model = fixture_model()
    tokens = torch.arange(16).view(2, 8) % model.config.vocab_size
    targets = (tokens + 1) % model.config.vocab_size
    _, loss, _ = model(tokens, targets)
    assert loss is not None
    assert abs(loss.item() - math.log(model.config.vocab_size)) < 0.35


def test_cached_and_uncached_logits_match() -> None:
    model = fixture_model()
    context = torch.tensor([[1, 2, 3, 4]])
    _, _, cache = model(context)
    for token in (5, 6, 7):
        next_token = torch.tensor([[token]])
        context = torch.cat((context, next_token), dim=1)
        cached, _, cache = model(next_token, cache=cache)
        uncached, _, _ = model(context)
        torch.testing.assert_close(cached[:, -1], uncached[:, -1], rtol=1e-5, atol=1e-6)
    assert torch.equal(
        generate(model, context, 4, use_cache=True),
        generate(model, context, 4, use_cache=False),
    )
    # generate is also attached as a method for downstream chapters.
    assert torch.equal(model.generate(context, 4), generate(model, context, 4))


def test_cache_storage_grows_by_one_fixed_slice_per_token() -> None:
    model = fixture_model()
    _, _, cache = model(torch.tensor([[1]]))
    sizes = [kv_cache_bytes(cache)]
    for token in (2, 3, 4):
        _, _, cache = model(torch.tensor([[token]]), cache=cache)
        sizes.append(kv_cache_bytes(cache))
    expected_step = 2 * model.config.n_layers * model.config.d_model * 4
    assert [right - left for left, right in zip(sizes, sizes[1:])] == [expected_step] * 3


def test_weight_tying_is_storage_identity() -> None:
    model = fixture_model()
    assert model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()
    # The tied tensor is counted once, so untying would add exactly V*d.
    config = model.config
    assert (
        param_breakdown(config)["embedding (tied with LM head)"]
        == config.vocab_size * config.d_model
    )


def test_param_breakdown_matches_built_models() -> None:
    for config in (
        GPTConfig(vocab_size=512),
        GPTConfig(vocab_size=280, block_size=20, d_model=32, n_heads=4, n_layers=2),
        GPTConfig(vocab_size=320, block_size=64, d_model=32, n_heads=4, n_layers=1, mlp_ratio=2.0),
    ):
        torch.manual_seed(0)
        model = TinyGPT(config)
        assert param_breakdown(config)["total"] == parameter_count(model)
        assert model.parameter_count() == parameter_count(model)


def test_flops_per_token_terms_are_consistent() -> None:
    config = GPTConfig(vocab_size=512)
    short = flops_per_token(config, context_len=1)
    long = flops_per_token(config, context_len=128)
    assert short["total"] == short["weights"] + short["attention"]
    # Weight FLOPs are context-independent; attention FLOPs grow linearly.
    assert short["weights"] == long["weights"]
    assert long["attention"] == 128 * short["attention"]
    # The folk 2N rule bounds the weight term (embeddings are lookup, not matmul).
    torch.manual_seed(0)
    assert short["weights"] < 2 * parameter_count(TinyGPT(config))


def test_attention_weights_helper_is_causal_and_normalized() -> None:
    model = fixture_model()
    weights = attention_weights(model, [3, 1, 4, 1, 5], layer=1)
    heads, steps, _ = weights.shape
    assert (heads, steps) == (model.config.n_heads, 5)
    torch.testing.assert_close(weights.sum(-1), torch.ones(heads, steps))
    future = torch.triu(torch.ones(steps, steps, dtype=torch.bool), diagonal=1)
    assert weights.masked_select(future.expand(heads, steps, steps)).eq(0).all()


def test_lr_schedule_warms_up_then_decays() -> None:
    peak = 3e-3
    values = [lr_schedule(step, 400, peak) for step in range(400)]
    warmup = 400 // 20
    assert values[:warmup] == sorted(values[:warmup])  # rising
    assert max(values) <= peak + 1e-12
    assert abs(values[-1] - 0.1 * peak) < 1e-4  # cosine floor at one tenth of peak
    assert values[warmup:] == sorted(values[warmup:], reverse=True)  # decaying


def test_generate_is_seed_deterministic_and_respects_block_size() -> None:
    model = fixture_model()
    prompt = torch.tensor([[1, 2, 3]])
    first = generate(model, prompt, 5, temperature=0.9, seed=11)
    second = generate(model, prompt, 5, temperature=0.9, seed=11)
    assert torch.equal(first, second)
    try:
        generate(model, prompt, model.config.block_size, temperature=0.0)
    except ValueError:
        pass
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("generation beyond block_size must raise ValueError")
