"""Executable invariants for the Chapter 3 teaching code.

Imports the tangled module ``code/ch03/_generated.py`` (produced from the
chapter's ``# @save`` cells by ``scripts/tangle.py``) and checks the real
properties the chapter claims: the canonical KV-byte arithmetic and its MHA →
GQA → MLA ratios, that the grouped-query dial changes stored state but not
output shape and its cached path matches a full pass, that the latent cache
compresses while attention still works, that visibility masks realize causal /
windowed / sink patterns, that RoPE preserves norm and depends only on relative
offset, that the PI/NTK/YaRN band constructions behave as designed, and that the
retrieval probe is deterministic and ranks YaRN above unscaled RoPE.

The module is loaded under a unique name (``ch03_generated``) rather than the
bare ``sys.path`` pattern because several chapters each ship a module called
``_generated``; a plain import would collide inside one pytest process.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch03_generated", ROOT / "code" / "ch03" / "_generated.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ch03 = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("ch03_generated", ch03)
_SPEC.loader.exec_module(ch03)

KVConfig = ch03.KVConfig
kv_bytes = ch03.kv_bytes
GroupedQueryAttention = ch03.GroupedQueryAttention
ToyLatentKVAttention = ch03.ToyLatentKVAttention
visibility_mask = ch03.visibility_mask
apply_rope = ch03.apply_rope
inverse_frequencies = ch03.inverse_frequencies
ntk_base = ch03.ntk_base
retrieval_probe = ch03.retrieval_probe


def test_canonical_kv_arithmetic_and_gqa_ratio() -> None:
    mha = KVConfig("MHA", layers=32, query_heads=32, kv_heads=32, head_dim=128, bytes_per_scalar=2)
    gqa = KVConfig("GQA", layers=32, query_heads=32, kv_heads=8, head_dim=128, bytes_per_scalar=2)
    assert kv_bytes(mha, 1) == 512 * 1024              # 512 KiB per token
    assert kv_bytes(mha, 4_096) == 2 * 2**30           # 2 GiB
    assert kv_bytes(mha, 131_072) == 64 * 2**30        # 64 GiB
    assert kv_bytes(gqa, 65_536) == 8 * 2**30          # 8 GiB (8.6 GB) at 64K
    assert kv_bytes(gqa, 1) * 4 == kv_bytes(mha, 1)    # ratio H_kv / H_q = 1/4


def test_mla_cache_width_is_latent_plus_decoupled_rope_key() -> None:
    mla = KVConfig("MLA", layers=27, query_heads=16, head_dim=128, bytes_per_scalar=2,
                   latent_rank=512, rope_key_dim=64)
    assert mla.cached_scalars_per_layer_token() == 576
    assert kv_bytes(mla, 1) == 27 * 576 * 2


def test_kv_bytes_rejects_bad_arguments() -> None:
    cfg = KVConfig("x", layers=1, query_heads=1, kv_heads=1, head_dim=1, bytes_per_scalar=2)
    for tokens, batch in ((-1, 1), (1, 0)):
        try:
            kv_bytes(cfg, tokens, batch)
        except ValueError:
            continue
        raise AssertionError("kv_bytes should reject negative tokens / non-positive batch")


def test_grouped_query_dial_changes_state_not_output_shape() -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 5, 64)                           # 8 query heads -> head_dim 8
    scalars = []
    for kv_heads in (8, 2, 1):
        layer = GroupedQueryAttention(64, query_heads=8, kv_heads=kv_heads)
        out, (k, v) = layer(x)
        assert out.shape == x.shape
        assert tuple(k.shape) == (2, kv_heads, 5, 8)
        scalars.append(k.numel() + v.numel())
    assert scalars == [1280, 320, 160]                 # 1, 1/4, 1/8 of MHA


def test_grouped_query_cached_chunks_match_full_sequence() -> None:
    torch.manual_seed(5)
    layer = GroupedQueryAttention(64, query_heads=8, kv_heads=2).eval()
    x = torch.randn(1, 6, 64)
    full, _ = layer(x)
    cache, pieces = None, []
    for t in range(x.size(1)):
        step, cache = layer(x[:, t:t + 1], cache)
        pieces.append(step)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-5, atol=1e-6)


def test_toy_mla_compresses_and_reconstructs() -> None:
    torch.manual_seed(11)
    layer = ToyLatentKVAttention(64, query_heads=4, latent_rank=16, rope_key_dim=4).eval()
    x = torch.randn(1, 8, 64)
    full, latent = layer(x)
    assert full.shape == x.shape
    assert layer.latent_width() == 20                  # r + d_R
    assert tuple(latent.shape) == (1, 8, 16)           # only the latent is cached
    cache, pieces = None, []
    for t in range(x.size(1)):
        step, cache = layer(x[:, t:t + 1], cache)
        pieces.append(step)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-5, atol=1e-6)


def test_visibility_mask_realizes_causal_window_and_sinks() -> None:
    causal = visibility_mask(4)
    assert torch.equal(causal, torch.tril(torch.ones(4, 4, dtype=torch.bool)))
    windowed = visibility_mask(4, window=2)
    assert not windowed[3, 0] and windowed[3, 2] and windowed[3, 3]
    sinked = visibility_mask(6, window=2, sinks=1)
    assert sinked[5, 0] and not visibility_mask(6, window=2)[5, 0]


def test_rope_preserves_norm_and_depends_only_on_relative_offset() -> None:
    torch.manual_seed(0)
    freqs, _ = inverse_frequencies(16)
    q, k = torch.randn(1, 1, 16), torch.randn(1, 1, 16)

    def score(m: int, n: int) -> float:
        qm = apply_rope(q, torch.tensor([m]), freqs)
        kn = apply_rope(k, torch.tensor([n]), freqs)
        return (qm * kn).sum().item()

    assert abs(score(3, 8) - score(103, 108)) < 1e-4   # equal offset -> equal score
    assert abs(score(3, 8) - score(3, 9)) > 1e-3       # different offset -> different score
    rotated = apply_rope(q, torch.tensor([50]), freqs)
    torch.testing.assert_close(rotated.norm(), q.norm(), rtol=1e-6, atol=1e-6)


def test_scaling_reshapes_frequency_bands_as_designed() -> None:
    rope, _ = inverse_frequencies(16)
    pi, _ = inverse_frequencies(16, method="pi", factor=4)
    ntk, _ = inverse_frequencies(16, method="ntk", factor=4)
    yarn, magnitude = inverse_frequencies(16, method="yarn", factor=4, original_context=32)
    torch.testing.assert_close(pi, rope / 4)
    assert ntk_base(10_000, 4, 16) > 10_000
    assert ntk[0] == rope[0] and ntk[-1] < rope[-1]    # NTK pins the top band
    assert torch.all(yarn <= rope) and torch.all(yarn >= pi)
    assert magnitude > 1


def test_retrieval_probe_is_deterministic_and_ranks_yarn_above_rope() -> None:
    rope_rows = retrieval_probe("rope")
    assert len(rope_rows) == 21
    assert retrieval_probe("rope") == rope_rows        # seeded -> reproducible
    mean = lambda rows: sum(r["accuracy"] for r in rows) / len(rows)
    assert mean(retrieval_probe("yarn")) > mean(rope_rows)
