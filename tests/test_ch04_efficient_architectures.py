"""Executable contracts for Chapter 4 (MoE and efficient architectures).

Imports the chapter's tangled teaching module under a unique name so it never
collides with another chapter's ``_generated`` during a full-suite run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_generated():
    spec = importlib.util.spec_from_file_location(
        "ch04_generated", ROOT / "code" / "ch04" / "_generated.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ch04_generated"] = module
    spec.loader.exec_module(module)
    return module


ch04 = _load_generated()


def test_router_shapes_scores_and_top_k_load() -> None:
    torch.manual_seed(2)
    tokens = torch.randn(3, 5, 8)
    for scoring in ("softmax", "sigmoid"):
        router = ch04.TopKRouter(8, experts=4, top_k=2, scoring=scoring)
        state = router(tokens)
        assert state.probabilities.shape == (3, 5, 4)
        assert state.indices.shape == state.weights.shape == (3, 5, 2)
        torch.testing.assert_close(state.weights.sum(-1), torch.ones(3, 5))
        torch.testing.assert_close(state.load.sum(), torch.tensor(1.0))
        assert torch.isfinite(state.balance_loss) and torch.isfinite(state.z_loss)


def test_aux_free_controller_pushes_down_an_overloaded_expert() -> None:
    router = ch04.TopKRouter(4, experts=4, top_k=1)
    router.update_selection_bias(torch.tensor([0.7, 0.1, 0.1, 0.1]), rate=0.02)
    assert router.selection_bias[0] < 0
    assert torch.all(router.selection_bias[1:] > 0)
    torch.testing.assert_close(router.selection_bias.mean(), torch.tensor(0.0))


def test_balancing_prevents_seeded_router_collapse() -> None:
    unbalanced = ch04.train_router(0.0)
    balanced = ch04.train_router(0.5)
    assert max(unbalanced["load"]) > 0.95          # collapses onto one expert
    assert max(balanced["load"]) < 0.35            # spread near the 25% target
    assert min(unbalanced["accuracy"], balanced["accuracy"]) > 0.9


def test_gated_linear_attention_is_chunk_equivalent_with_fixed_state() -> None:
    torch.manual_seed(7)
    layer = ch04.GatedLinearAttention(width=12, state_width=5).eval()
    tokens = torch.randn(2, 9, 12)
    full, full_state = layer(tokens)
    recurrent, pieces = None, []
    for i in range(tokens.size(1)):
        out, recurrent = layer(tokens[:, i : i + 1], recurrent)
        pieces.append(out)
    torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-5, atol=1e-6)
    assert recurrent is not None
    assert sum(t.numel() for t in recurrent) == 2 * (5 * 12 + 5)  # fixed in context length
    assert [t.shape for t in recurrent] == [t.shape for t in full_state]


def test_fixed_state_recall_falls_and_wider_state_helps() -> None:
    counts = [2, 8, 16, 64, 128]
    narrow = {r["pairs"]: r["accuracy"] for r in ch04.associative_recall_curve(counts, feature_dim=16)}
    wide = {r["pairs"]: r["accuracy"] for r in ch04.associative_recall_curve(counts, feature_dim=64)}
    assert narrow[2] > 0.95                         # near-perfect when far below capacity
    assert narrow[128] < narrow[8]                  # degrades as pairs pile up
    assert wide[64] > narrow[64]                    # wider state recalls more at the same load
    assert wide[128] < 0.5                          # but the limit is not repealed


def test_config_parser_reconstructs_totals_and_known_kv_cases() -> None:
    estimates = {name: ch04.estimate_config(name, cfg) for name, cfg in ch04.REAL_CONFIGS.items()}
    assert set(estimates) == {"Llama 3.1 8B", "DeepSeek-V3", "Qwen3-Next 80B-A3B"}
    assert all(abs(e.total_error_percent) < 2 for e in estimates.values())
    llama, deepseek, qwen = estimates["Llama 3.1 8B"], estimates["DeepSeek-V3"], estimates["Qwen3-Next 80B-A3B"]
    assert llama.active == llama.total                       # dense: one ledger
    assert llama.kv_bytes_per_token == 131_072
    assert llama.state_gib_32k == 4.0
    assert deepseek.kv_bytes_per_token == 61 * 576 * 2       # MLA latent + rope key
    assert deepseek.active < deepseek.total                  # MoE bends the active curve
    assert qwen.kv_bytes_per_token == 12 * 2 * 2 * 256 * 2   # 12 full-attention layers only
    assert qwen.fixed_state_bytes > 0                        # linear layers hold flat state
    assert qwen.active_fraction < 0.1


def test_kv_arithmetic_is_reused_from_chapter_three() -> None:
    # The parser must not reimplement KV bytes; it re-exports Chapter 3's function.
    cfg = ch04.KVConfig("probe", layers=2, query_heads=4, kv_heads=2, head_dim=8, bytes_per_scalar=2)
    assert ch04.kv_bytes(cfg, 10) == 10 * 2 * (2 * 2 * 8) * 2
