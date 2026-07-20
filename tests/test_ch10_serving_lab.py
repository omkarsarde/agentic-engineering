"""Executable invariants for the Chapter 10 serving-lab teaching code.

Imports the tangled module ``code/ch10/_generated.py`` (produced from the
chapter's ``# @save`` cells by ``scripts/tangle.py``) and checks the real
properties the chapter claims: that continuous batching beats static batching on
utilization and SLO-qualified goodput, that the load sweep shows throughput
saturating while goodput collapses past the knee, that paged allocation wastes
far less KV than a contiguous one, that a prefix cache key isolates tenants and
its break-even and effective-cost math are right, that speculative sampling
preserves the target distribution yet can lose speedup under contention, that
grammar masking renormalizes and rejects unsatisfiable sets, that per-row int8
quantization of the Chapter 2 model is near-lossless while int4 drifts, that a
finer quantization block lowers RMSE, that the disaggregation inequality flips
with prompt size, that a tenant-scoped cache key closes the presence side
channel, and that the reasoning-budget sweep saturates and then declines.

The module is loaded under a unique name (``ch10_generated``) because several
chapters each ship a module called ``_generated``; a plain import would collide
inside one pytest process.
"""

from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch10_generated", ROOT / "code" / "ch10" / "_generated.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ch10 = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("ch10_generated", ch10)
_SPEC.loader.exec_module(ch10)


def test_continuous_batching_beats_static_on_goodput() -> None:
    static = ch10.static_schedule(ch10.REQUEST_LENGTHS, 4)
    continuous = ch10.continuous_schedule(ch10.REQUEST_LENGTHS, 4)
    assert static["makespan_steps"] == 32 and continuous["makespan_steps"] == 26
    assert continuous["utilization"] > static["utilization"]
    static_g = ch10.goodput(static, ch10.REQUEST_LENGTHS)
    continuous_g = ch10.goodput(continuous, ch10.REQUEST_LENGTHS)
    assert static_g["qualified"] == 8 and continuous_g["qualified"] == 12
    assert continuous_g["per_step"] > static_g["per_step"]


def test_load_sweep_shows_the_goodput_knee() -> None:
    capacity = 8 / 20.0
    rates = [round(capacity * f, 3) for f in (0.5, 1.0, 2.0)]
    rows = ch10.sweep_load(rates)
    # throughput rises then saturates near capacity while goodput collapses.
    assert rows[0]["throughput"] < rows[1]["throughput"]
    assert rows[2]["throughput"] <= rows[1]["throughput"] * 1.1      # saturated
    assert rows[2]["throughput"] < capacity * 1.05                   # near ceiling
    assert rows[-1]["goodput"] < rows[0]["goodput"]                  # past-knee collapse
    assert rows[0]["qualified"] > rows[-1]["qualified"]


def test_paged_allocation_wastes_far_less_kv() -> None:
    frag = ch10.kv_fragmentation([17, 5, 130, 44, 9, 88, 3, 61, 200, 12, 150, 7], 16, 256)
    assert frag["paged_waste"] < frag["naive_waste"]
    assert frag["naive_over_paged"] > 3.0
    # paged waste is at most one block per sequence.
    assert frag["paged_reserved"] - frag["used"] <= 16 * 12


def test_cache_key_isolates_tenants_and_prices_reuse() -> None:
    prefix = list(range(45))
    assert ch10.make_cache_key("a", prefix) == ch10.make_cache_key("a", prefix)
    assert ch10.make_cache_key("a", prefix) != ch10.make_cache_key("b", prefix)
    assert math.isclose(ch10.cache_breakeven(1.0, 1.25, 0.10), 0.25 / 0.90)
    assert ch10.cache_breakeven(1.0, 1.25, 1.0) == math.inf
    # effective cost crosses the recompute baseline at ~0.22.
    assert ch10.effective_cost(0.0) > 1.0 and ch10.effective_cost(0.5) < 1.0


def test_speculative_sampling_is_exact_but_can_lose_speedup() -> None:
    target, draft = [0.52, 0.28, 0.15, 0.05], [0.42, 0.32, 0.18, 0.08]
    counts, rng = [0] * 4, random.Random(731)
    for _ in range(40_000):
        token, _ = ch10.speculative_draw(target, draft, rng)
        counts[token] += 1
    emitted = [c / 40_000 for c in counts]
    tv = sum(abs(a - b) for a, b in zip(emitted, target)) / 2
    assert tv < 0.01                                     # emitted == target
    assert math.isclose(ch10.expected_accepted_tokens(0.9, 4), (1 - 0.9 ** 5) / 0.1)
    assert ch10.speculative_speedup(0.9, 4, 0.05) > 1.0  # idle fleet wins
    assert ch10.speculative_speedup(0.9, 4, 0.85) < 1.0  # saturated fleet loses


def test_grammar_mask_renormalizes_and_rejects_empty() -> None:
    masked = ch10.constrained_distribution([0.60, 0.25, 0.15], {1, 2})
    assert math.isclose(sum(masked["distribution"]), 1.0)
    assert math.isclose(masked["removed_mass"], 0.60)
    assert masked["distribution"][0] == 0.0
    try:
        ch10.constrained_distribution([0.5, 0.5], set())
    except ValueError:
        pass
    else:
        raise AssertionError("empty allowed set must raise")


def test_int8_is_near_lossless_while_int4_drifts() -> None:
    weight = torch.randn(32, 48)
    err8 = (weight - ch10.quantize_tensor(weight, 8)).abs().max().item()
    err4 = (weight - ch10.quantize_tensor(weight, 4)).abs().max().item()
    assert err8 < err4                                   # more bits, less error
    fp32, q8 = ch10.model_storage(torch.nn.Linear(64, 64), 8)
    assert q8 < fp32                                     # int8 is smaller


def test_finer_quant_block_lowers_rmse() -> None:
    signal = [2 * math.sin(i / 11) for i in range(256)]
    signal[47] = 38.0
    _, fine = ch10.quantize_blockwise(signal, 4, 16)
    _, coarse = ch10.quantize_blockwise(signal, 4, 256)
    assert fine < coarse


def test_disaggregation_flips_with_prompt_size() -> None:
    assert ch10.disaggregation_decision(0.25, 3.0)["disaggregate"] is False
    assert ch10.disaggregation_decision(4.0, 35.0)["disaggregate"] is True


def test_tenant_scoped_key_closes_the_presence_channel() -> None:
    assert ch10.shared_prefix_leak(include_tenant=False)["leaked_presence"] is True
    assert ch10.shared_prefix_leak(include_tenant=True)["leaked_presence"] is False


def test_reasoning_budget_saturates_then_declines() -> None:
    rows = ch10.budget_forcing_sweep([32, 256, 384, 768])
    accuracies = [r["accuracy"] for r in rows]
    assert accuracies[0] < accuracies[2]                 # rises early
    assert accuracies[-1] < max(accuracies)              # overthinking tail
    per_1k = [r["per_1k"] for r in rows]
    assert per_1k == sorted(per_1k, reverse=True)        # efficiency falls throughout
