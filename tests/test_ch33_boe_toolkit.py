"""Executable invariants for the Chapter 33 back-of-envelope toolkit.

Imports the tangled module ``code/ch33/_generated.py`` (produced from the
chapter's ``# @save`` cells by ``scripts/tangle.py``) and checks the arithmetic
the chapter claims: the three-term cost split and its cached-rate discount, the
TTFT + TPOT completion composition (cross-checked against Chapter 1's decode
simulator), that the serving-memory division sits on top of Chapter 3's
canonical KV calculator and reproduces the 7B-class GQA worked example, Little's
law and the device count it implies, the five-factor index-sizing product
(matching Appendix B's worked constant), loop-cost monotonicity and its
closed-form quadratic input count, the budget-to-steps inversion, and context
budget conservation with refusal of overcommitted plans.

Each chapter module is loaded under a unique name (``ch33_generated``,
``ch01_generated``, ``ch03_generated``) and registered in ``sys.modules``
before execution, because several chapters ship a module called ``_generated``
and frozen dataclasses under ``from __future__ import annotations`` resolve
their module at class-creation time.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(chapter: str, name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "code" / chapter / "_generated.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


ch33 = _load("ch33", "ch33_generated")
ch01 = _load("ch01", "ch01_generated")
ch03 = _load("ch03", "ch03_generated")

PRICE = ch33.TokenPrice(
    input_per_million=3.00, output_per_million=15.00, cached_input_per_million=0.30
)


# --- cost_per_query -----------------------------------------------------------


def test_cost_per_query_three_term_split() -> None:
    # 3,500 fresh in at $3/M + 250 out at $15/M, computed by hand.
    expected = (3_500 * 3.00 + 250 * 15.00) / 1e6
    assert ch33.cost_per_query(3_500, 250, PRICE) == pytest.approx(expected)
    assert expected == pytest.approx(0.01425)


def test_cost_per_query_cached_tokens_bill_at_cached_rate() -> None:
    warm = ch33.cost_per_query(3_500, 250, PRICE, cached_input_tokens=2_500)
    expected = (1_000 * 3.00 + 2_500 * 0.30 + 250 * 15.00) / 1e6
    assert warm == pytest.approx(expected)
    assert warm < ch33.cost_per_query(3_500, 250, PRICE)


def test_cost_per_query_rejects_cached_exceeding_input() -> None:
    with pytest.raises(ValueError):
        ch33.cost_per_query(100, 10, PRICE, cached_input_tokens=101)


# --- completion_ms ------------------------------------------------------------


def test_completion_composition() -> None:
    assert ch33.completion_ms(620.0, 250, 20.0) == pytest.approx(5_600.0)
    assert ch33.completion_ms(620.0, 1, 20.0) == pytest.approx(620.0)
    assert ch33.completion_ms(620.0, 0, 20.0) == 0.0


def test_completion_matches_ch01_simulated_stream() -> None:
    # TTFT + (N-1) * TPOT must reproduce the last arrival time exactly,
    # because request-level TPOT is defined as (t_N - t_1) / (N - 1).
    times = ch01.simulate_decode(prompt_tokens=800, output_tokens=250)
    composed = ch33.completion_ms(times[0], len(times), ch01.tpot_ms(times))
    assert composed == pytest.approx(times[-1])


# --- serving memory on top of Chapter 3's KV calculator -----------------------


def test_kv_worked_example_matches_ch03_calculator() -> None:
    gqa = ch03.KVConfig(
        "7B-class GQA-8", layers=32, query_heads=32, kv_heads=8,
        head_dim=128, bytes_per_scalar=2,
    )
    per_request = ch03.kv_bytes(gqa, tokens=4_096)
    assert per_request == 2**29                          # 0.5 GiB per request
    assert ch03.kv_bytes(gqa, tokens=4_096, batch=16) == 2**33   # 8 GiB fleet
    # The classic trap: query heads instead of KV heads is exactly 4x too big.
    mha_trap = ch03.KVConfig(
        "trap", layers=32, query_heads=32, kv_heads=32,
        head_dim=128, bytes_per_scalar=2,
    )
    assert ch03.kv_bytes(mha_trap, 4_096, 16) == 4 * 2**33


def test_kv_concurrency_per_device_division() -> None:
    per_request = 2**29
    fit = ch33.kv_concurrency_per_device(
        hbm_bytes=80e9, weights_bytes=16e9, kv_bytes_per_request=per_request
    )
    assert fit == int((80e9 * 0.9 - 16e9) // per_request) == 104
    # No memory left for cache -> zero requests, never negative.
    assert ch33.kv_concurrency_per_device(16e9, 16e9, per_request) == 0
    with pytest.raises(ValueError):
        ch33.kv_concurrency_per_device(80e9, 16e9, 0)


# --- Little's law and devices -------------------------------------------------


def test_littles_law_and_device_count() -> None:
    assert ch33.concurrent_requests(50, 5.6) == pytest.approx(280.0)
    assert ch33.devices_for_load(50, 5.6, 104) == math.ceil(280 * 1.3 / 104) == 4
    assert ch33.devices_for_load(150, 5.6, 104) == math.ceil(840 * 1.3 / 104) == 11
    # Tiny load still provisions one device.
    assert ch33.devices_for_load(0.01, 1.0, 104) == 1
    with pytest.raises(ValueError):
        ch33.concurrent_requests(-1, 1.0)
    with pytest.raises(ValueError):
        ch33.devices_for_load(50, 5.6, 0)


# --- index sizing -------------------------------------------------------------


def test_index_bytes_five_factor_product() -> None:
    # Appendix B's worked constant: 10M x 1024 x fp32 x 1.5 x 2 = 114.44 GiB.
    total = ch33.index_bytes(10_000_000, 1024, 4, overhead=1.5, replicas=2)
    assert total == pytest.approx(10_000_000 * 1024 * 4 * 1.5 * 2)
    assert total / 2**30 == pytest.approx(114.44, abs=0.01)
    # int8 at the same shape is exactly a quarter of fp32.
    assert ch33.index_bytes(10_000_000, 1024, 1, 1.5, 2) == pytest.approx(total / 4)
    with pytest.raises(ValueError):
        ch33.index_bytes(0, 1024, 4)
    with pytest.raises(ValueError):
        ch33.index_bytes(1_000, 1024, 4, overhead=0.5)


# --- loop cost and the step cap -----------------------------------------------


def test_loop_cost_matches_closed_form_input_count() -> None:
    # Uncached input tokens across k steps: k*c0 + g*k*(k-1)/2 (@eq-ch33-loop).
    k, c0, g, out = 20, 2_000, 800, 300
    n_in = k * c0 + g * k * (k - 1) // 2
    expected = (n_in * 3.00 + k * out * 15.00) / 1e6
    assert ch33.loop_cost(k, c0, g, out, PRICE) == pytest.approx(expected)


def test_loop_cost_monotone_in_steps_and_cheaper_cached() -> None:
    costs = [ch33.loop_cost(k, 2_000, 800, 300, PRICE) for k in range(1, 26)]
    assert all(a < b for a, b in zip(costs, costs[1:]))
    for k in (2, 10, 25):
        cold = ch33.loop_cost(k, 2_000, 800, 300, PRICE)
        hot = ch33.loop_cost(k, 2_000, 800, 300, PRICE, cached_resend=True)
        assert hot < cold
    with pytest.raises(ValueError):
        ch33.loop_cost(0, 2_000, 800, 300, PRICE)


def test_max_affordable_steps_inverts_loop_cost() -> None:
    for cached in (False, True):
        best = ch33.max_affordable_steps(
            0.25, 2_000, 800, 300, PRICE, cached_resend=cached
        )
        assert best >= 1
        assert ch33.loop_cost(best, 2_000, 800, 300, PRICE, cached) <= 0.25
        assert ch33.loop_cost(best + 1, 2_000, 800, 300, PRICE, cached) > 0.25
    # A budget below one step's cost returns 0, not an error.
    assert ch33.max_affordable_steps(1e-9, 2_000, 800, 300, PRICE) == 0


# --- context budget -----------------------------------------------------------


def test_context_budget_conserves_the_window() -> None:
    plan = ch33.context_budget(
        128_000, system=3_000, history=45_000, retrieval=12_000, output_reserve=4_000
    )
    assert sum(plan.values()) == 128_000
    assert plan["free"] == 64_000


def test_context_budget_refuses_overcommitment() -> None:
    with pytest.raises(ValueError):
        ch33.context_budget(
            32_000, system=3_000, history=18_000, retrieval=40_000, output_reserve=2_000
        )
    with pytest.raises(ValueError):
        ch33.context_budget(
            1_000, system=-1, history=0, retrieval=0, output_reserve=0
        )
    exact = ch33.context_budget(
        10_000, system=2_000, history=3_000, retrieval=4_000, output_reserve=1_000
    )
    assert exact["free"] == 0
