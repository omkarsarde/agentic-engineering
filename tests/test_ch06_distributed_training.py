"""Executable invariants for Chapter 6's distributed-training arithmetic.

Imports only the tangled module ``code/ch06/_generated.py`` (the chapter's
``# @save`` cells in document order) under a chapter-unique module name.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch06_generated", ROOT / "code" / "ch06" / "_generated.py"
)
ch06 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["ch06_generated"] = ch06
_SPEC.loader.exec_module(ch06)

GIB = 1024**3


def test_param_count_reproduces_published_model_sizes() -> None:
    """The GQA-aware calculator lands on the real 7B and 70B totals."""
    seven = ch06.param_count(ch06.LLAMA_7B)
    seventy = ch06.param_count(ch06.LLAMA_70B)
    assert seven["total"] == pytest.approx(6.74e9, rel=0.005)
    assert seventy["total"] == pytest.approx(70.6e9, rel=0.005)
    for counts in (seven, seventy):
        assert counts["attention"] + counts["ffn"] + counts["embeddings"] == counts["total"]


def test_state_ledger_16_bytes_and_zero_stage_algebra() -> None:
    """fp32 and mixed both cost 16 bytes/param; each stage shards its promise."""
    params = 1_000_000
    assert ch06.state_ledger(params, "fp32")["total"] == 16 * params
    assert ch06.state_ledger(params, "mixed")["total"] == 16 * params
    shards = 8
    stage1 = ch06.state_ledger(params, "mixed", zero_stage=1, shards=shards)
    assert stage1["Adam moments"] == 8 * params / shards
    assert stage1["master weights"] == 4 * params / shards
    assert stage1["gradients"] == 2 * params  # not yet sharded
    stage2 = ch06.state_ledger(params, "mixed", zero_stage=2, shards=shards)
    assert stage2["gradients"] == 2 * params / shards
    assert stage2["weights"] == 2 * params  # not yet sharded
    stage3 = ch06.state_ledger(params, "mixed", zero_stage=3, shards=shards)
    assert stage3["total"] == 16 * params / shards


def test_activation_policies_are_ordered_and_match_the_formula() -> None:
    """materialize > fused > recompute, with the exact leading-order bytes."""
    cfg = ch06.LLAMA_7B
    s, b, h = cfg.seq_len, 2, cfg.d_model
    naive = ch06.activation_bytes_per_layer(cfg, b, "materialize scores")
    fused = ch06.activation_bytes_per_layer(cfg, b, "fused attention")
    recompute = ch06.activation_bytes_per_layer(cfg, b, "full recompute")
    assert naive == 34 * s * b * h + 5 * cfg.n_heads * s * s * b
    assert fused == 34 * s * b * h
    assert recompute == 2 * s * b * h
    assert naive > fused > recompute
    with pytest.raises(ValueError):
        ch06.activation_bytes_per_layer(cfg, 1, "wishful thinking")


def test_measured_saved_bytes_shrink_under_checkpointing() -> None:
    """The live autograd measurement agrees with the policy story."""
    ch02 = ch06.load_chapter_module("ch02", "ch02_generated")
    config = ch02.GPTConfig(
        vocab_size=64, block_size=32, d_model=32, n_heads=4, n_layers=2, mlp_ratio=2
    )
    torch.manual_seed(6)
    model = ch02.TinyGPT(config)
    tokens = torch.randint(config.vocab_size, (2, config.block_size))
    targets = torch.roll(tokens, -1, dims=1)
    plain = ch06.measure_saved_bytes(model, tokens, targets, checkpoint_blocks=False)
    model.zero_grad(set_to_none=True)
    checkpointed = ch06.measure_saved_bytes(model, tokens, targets, checkpoint_blocks=True)
    assert 0 < checkpointed < plain / 2


def test_tensor_parallel_ffn_matches_the_unsharded_computation() -> None:
    """Column-then-row sharding plus one sum reproduces the reference."""
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    w_up, w_down = torch.randn(64, 256) / 8, torch.randn(256, 64) / 16
    reference = torch.relu(x @ w_up) @ w_down
    for shards in (2, 4):
        combined = ch06.tensor_parallel_ffn(x, w_up, w_down, shards)
        assert torch.allclose(combined, reference, atol=1e-5)


def test_tp_comm_volume_is_zero_alone_and_scales_with_the_ring_factor() -> None:
    """No traffic at tp=1; ring factor 2(tp-1)/tp over four all-reduces."""
    cfg = ch06.LLAMA_7B
    assert ch06.tp_comm_bytes_per_layer(cfg, 1, 1) == 0.0
    payload = cfg.seq_len * cfg.d_model * 2
    assert ch06.tp_comm_bytes_per_layer(cfg, 1, 8) == pytest.approx(4 * 2 * 7 / 8 * payload)


def test_one_f1b_order_covers_every_operation_once() -> None:
    """Each stage's order holds every forward and backward exactly once."""
    stages, microbatches = 4, 8
    for stage in range(stages):
        ops = ch06.one_f1b_order(stage, stages, microbatches)
        assert sorted(op for op in ops if op[0] == "F") == [("F", m) for m in range(1, 9)]
        assert sorted(op for op in ops if op[0] == "B") == [("B", m) for m in range(1, 9)]
        warmup = min(microbatches, stages - stage)
        assert all(kind == "F" for kind, _ in ops[:warmup])


def test_simulated_pipeline_idle_matches_the_bubble_formula() -> None:
    """The measured idle fraction equals (S-1)/(m+S-1), any op-time scale."""
    for stages, microbatches in ((4, 8), (8, 32), (8, 8)):
        for backward_time in (1.0, 2.0):
            _, _, idle = ch06.simulate_pipeline(
                stages, microbatches, forward_time=1.0, backward_time=backward_time
            )
            formula = (stages - 1) / (microbatches + stages - 1)
            assert idle == pytest.approx(formula, abs=1e-9)


def test_plan_simulator_flips_verdicts_between_7b_and_70b() -> None:
    """Pure DP fits the 7B on 8 devices and fails the 70B on 64."""
    small = ch06.evaluate_plan(ch06.LLAMA_7B, 1, 1, 8, global_batch=64)
    assert small["fits"] and small["bubble"] == 0.0
    big = ch06.evaluate_plan(ch06.LLAMA_70B, 1, 1, 64, global_batch=128)
    assert not big["fits"]
    sharded = ch06.evaluate_plan(ch06.LLAMA_70B, 8, 2, 4, global_batch=128)
    assert sharded["fits"]
    assert sharded["bubble"] == pytest.approx(1 / 33)


def test_plan_table_enumerates_only_legal_factorizations(capsys) -> None:
    """Every printed plan multiplies to the device count and divides cleanly."""
    rows = ch06.plan_table(ch06.LLAMA_70B, devices=64, global_batch=128)
    capsys.readouterr()
    assert len(rows) == 16
    for row in rows:
        tp, pp, dp = (int(part.split("=")[1]) for part in row["plan"].split())
        assert tp * pp * dp == 64
        assert ch06.LLAMA_70B.n_layers % pp == 0
        assert 128 % dp == 0


def test_expert_dispatch_payload_worked_number() -> None:
    """The 2*T*d*k*bytes*offdevice estimate lands on the chapter's 7.0 GiB."""
    payload = ch06.expert_dispatch_bytes(tokens=8 * 8192, d_model=4096, top_k=8)
    assert payload == pytest.approx(2 * 65536 * 4096 * 8 * 2 * 0.875)
    assert payload / GIB == pytest.approx(7.0, abs=0.05)


def test_measured_matmul_roofline_is_positive_and_finite() -> None:
    """The benchmark returns a sane FLOP rate at a small size."""
    rate = ch06.measured_matmul_flops(n=128, repeats=3)
    assert math.isfinite(rate) and rate > 1e6


def test_wsd_multiplier_ramps_holds_and_decays() -> None:
    """Warmup rises toward 1, the plateau is exactly 1, the end reaches 0."""
    total = 200
    assert ch06.wsd_multiplier(0, total) < 1.0
    assert ch06.wsd_multiplier(total // 2, total) == 1.0
    assert ch06.wsd_multiplier(total - 1, total) == 0.0
    values = [ch06.wsd_multiplier(step, total) for step in range(total)]
    assert max(values) == 1.0 and min(values) >= 0.0


def test_young_interval_minimizes_the_waste_model() -> None:
    """Waste at the Young optimum beats half and double the interval."""
    save_s, mtbf_s = 30.0, 10_000.0
    best = ch06.young_interval(save_s, mtbf_s)
    assert best == pytest.approx(math.sqrt(2 * save_s * mtbf_s))
    at_best = ch06.checkpoint_waste(best, save_s, mtbf_s)
    assert at_best < ch06.checkpoint_waste(best / 2, save_s, mtbf_s)
    assert at_best < ch06.checkpoint_waste(best * 2, save_s, mtbf_s)
    assert at_best == pytest.approx(save_s / best + best / (2 * mtbf_s) + 300 / mtbf_s)


def test_run_economics_follows_6nd_and_is_fleet_invariant_in_hours() -> None:
    """Device-hours ignore fleet size; wall time and megawatts follow it."""
    small = ch06.run_economics(6.74e9, 2e12, 1_024)
    big = ch06.run_economics(6.74e9, 2e12, 2_048)
    assert small["flops"] == 6 * 6.74e9 * 2e12
    assert small["device_hours"] == pytest.approx(big["device_hours"])
    assert small["wall_days"] == pytest.approx(2 * big["wall_days"])
    assert big["megawatts"] == pytest.approx(2 * small["megawatts"])
    assert small["cost"] == pytest.approx(small["device_hours"] * 2.50)


def test_capital_recovery_factor_annualizes_sensibly() -> None:
    """CRF exceeds the rate, matches the closed form, and repays principal."""
    crf = ch06.capital_recovery_factor(0.12, 4)
    assert crf == pytest.approx(0.12 * 1.12**4 / (1.12**4 - 1))
    assert crf > 0.12
    assert crf * 4 > 1.0  # four payments repay principal plus interest
