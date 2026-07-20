"""Executable invariants for Chapter 31's world-model / VLA / embodied stack.

Imports only the tangled module ``code/ch31/_generated.py`` (the chapter's
``# @save`` cells in document order) under a chapter-unique module name, then
checks the load-bearing claims: the action codec's error bound, the
compounding gap between one-step and free-running model error, that
observe-and-replan beats open-loop planning while a weak model is not rescued
by feedback, that the token policy navigates closed-loop, and that grounded
skill selection flips the language-only winner.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch31_generated", ROOT / "code" / "ch31" / "_generated.py"
)
ch31 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["ch31_generated"] = ch31
_SPEC.loader.exec_module(ch31)

# Train the shared models once (smaller than the chapter's, still competent).
GOOD = ch31.train_dynamics(4000, seed=0, steps=700)
WEAK = ch31.train_dynamics(40, seed=1, steps=120)
POLICY, DEMO_OBS, DEMO_ACTS = ch31.train_policy(seed=0, steps=800)


# --- action codec ---------------------------------------------------------

def test_codec_roundtrip_within_half_bin_bound() -> None:
    action = np.array([0.30, -0.55, 0.80, -0.10, 0.05, -0.95, 1.0])
    assert ch31.roundtrip_error(action, bins=256) <= 1.0 / 255 + 1e-12
    # a fine grid tightens the bound; a coarse grid loosens it
    assert ch31.roundtrip_error(action, bins=21) <= 1.0 / 20 + 1e-12


def test_codec_is_invertible_at_bin_centers() -> None:
    tokens = np.array([0, 5, 10, 20])
    back = ch31.tokenize_action(ch31.detokenize_action(tokens, 21), 21)
    assert np.array_equal(back, tokens)


def test_codec_rejects_unnormalized_input_and_tiny_vocab() -> None:
    for bad in (np.array([1.5, 0.0]), np.array([0.0, -2.0])):
        try:
            ch31.tokenize_action(bad, 256)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on out-of-range action")
    try:
        ch31.tokenize_action(np.array([0.0]), bins=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on bins < 2")


def test_dct_roundtrip_and_smooth_compression() -> None:
    steps = np.linspace(0, 1, 16)
    chunk = 0.8 * np.sin(2 * np.pi * steps) + 0.1 * steps
    coef = ch31.dct_ii(chunk)
    assert np.allclose(ch31.idct_ii(coef), chunk, atol=1e-9)  # exact round trip
    # keeping more low-frequency coefficients cannot increase reconstruction error
    errs = []
    for keep in (2, 4, 6, 16):
        trimmed = coef.copy()
        trimmed[keep:] = 0.0
        errs.append(np.sqrt(np.mean((ch31.idct_ii(trimmed) - chunk) ** 2)))
    assert errs[0] >= errs[1] >= errs[2] >= errs[3]
    assert errs[-1] < 1e-9  # keeping all coeffs is lossless


# --- true dynamics --------------------------------------------------------

def test_true_step_stays_in_box_and_has_state_dependent_drift() -> None:
    rng = np.random.default_rng(0)
    s = rng.uniform(0, ch31.SIZE, size=(50, 2))
    a = rng.uniform(-1, 1, size=(50, 2))
    nxt = ch31.true_step(s, a)
    assert nxt.min() >= 0.0 and nxt.max() <= ch31.SIZE
    # a pure horizontal command still moves in y, because the current does
    moved = ch31.true_step(np.array([3.0, 3.0]), np.array([1.0, 0.0]))
    assert abs(moved[1] - 3.0) > 1e-3


# --- world model: one-step accuracy vs compounding ------------------------

def _one_step_rmse(model) -> float:
    s, a, d = ch31.collect_transitions(2000, seed=99)
    import torch
    with torch.no_grad():
        pred = model(torch.as_tensor(s), torch.as_tensor(a)).numpy()
    return float(np.sqrt(((pred - d) ** 2).sum(-1)).mean())


def test_good_model_predicts_one_step_far_better_than_weak() -> None:
    good, weak = _one_step_rmse(GOOD), _one_step_rmse(WEAK)
    assert good < 0.20             # well under a tenth of the box
    assert weak > 2.0 * good       # the under-fit model is much worse


def test_error_compounds_under_free_running_rollout() -> None:
    one_step, free_run = ch31.rollout_error(GOOD, horizon=20, n_traj=120, seed=7)
    # one-step error stays roughly flat; free-running error climbs far above it
    assert one_step[-1] < 2.0 * one_step[0] + 0.05
    assert free_run[-1] > one_step[-1] + 0.3
    assert free_run[-1] > free_run[4] > free_run[0]  # monotone-ish growth


# --- planning: decision-grade behavior ------------------------------------

def test_replanning_beats_open_loop_at_a_long_horizon() -> None:
    open_loop = ch31.planning_success(GOOD, horizon=8, replan=False, n_starts=16)
    replan = ch31.planning_success(GOOD, horizon=8, replan=True, n_starts=16)
    assert replan > open_loop
    assert replan >= 0.8


def test_short_and_long_open_loop_horizons_differ() -> None:
    short = ch31.planning_success(GOOD, horizon=1, replan=False, n_starts=16)
    long_ = ch31.planning_success(GOOD, horizon=16, replan=False, n_starts=16)
    assert short >= long_  # committing longer to the imagined rollout hurts


def test_feedback_does_not_rescue_an_inadequate_model() -> None:
    good_replan = ch31.planning_success(GOOD, horizon=1, replan=True, n_starts=16)
    weak_replan = ch31.planning_success(WEAK, horizon=1, replan=True, n_starts=16)
    assert weak_replan < good_replan
    assert good_replan >= 0.8


# --- VLA token policy -----------------------------------------------------

def test_policy_matches_expert_tokens_and_navigates_closed_loop() -> None:
    import torch
    targets = ch31.tokenize_action(DEMO_ACTS, ch31.ACTION_BINS)
    with torch.no_grad():
        p0 = POLICY.logits(torch.as_tensor(DEMO_OBS), None, 0).argmax(-1).numpy()
    within_one = float((np.abs(p0 - targets[:, 0]) <= 1).mean())
    assert within_one >= 0.85
    rate, paths = ch31.vla_success(POLICY, n=16, seed=5)
    assert rate >= 0.8
    assert all(path.shape[1] == 2 for path, _, _ in paths)


def test_policy_action_is_bounded_and_two_dimensional() -> None:
    action = POLICY.act([1.0, 1.0, 4.0, 3.5])
    assert action.shape == (2,)
    assert np.all(np.abs(action) <= 1.0)


# --- grounded skills + safety gate ----------------------------------------

def test_saycan_grounding_flips_the_language_only_winner() -> None:
    skills = ["pick up sponge", "wipe table", "pick up mug", "open drawer"]
    p_lang = [0.35, 0.45, 0.05, 0.15]
    p_aff = [0.95, 0.15, 0.90, 0.85]
    best, products = ch31.saycan_select(skills, p_lang, p_aff)
    assert skills[int(np.argmax(p_lang))] == "wipe table"   # language alone
    assert best == "pick up sponge"                          # grounding flips it
    assert products[0] == max(products)


def test_reversibility_gate_escalates_irreversible_actions() -> None:
    gentle = np.array([0.3, 0.2, 0.0])
    lunge = np.array([1.0, 0.9, -1.0])
    assert ch31.reversibility_gate(gentle) is True
    assert ch31.reversibility_gate(lunge) is False               # denied by default
    assert ch31.reversibility_gate(lunge, approve=lambda p: True) is True
    assert ch31.reversibility_gate(lunge, approve=lambda p: False) is False
