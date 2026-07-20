"""Tests for Chapter 25: interpretability primitives and the safety-case engine."""
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ch25 = _load("ch25_generated", "code/ch25/_generated.py")
ch02 = _load("ch02_generated_for_ch25", "code/ch02/_generated.py")


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    config = ch02.GPTConfig(vocab_size=64, block_size=16, d_model=32, n_heads=4, n_layers=2)
    return ch02.TinyGPT(config)


def _diagnostics(rank_drop=18.0, selectivity=0.13, behavior_change=True, on_target=2.4):
    return {
        "lens": {"rank_drop": rank_drop},
        "probe": {"selectivity": selectivity},
        "steering": {"behavior_change": behavior_change, "on_target": on_target},
    }


def _base_case():
    return {
        "system": {"name": "t", "scope": "s"},
        "top_claim": {"id": "G0", "text": "safe", "decision_rule": "block on gaps"},
        "claims": [
            {"id": "G1", "severity": "critical", "argument": "control", "hazard": "h",
             "controls": [{"id": "C1", "evidence": ["E-CONTAINMENT"]}]},
            {"id": "G2", "severity": "critical", "argument": "inability", "hazard": "h",
             "capability": {"score": 0.04, "threshold": 0.08, "elicitation_coverage": 0.55,
                            "required_coverage": 0.8, "sandbagging_check": "open"},
             "evidence": ["E-CAPABILITY"]},
            {"id": "G3", "severity": "high", "argument": "trustworthiness", "hazard": "h",
             "evidence": ["E-LENS", "E-PROBE", "E-STEER"], "unresolved_counterevidence": True},
        ],
        "evidence": [
            {"id": "E-CONTAINMENT", "kind": "control_test", "status": "pass", "scope": "x"},
            {"id": "E-CAPABILITY", "kind": "capability_eval", "status": "pass", "scope": "x"},
        ],
    }


# ---- interpretability primitives ----

def test_residual_states_shapes():
    model = _tiny_model()
    tokens = torch.randint(0, 64, (1, 12))
    states = ch25.residual_states(model, tokens)
    assert len(states) == model.config.n_layers + 1
    for state in states:
        assert tuple(state.shape) == (1, 12, model.config.d_model)


def test_residual_states_deterministic():
    model = _tiny_model()
    tokens = torch.randint(0, 64, (1, 10))
    a = ch25.residual_states(model, tokens)
    b = ch25.residual_states(model, tokens)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_logit_lens_final_state_matches_forward():
    model = _tiny_model()
    tokens = torch.randint(0, 64, (1, 8))
    final = ch25.residual_states(model, tokens)[-1]
    lens_logits = ch25.logit_lens(model, final)
    forward_logits = model(tokens)[0]
    assert torch.allclose(lens_logits, forward_logits, atol=1e-5)


def test_word_boundary_labels_surface_vs_next():
    ids = list(range(10))

    class Tok:
        def decode(self, xs, errors="strict"):
            return " w" if xs[0] % 2 == 0 else "x"

    surface = ch25.word_boundary_labels(Tok(), ids, predict_next=False)
    nxt = ch25.word_boundary_labels(Tok(), ids, predict_next=True)
    assert surface[0] == 1.0 and surface[1] == 0.0
    assert nxt[0] == surface[1] and nxt[8] == surface[9]


def test_probe_selectivity_positive_on_separable_data():
    rng = np.random.default_rng(0)
    labels = np.array([1.0, 0.0] * 60)
    features = rng.normal(size=(120, 8)) + labels[:, None] * np.array([3.0] + [0.0] * 7)
    result = ch25.probe_with_control(features, labels)
    assert result["accuracy"] >= 0.9
    assert result["selectivity"] > 0.2
    assert 0.4 <= result["control"] <= 0.7


def test_mean_difference_direction_is_unit_norm():
    features = np.array([[2.0, 0.0], [2.0, 0.0], [-2.0, 0.0], [-2.0, 0.0]])
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    direction = ch25.mean_difference_direction(features, labels)
    assert abs(float(direction.norm()) - 1.0) < 1e-5
    assert direction[0] > 0.99


def test_steering_changes_generation():
    model = _tiny_model()
    prompt = [1, 2, 3]
    direction = torch.zeros(model.config.d_model)
    direction[0] = 1.0
    baseline = ch25.generate_with_steering(model, prompt, 8, direction, 0.0, layer=1)
    steered = ch25.generate_with_steering(model, prompt, 8, direction, 25.0, layer=1)
    assert baseline != steered


# ---- safety-case engine ----

def test_integrated_case_blocks_on_gaps():
    report = ch25.build_safety_case(_base_case(), _diagnostics())
    assert report["decision"] == "BLOCK"
    assert report["summary"]["supported"] == 1
    assert report["summary"]["gaps"] == 2
    assert report["summary"]["blocking"] == ["G2", "G3"]


def test_case_is_reproducible():
    a = ch25.build_safety_case(_base_case(), _diagnostics())
    b = ch25.build_safety_case(_base_case(), _diagnostics())
    assert a == b


def test_diagnostics_alone_cannot_validate_control():
    by_id = {e["id"]: e for e in ch25.diagnostic_evidence(_diagnostics())}
    claim = {"id": "C", "severity": "critical", "argument": "control",
             "controls": [{"id": "C1", "evidence": ["E-PROBE", "E-STEER"]}]}
    result = ch25.evaluate_claim(claim, by_id)
    assert result["status"] == "gap"
    assert any("diagnostics" in r for r in result["reasons"])


def test_inability_requires_elicitation_and_integrity():
    case = _base_case()
    by_id = {e["id"]: e for e in case["evidence"] + ch25.diagnostic_evidence(_diagnostics())}
    claim = deepcopy(case["claims"][1])
    assert ch25.evaluate_claim(claim, by_id)["status"] == "gap"
    claim["capability"]["elicitation_coverage"] = 0.9
    claim["capability"]["sandbagging_check"] = "pass"
    assert ch25.evaluate_claim(claim, by_id)["status"] == "supported"


def test_trustworthiness_needs_two_nondiagnostic_classes():
    diag = ch25.diagnostic_evidence(_diagnostics())
    extra = [{"id": "E-AUDIT", "kind": "behavioral_audit", "status": "pass"},
             {"id": "E-LINEAGE", "kind": "data_lineage", "status": "pass"}]
    by_id = {e["id"]: e for e in diag + extra}
    claim = {"id": "G3", "severity": "high", "argument": "trustworthiness",
             "evidence": ["E-LENS", "E-AUDIT", "E-LINEAGE"], "unresolved_counterevidence": False}
    assert ch25.evaluate_claim(claim, by_id)["status"] == "supported"


def test_dangling_evidence_is_rejected():
    case = _base_case()
    case["claims"][0]["controls"][0]["evidence"] = ["missing"]
    try:
        ch25.build_safety_case(case, _diagnostics())
    except ValueError as exc:
        assert "missing evidence" in str(exc)
    else:
        raise AssertionError("expected ValueError for dangling evidence")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
