# Auto-generated from chapters/25-alignment-interpretability-governance.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import torch


@torch.inference_mode()
def residual_states(model, tokens):
    """Return the residual stream after the embedding and after each block.

    This is the transformer's forward pass with the intermediate vectors
    exposed: every interpretability tool in this chapter reads one of the
    returned states rather than only the final logits. The list has one entry
    more than the model has blocks — index 0 is the token-plus-position
    embedding, index i is the stream after block i-1.

    Args:
        model: A decoder-only model exposing ``token_embedding``,
            ``position_embedding``, and ``blocks`` (the Chapter 2 ``TinyGPT``).
        tokens: Token IDs shaped ``(1, T)``.

    Returns:
        A list of tensors shaped ``(1, T, d)``, one per inspection point.
    """
    model.eval()
    _, steps = tokens.shape
    positions = torch.arange(steps)
    x = model.token_embedding(tokens) + model.position_embedding(positions)
    states = [x.clone()]
    for block in model.blocks:
        x, _ = block(x)
        states.append(x.clone())
    return states


import torch


@torch.inference_mode()
def logit_lens(model, state):
    """Decode an intermediate residual with the model's final readout.

    Applies the model's final normalization and unembedding to a residual
    state, producing the tokens the model's own vocabulary geometry would
    score highly *if this were the last layer*. It is the model reading its
    own mind partway through, in its own basis.

    Args:
        model: A model exposing ``final_norm`` and ``lm_head``.
        state: A residual state shaped ``(1, T, d)`` from ``residual_states``.

    Returns:
        Logits shaped ``(1, T, vocab_size)``.
    """
    return model.lm_head(model.final_norm(state))


import numpy as np


def word_boundary_labels(tokenizer, token_ids, predict_next=False):
    """Label each position by its relation to a word boundary.

    Two concepts share one helper. With ``predict_next=False`` the label is a
    surface fact about the current token — does it begin a new word (a leading
    space or newline)? With ``predict_next=True`` the label is a prediction the
    model must compute — will the *next* token begin a word? The surface label
    is available at the embedding; the predictive one requires integrating the
    current partial word, so it should strengthen deeper in the stack.

    Args:
        tokenizer: A tokenizer exposing ``decode``.
        token_ids: The token-ID stream to label.
        predict_next: Label the next token's boundary status instead of this one.

    Returns:
        A float array of 0/1 labels, one per position.
    """
    pieces = [tokenizer.decode([t], errors="replace") for t in token_ids]
    starts = np.array([1.0 if p[:1] in (" ", "\n") else 0.0 for p in pieces])
    if not predict_next:
        return starts
    labels = np.zeros(len(token_ids))
    labels[:-1] = starts[1:]
    return labels


import numpy as np
import torch


def collect_residuals(model, token_ids, block_size):
    """Gather per-layer residual activations over the whole stream.

    Slides non-overlapping windows across the token stream, runs the residual
    hook on each, and stacks the results so a probe can be fit at any layer.
    Returns the covering index map so labels computed on the full stream line
    up with the collected activations position for position.

    Args:
        model: The model to inspect.
        token_ids: The full token-ID stream.
        block_size: Window length; also the model's context limit.

    Returns:
        A tuple ``(states, index_map)`` where ``states[layer]`` is an array
        shaped ``(N, d)`` and ``index_map`` gives each row's stream position.
    """
    layers = len(model.blocks) + 1
    buckets = [[] for _ in range(layers)]
    index_map = []
    for start in range(0, len(token_ids) - block_size, block_size):
        window = token_ids[start:start + block_size]
        states = residual_states(model, torch.tensor([window]))
        for layer, state in enumerate(states):
            buckets[layer].append(state[0].numpy())
        index_map.extend(range(start, start + block_size))
    return [np.concatenate(bucket) for bucket in buckets], np.array(index_map)


import numpy as np
from sklearn.linear_model import LogisticRegression


def probe_with_control(features, labels, seed=25, train_frac=0.7):
    """Fit a linear probe and an equal-capacity shuffled-label control.

    The probe measures whether the concept is linearly decodable from the
    activations; the control fits the same model class to a random relabeling.
    Selectivity — probe accuracy minus control accuracy — is the honest signal,
    because a probe that only memorizes labels scores as well on the shuffle.

    Args:
        features: Activations shaped ``(N, d)``.
        labels: Binary labels shaped ``(N,)``.
        seed: Seed for the control-label permutation.
        train_frac: Fraction of positions used to fit; the rest are held out.

    Returns:
        A dict with ``accuracy``, ``control``, ``selectivity``, and ``majority``.
    """
    n_train = int(train_frac * len(features))
    x_train, x_test = features[:n_train], features[n_train:]
    y_train, y_test = labels[:n_train], labels[n_train:]
    probe = LogisticRegression(max_iter=2000).fit(x_train, y_train)
    accuracy = probe.score(x_test, y_test)
    shuffled = np.random.default_rng(seed).permutation(labels)
    control = LogisticRegression(max_iter=2000).fit(x_train, shuffled[:n_train])
    control_accuracy = control.score(x_test, shuffled[n_train:])
    majority = max(y_test.mean(), 1 - y_test.mean())
    return {
        "accuracy": round(float(accuracy), 3),
        "control": round(float(control_accuracy), 3),
        "selectivity": round(float(accuracy - control_accuracy), 3),
        "majority": round(float(majority), 3),
    }


import numpy as np
import torch


def mean_difference_direction(features, labels):
    """Return the unit direction separating the two labeled activation groups.

    The difference of class means is the simplest causal handle on a concept a
    probe found: it points from the average "off" activation toward the average
    "on" one. Adding it to the residual stream during generation tests whether
    the concept the probe decoded actually drives behavior.

    Args:
        features: Activations shaped ``(N, d)``.
        labels: Binary labels shaped ``(N,)``.

    Returns:
        A unit-norm float32 tensor shaped ``(d,)``.
    """
    difference = features[labels == 1].mean(0) - features[labels == 0].mean(0)
    difference = difference / np.linalg.norm(difference)
    return torch.tensor(difference, dtype=torch.float32)


@torch.inference_mode()
def generate_with_steering(model, prompt_ids, max_new_tokens, direction, alpha, layer):
    """Greedily generate while adding a steering vector at one layer.

    Reproduces the residual hook of ``residual_states`` but, after block
    ``layer``, adds ``alpha * direction`` to the stream before the remaining
    blocks and the readout run. ``alpha = 0`` is ordinary generation; positive
    and negative doses push the concept up and down.

    Args:
        model: The model to steer.
        prompt_ids: The prompt as a list of token IDs.
        max_new_tokens: How many tokens to append.
        direction: A ``(d,)`` steering direction.
        alpha: The dose; scales the added direction.
        layer: Add the direction after this many blocks (1-indexed).

    Returns:
        The prompt with generated token IDs appended, as a list.
    """
    block_size = model.config.block_size
    output = list(prompt_ids)
    for _ in range(max_new_tokens):
        tokens = torch.tensor([output[-block_size:]])
        positions = torch.arange(tokens.size(1))
        x = model.token_embedding(tokens) + model.position_embedding(positions)
        for index, block in enumerate(model.blocks):
            x, _ = block(x)
            if index == layer - 1:
                x = x + alpha * direction
        logits = model.lm_head(model.final_norm(x))[0, -1]
        output.append(int(logits.argmax()))
    return output


DIAGNOSTIC_KINDS = {"logit_lens", "linear_probe", "activation_steering", "sae_feature"}


def diagnostic_evidence(diagnostics):
    """Turn internal measurements into scoped, non-dispositive evidence items.

    The lens, probe, and steering results from the real model each become one
    evidence item tagged with a diagnostic ``kind``. Tagging matters: the claim
    rules treat diagnostic evidence as hypothesis-generating, never as proof of
    a control or a stable objective, so these items can support a narrow
    statement but cannot by themselves discharge a safety claim.

    Args:
        diagnostics: A dict with ``lens``, ``probe``, and ``steering`` results.

    Returns:
        A list of three evidence dicts with ``id``, ``kind``, ``status``,
        ``value``, and ``limitation``.
    """
    lens, probe, steer = diagnostics["lens"], diagnostics["probe"], diagnostics["steering"]
    return [
        {"id": "E-LENS", "kind": "logit_lens",
         "status": "pass" if lens["rank_drop"] > 0 else "fail", "value": lens["rank_drop"],
         "limitation": "a decoded token is not a causal computation"},
        {"id": "E-PROBE", "kind": "linear_probe",
         "status": "pass" if probe["selectivity"] > 0.02 else "fail", "value": probe["selectivity"],
         "limitation": "decodability does not prove the policy uses the feature"},
        {"id": "E-STEER", "kind": "activation_steering",
         "status": "pass" if steer["behavior_change"] else "fail", "value": steer["on_target"],
         "limitation": "a local edit may be off-target or fail out of distribution"},
    ]


def evaluate_claim(claim, evidence_by_id):
    """Judge one scoped safety argument against explicit, per-pattern rules.

    Control arguments must rest on passing evidence that is not purely
    diagnostic. Inability arguments require a below-threshold score, sufficient
    elicitation coverage, a closed sandbagging check, and passing evidence.
    Trustworthiness arguments require passing evidence, at least two independent
    non-diagnostic evidence classes, and no unresolved counterevidence. Any
    unmet condition makes the claim a gap; the reasons are returned so a
    reviewer sees exactly why.

    Args:
        claim: One claim dict from the case.
        evidence_by_id: All evidence items keyed by ``id``.

    Returns:
        A dict with ``id``, ``severity``, ``argument``, ``status``, ``reasons``.
    """
    def passing(ids):
        return bool(ids) and all(evidence_by_id[i]["status"] == "pass" for i in ids)

    reasons = []
    argument = claim["argument"]
    if argument == "control":
        controls = claim.get("controls", [])
        if not controls:
            reasons.append("no control is named")
        for control in controls:
            ids = control.get("evidence", [])
            kinds = {evidence_by_id[i]["kind"] for i in ids}
            if not passing(ids):
                reasons.append(f"{control['id']} lacks passing evidence")
            if kinds and kinds <= DIAGNOSTIC_KINDS:
                reasons.append(f"{control['id']} rests only on internal diagnostics")
    elif argument == "inability":
        cap = claim["capability"]
        if cap["score"] > cap["threshold"]:
            reasons.append("capability score exceeds the deployment threshold")
        if cap["elicitation_coverage"] < cap["required_coverage"]:
            reasons.append("elicitation coverage is below the declared minimum")
        if cap.get("sandbagging_check") != "pass":
            reasons.append("evaluation-integrity or sandbagging checks remain open")
        if not passing(claim.get("evidence", [])):
            reasons.append("capability evidence is missing or failing")
    else:  # trustworthiness
        ids = claim.get("evidence", [])
        kinds = {evidence_by_id[i]["kind"] for i in ids}
        if not passing(ids):
            reasons.append("trustworthiness evidence is missing or failing")
        if len(kinds - DIAGNOSTIC_KINDS) < 2:
            reasons.append("fewer than two independent non-diagnostic evidence classes")
        if claim.get("unresolved_counterevidence", True):
            reasons.append("counterevidence remains unresolved")
    return {"id": claim["id"], "severity": claim["severity"], "argument": argument,
            "status": "supported" if not reasons else "gap", "reasons": reasons}


def build_safety_case(case, diagnostics):
    """Assemble a case, evaluate every claim, and return a fail-closed decision.

    Joins the diagnostic evidence to the case's own evidence, rejects dangling
    evidence references, evaluates each claim under ``evaluate_claim``, and
    blocks deployment when any high- or critical-severity claim is a gap. The
    decision is structural: it does not average evidence into one score, so the
    blocking reason is always a specific claim, not a threshold on a number.

    Args:
        case: The deployment case (system, top claim, claims, base evidence).
        diagnostics: Real interpretability measurements for the diagnostic items.

    Returns:
        A report dict with ``claims`` results, a ``summary``, and ``decision``.

    Raises:
        ValueError: If a claim references an evidence ID that does not exist.
    """
    evidence = case["evidence"] + diagnostic_evidence(diagnostics)
    by_id = {item["id"]: item for item in evidence}
    for claim in case["claims"]:
        referenced = list(claim.get("evidence", []))
        for control in claim.get("controls", []):
            referenced += control.get("evidence", [])
        missing = sorted(set(referenced) - set(by_id))
        if missing:
            raise ValueError(f"{claim['id']} references missing evidence: {missing}")
    results = [evaluate_claim(claim, by_id) for claim in case["claims"]]
    blocking = [r["id"] for r in results
                if r["status"] == "gap" and r["severity"] in {"high", "critical"}]
    return {
        "claims": results,
        "summary": {"supported": sum(r["status"] == "supported" for r in results),
                    "gaps": sum(r["status"] == "gap" for r in results),
                    "blocking": blocking},
        "decision": "BLOCK" if blocking else "APPROVE",
    }
