# Auto-generated from chapters/07-post-training.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


from dataclasses import dataclass


@dataclass(frozen=True)
class SpecRule:
    """One versioned behavior rule: the atom a dataset and an eval can trace to.

    A rule is written once and referenced by ``rule_id`` everywhere else, so a
    regression can be answered with two questions a prose policy cannot: which
    intended behavior produced this data, and which data must be reconsidered
    when the rule changes. ``applies_when`` scopes it, ``expected`` and
    ``prohibited`` state the behavior, and ``examples`` anchors it to concrete
    user turns.
    """

    rule_id: str
    applies_when: str
    expected: str
    prohibited: str
    examples: tuple[str, ...]


REFUND_ID_RULE = SpecRule(
    rule_id="refund.identity.v3",
    applies_when="account-specific refund or status question without an order id",
    expected="ask for the order id; do not estimate eligibility",
    prohibited="promising, denying, or estimating a refund before identity is verified",
    examples=("can i get a refund", "is my order eligible for a refund"),
)


DEMOS = [
    ("how do i reset my password", "open settings, then security, then reset password."),
    ("what are your support hours", "we answer messages every day from nine to five."),
    ("can i get a refund", "please share your order id and i will check eligibility."),
    ("is my order eligible for a refund", "please share your order id and i will check eligibility."),
    ("how do i track my order", "open your account, choose orders, then select track."),
    ("do you offer gift cards", "yes, we offer gift cards in several amounts."),
    ("i want to cancel my subscription", "open settings, choose billing, then cancel subscription."),
    ("where is my package", "please share your order id and i will check the status."),
    ("can you help me", "yes, tell me what you need and i will help."),
    ("thank you", "you are welcome, let me know if you need anything else."),
]
DEMO_RULES = {2: ("refund.identity.v3",), 3: ("refund.identity.v3",), 7: ("refund.identity.v3",)}


def rows_for_rule(rule_id: str) -> list[int]:
    """Return the demonstration indices governed by a rule.

    This is the traceability query a specification exists to answer: when
    ``rule_id`` changes, exactly these training rows must be re-reviewed and
    possibly relabeled, and no others.

    Args:
        rule_id: The rule whose data lineage we want.

    Returns:
        Indices into ``DEMOS`` tagged with ``rule_id``.
    """
    return [i for i, rules in DEMO_RULES.items() if rule_id in rules]


import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _find_root(anchor: str = "code/ch02/_generated.py") -> Path:
    starts = []
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    starts.append(Path.cwd().resolve())
    for start in starts:
        for base in [start, *start.parents]:
            if (base / anchor).exists():
                return base
    raise FileNotFoundError(anchor)


def load_module(name: str, relative_path: str):
    """Import a committed chapter's tangled module under an explicit name.

    Args:
        name: The module name to register (avoids cross-chapter collisions).
        relative_path: Path to the ``_generated.py``, relative to the repo root.

    Returns:
        The imported module object.
    """
    spec = importlib.util.spec_from_file_location(name, _find_root() / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ch02 = load_module("ch02_generated", "code/ch02/_generated.py")
torch.set_num_threads(1)


CANDIDATES = {
    "can i get a refund": [
        ("please share your order id and i will check eligibility.", 1.0, 1.0, 0.25),
        ("please share your order id and i will check eligibility right away for you.", 1.0, 1.0, 0.85),
        ("sure, i can approve a full refund right now for you.", 0.0, 0.0, 0.50),
    ],
    "how do i reset my password": [
        ("open settings, then security, then reset password.", 1.0, 1.0, 0.25),
        ("open settings, then security, then reset password and follow the emailed steps.", 1.0, 1.0, 0.85),
        ("just tell me your current password and i will reset it.", 0.0, 0.0, 0.50),
    ],
    "where is my package": [
        ("please share your order id and i will check the status.", 1.0, 1.0, 0.25),
        ("please share your order id and i will check the status and arrival window.", 1.0, 1.0, 0.85),
        ("it will definitely arrive tomorrow, i guarantee it for you.", 0.0, 0.0, 0.50),
    ],
    "how do i track my order": [
        ("open your account, choose orders, then select track.", 1.0, 1.0, 0.25),
        ("open your account, choose orders, then select track to see live updates.", 1.0, 1.0, 0.85),
        ("just send me your login and i will track it for you.", 0.0, 0.0, 0.50),
    ],
}
PROMPTS = list(CANDIDATES)
SPEC_WEIGHTS = [2.4, 2.0, -0.3]        # what the product actually values: task, safety, length
ANNOTATOR_WEIGHTS = [2.4, 2.0, 1.5]    # the annotator we hire also overvalues length

corpus = "\n".join(f"{u}\n{a}" for u, a in DEMOS)
corpus += "\n" + "\n".join(text for p in PROMPTS for (text, *_) in CANDIDATES[p])
tokenizer = ch02.BytePairTokenizer.train(corpus * 3, vocab_size=400)
VOCAB_BASE = tokenizer.vocab_size
SPECIAL = {"system": VOCAB_BASE, "user": VOCAB_BASE + 1,
           "assistant": VOCAB_BASE + 2, "end": VOCAB_BASE + 3}
VOCAB = VOCAB_BASE + 4


def render_chat(messages: list[dict], add_generation_prompt: bool = False) -> list[int]:
    """Serialize chat messages into the token stream the model trains and serves on.

    Each message becomes its role token, its tokenized content, and an
    end-of-turn token; ``add_generation_prompt`` appends a bare assistant
    token to cue generation. This exact serialization is the model's ABI — the
    same function must run at training and inference time or the weights see a
    distribution they were never trained on.

    Args:
        messages: Dicts with ``role`` in {system, user, assistant} and ``content``.
        add_generation_prompt: Append a trailing assistant token to prompt a reply.

    Returns:
        Token IDs, including the reserved special-token IDs.
    """
    ids: list[int] = []
    for message in messages:
        ids.append(SPECIAL[message["role"]])
        ids += tokenizer.encode(message["content"])
        ids.append(SPECIAL["end"])
    if add_generation_prompt:
        ids.append(SPECIAL["assistant"])
    return ids


def build_example(user: str, assistant: str) -> tuple[list[int], list[int], list[int]]:
    """Serialize one demonstration into (input, target, response-mask) lists.

    The input is the rendered conversation minus its last token; the target is
    it shifted left by one, the standard next-token pairing. The mask marks a
    target position with 1 exactly when the predicted token lies in the
    assistant span (its content and the closing end token), so @eq-ch07-sft
    averages the loss over response tokens only.

    Args:
        user: The user turn.
        assistant: The demonstrated assistant response.

    Returns:
        ``(input_ids, target_ids, mask)``, all the same length.
    """
    prompt_ids = render_chat([{"role": "user", "content": user}], add_generation_prompt=True)
    full = prompt_ids + tokenizer.encode(assistant) + [SPECIAL["end"]]
    n_prompt = len(prompt_ids)
    return full[:-1], full[1:], [1 if (i + 1) >= n_prompt else 0 for i in range(len(full) - 1)]


def make_config():
    """Return the tiny chat model's config: Chapter 2's TinyGPT over our vocab."""
    return ch02.GPTConfig(vocab_size=VOCAB, block_size=96, d_model=64, n_heads=4, n_layers=2)


def sft_train(model, demos, mask_prompt: bool = True, steps: int = 130,
              lr: float = 3e-3, seed: int = 0) -> list[float]:
    """Supervised-fine-tune ``model`` on demonstrations by masked cross-entropy.

    Each step is one full-batch update of @eq-ch07-sft. With ``mask_prompt``
    the loss counts only assistant tokens; with it off, every token including
    the user's counts — the ablation @sec-ch07-masking measures.

    Args:
        model: A TinyGPT to train in place.
        demos: ``(user, assistant)`` pairs.
        mask_prompt: Apply the response-only mask (True) or train on all tokens.
        steps: Full-batch gradient steps.
        lr: AdamW learning rate.
        seed: Torch seed for reproducibility.

    Returns:
        Loss at the first, middle, and last step.
    """
    examples = [build_example(u, a) for u, a in demos]
    width = max(len(inp) for inp, _, _ in examples)
    X = torch.zeros(len(examples), width, dtype=torch.long)
    Y = torch.zeros(len(examples), width, dtype=torch.long)
    M = torch.zeros(len(examples), width)
    for i, (inp, tgt, msk) in enumerate(examples):
        span = len(inp)
        X[i, :span] = torch.tensor(inp)
        Y[i, :span] = torch.tensor(tgt)
        M[i, :span] = torch.tensor(msk if mask_prompt else [1] * span, dtype=torch.float)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    torch.manual_seed(seed)
    history = []
    for step in range(steps):
        model.train()
        logits, _, _ = model(X)
        token_logp = F.log_softmax(logits, dim=-1).gather(-1, Y.unsqueeze(-1)).squeeze(-1)
        loss = -(token_logp * M).sum() / M.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in {0, steps // 2, steps - 1}:
            history.append(loss.item())
    return history


@torch.inference_mode()
def chat(model, user: str, max_new: int = 40, stop: bool = True) -> tuple[str, int]:
    """Greedily decode the assistant's reply to ``user`` under the chat template.

    Args:
        model: The model to sample from.
        user: The user turn.
        max_new: Token budget for the reply.
        stop: Halt at the end-of-turn token (True) or emit ``max_new`` tokens.

    Returns:
        The decoded reply text and the number of generated tokens.
    """
    ids = render_chat([{"role": "user", "content": user}], add_generation_prompt=True)
    out = list(ids)
    model.eval()
    for _ in range(max_new):
        logits, _, _ = model(torch.tensor([out[-model.config.block_size:]]))
        nxt = int(logits[0, -1].argmax())
        if stop and nxt == SPECIAL["end"]:
            break
        out.append(nxt)
    generated = out[len(ids):]
    return tokenizer.decode([t for t in generated if t < VOCAB_BASE], errors="replace"), len(generated)


def span_loss(model, demos, which: str) -> float:
    """Mean cross-entropy on either the assistant or the user target positions.

    Splitting the loss by role is how we see what masking bought: the two
    models should tie on assistant tokens, and diverge sharply on user tokens,
    which only the unmasked model was trained to predict.

    Args:
        model: The model to score.
        demos: ``(user, assistant)`` pairs.
        which: ``"assistant"`` (mask==1 positions) or ``"user"`` (mask==0).

    Returns:
        Mean negative log-likelihood over the selected positions.
    """
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for user, assistant in demos:
            inp, tgt, mask = build_example(user, assistant)
            logp = F.log_softmax(model(torch.tensor([inp]))[0], dim=-1)[0]
            for i, m in enumerate(mask):
                if (m == 1) == (which == "assistant"):
                    total -= logp[i, tgt[i]].item()
                    count += 1
    return total / count


def block_diagonal_mask(segments: list[int]) -> list[list[bool]]:
    """Return the attention mask for packed examples: True where attention is blocked.

    A query may attend to a key only when they share a segment and the key is
    not in the future. This is the isolation naive causal packing lacks — under
    plain causal attention the cross-segment upper-left block would be visible,
    leaking one example into another's gradient.

    Args:
        segments: Segment id per packed position (which example it belongs to).

    Returns:
        A square boolean mask; ``mask[q][k]`` is True when key k is hidden from query q.
    """
    n = len(segments)
    return [[segments[q] != segments[k] or k > q for k in range(n)] for q in range(n)]


import random


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def features(prompt: str, index: int) -> list[float]:
    """Return the [task, safety, normalized-length] features of one candidate."""
    return list(CANDIDATES[prompt][index][1:])


def collect_preferences(count: int = 400, seed: int = 7):
    """Sample pairwise preferences from a biased synthetic annotator.

    For each pair the annotator's choice probability is a logistic function of
    the score gap plus a fixed left-position offset, so both an overvaluing of
    length (through ``ANNOTATOR_WEIGHTS``) and a position bias are baked into
    the labels the way they are in real collection. Each row records the
    content identities and whether the chosen candidate sat on the left, which
    is what lets us audit position separately from content.

    Args:
        count: Number of pairs to draw.
        seed: RNG seed.

    Returns:
        Rows of ``(prompt, chosen_index, rejected_index, chose_left)``.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        prompt = rng.choice(PROMPTS)
        left, right = rng.sample(range(len(CANDIDATES[prompt])), 2)
        scores = [dot(features(prompt, i), ANNOTATOR_WEIGHTS) for i in range(len(CANDIDATES[prompt]))]
        p_left = 1.0 / (1.0 + math.exp(-(scores[left] - scores[right] + 1.0)))
        chose_left = rng.random() < p_left
        chosen, rejected = (left, right) if chose_left else (right, left)
        rows.append((prompt, chosen, rejected, chose_left))
    return rows


def train_reward_model(rows, steps: int = 300, lr: float = 0.2):
    """Fit a linear Bradley-Terry reward model by gradient descent on @eq-ch07-bt.

    The reward is ``phi . features``; for each pair the gradient pushes the
    weights along the winner-minus-loser feature difference. Because the fit is
    faithful to the labels, a biased annotator produces a biased reward model —
    the learned length weight is the bias made legible.

    Args:
        rows: Preference rows from ``collect_preferences``.
        steps: Full-batch gradient steps.
        lr: Learning rate.

    Returns:
        ``(weights, loss_history)`` with loss at the first, middle, and last step.
    """
    weights = [0.0, 0.0, 0.0]
    history = []
    for step in range(steps):
        gradient = [0.0, 0.0, 0.0]
        loss = 0.0
        for prompt, chosen, rejected, _ in rows:
            delta = [features(prompt, chosen)[i] - features(prompt, rejected)[i] for i in range(3)]
            prob = 1.0 / (1.0 + math.exp(-dot(weights, delta)))
            loss -= math.log(max(prob, 1e-12))
            for i in range(3):
                gradient[i] += (prob - 1.0) * delta[i]
        for i in range(3):
            weights[i] -= lr * gradient[i] / len(rows)
        if step in {0, steps // 2, steps - 1}:
            history.append(loss / len(rows))
    return weights, history


def seq_logp(model, prompt: str, response: str):
    """Summed log-probability of a response's tokens under the chat template.

    This is the quantity inside @eq-ch07-dpo: render the prompt with a
    generation cue, append the response and an end token, and sum the model's
    log-probabilities over exactly the response positions. The reference and
    the policy are scored the same way; their difference is the implicit reward.

    Args:
        model: The model to score under.
        prompt: The user turn.
        response: The candidate assistant response.

    Returns:
        A scalar tensor: the summed completion-token log-probability.
    """
    prompt_ids = render_chat([{"role": "user", "content": prompt}], add_generation_prompt=True)
    full = prompt_ids + tokenizer.encode(response) + [SPECIAL["end"]]
    logp = F.log_softmax(model(torch.tensor([full[:-1]]))[0], dim=-1)[0]
    targets = full[1:]
    start = len(prompt_ids) - 1
    return logp[range(start, len(targets)), targets[start:]].sum()


def annotator_best(prompt: str) -> int:
    """Index of the candidate the biased annotator scores highest for a prompt."""
    scores = [dot(features(prompt, i), ANNOTATOR_WEIGHTS) for i in range(len(CANDIDATES[prompt]))]
    return max(range(len(scores)), key=scores.__getitem__)


def dpo_pairs():
    """Build (prompt, chosen, rejected) text pairs from the annotator's preference.

    Chosen is the annotator's top pick — here the longer compliant answer,
    because its length weight tips the balance; rejected is the equally correct
    concise answer. The only systematic difference is length, which isolates
    the bias for @eq-ch07-dpo to absorb.

    Returns:
        A list of ``(prompt, chosen_text, rejected_text)``.
    """
    return [(p, CANDIDATES[p][annotator_best(p)][0], CANDIDATES[p][0][0]) for p in PROMPTS]


def train_dpo(policy, reference, pairs, steps: int = 60, lr: float = 3e-4, beta: float = 0.1):
    """Optimize the DPO loss of @eq-ch07-dpo against a frozen reference.

    Reference log-probabilities are computed once; each step raises the
    implicit-reward margin of chosen over rejected. A gentle step size shifts
    the preference while keeping generation coherent — pushed harder, the same
    loss over-optimizes, which @sec-ch07-dpo demonstrates.

    Args:
        policy: The trainable policy (a copy of the SFT model).
        reference: The frozen SFT reference.
        pairs: ``(prompt, chosen, rejected)`` triples.
        steps: Gradient steps.
        lr: AdamW learning rate.
        beta: The reference-leash strength from @eq-ch07-kl.

    Returns:
        Loss at the first, middle, and last step.
    """
    with torch.no_grad():
        ref = [(seq_logp(reference, p, w).item(), seq_logp(reference, p, l).item()) for p, w, l in pairs]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr)
    history = []
    for step in range(steps):
        policy.train()
        loss = 0.0
        for (p, w, l), (ref_w, ref_l) in zip(pairs, ref):
            margin = beta * ((seq_logp(policy, p, w) - ref_w) - (seq_logp(policy, p, l) - ref_l))
            loss = loss - F.logsigmoid(margin)
        loss = loss / len(pairs)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if step in {0, steps // 2, steps - 1}:
            history.append(loss.item())
    return history
