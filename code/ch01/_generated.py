# Auto-generated from chapters/01-operational-on-ramp.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


from dataclasses import dataclass


@dataclass(frozen=True)
class TokenScore:
    """One returned token paired with its log probability under the model.

    The log probability is what the API exposes about the model's own certainty
    for that token. It is a statement about likely text, not about truth, and
    keeping it beside the token is what lets us exponentiate it later.
    """

    token: str
    logprob: float


@dataclass(frozen=True)
class Completion:
    """A minimal, provider-neutral result for one model call.

    Every field is a thing the caller can measure without trusting the prose:
    the text, the per-token scores, the token counts (never guessed from
    characters), and the two latencies that a streaming client can observe.
    """

    text: str
    token_scores: tuple[TokenScore, ...]
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float
    total_ms: float


from collections import Counter


def train_bpe(corpus: list[str], num_merges: int) -> list[tuple[str, str]]:
    """Learn byte-pair-encoding merge rules from a small corpus.

    Every word starts split into characters plus an end-of-word marker. At each
    step we count adjacent symbol pairs across the whole corpus and merge the
    most frequent one; frequent sequences become single tokens, which is why a
    common word ends up cheaper (fewer tokens) than a rare name.

    Args:
        corpus: Whitespace-tokenizable training sentences.
        num_merges: How many merge rules to learn; more merges means a larger
            vocabulary and shorter encodings.

    Returns:
        The learned merges in application order, each a pair of symbols to join.
    """
    words = Counter(word for line in corpus for word in line.split())
    splits = {word: list(word) + ["</w>"] for word in words}
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pairs: Counter[tuple[str, str]] = Counter()
        for word, freq in words.items():
            symbols = splits[word]
            for pair in zip(symbols, symbols[1:]):
                pairs[pair] += freq
        if not pairs:
            break
        best = max(pairs, key=lambda pair: (pairs[pair], pair))
        merges.append(best)
        for word, symbols in splits.items():
            merged, i = [], 0
            while i < len(symbols):
                if (symbols[i], symbols[i + 1] if i + 1 < len(symbols) else None) == best:
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            splits[word] = merged
    return merges


def bpe_encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Encode one word into subword pieces by replaying learned merges.

    Args:
        word: A single whitespace-free token.
        merges: Merge rules from :func:`train_bpe`, applied in order.

    Returns:
        The word's pieces, ending in the ``</w>`` marker; a frequent word may
        collapse to a single piece while a rare word stays fragmented.
    """
    symbols = list(word) + ["</w>"]
    for left, right in merges:
        merged, i = [], 0
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == left and symbols[i + 1] == right:
                merged.append(left + right)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        symbols = merged
    return symbols


import math


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Turn next-token logits into a probability distribution over candidates.

    Temperature rescales the logits before exponentiation: below 1 sharpens the
    distribution toward the top token, above 1 flattens it toward uniform. The
    subtraction of the maximum is numerical hygiene and does not change the
    result.

    Args:
        logits: Unnormalized scores, one per candidate token.
        temperature: Positive scale; smaller is greedier, larger is flatter.

    Returns:
        Probabilities in the same order as ``logits``, summing to 1.
    """
    scaled = [value / temperature for value in logits]
    ceiling = max(scaled)
    weights = [math.exp(value - ceiling) for value in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


import random


def sample_token(probs: list[float], rng: random.Random) -> int:
    """Draw one token index from a categorical distribution.

    Walks the cumulative probability and stops where a uniform draw lands
    (inverse-CDF sampling). Passing a seeded ``random.Random`` makes the draw
    reproducible, which is how an offline probe pins nondeterminism it cannot
    control on a real server.

    Args:
        probs: A probability vector that sums to 1.
        rng: A seeded random generator supplying the uniform draw.

    Returns:
        The index of the sampled token.
    """
    threshold = rng.random()
    cumulative = 0.0
    for index, prob in enumerate(probs):
        cumulative += prob
        if threshold <= cumulative:
            return index
    return len(probs) - 1


def token_confidence(logprob: float) -> float:
    """Return exp(logprob): the model's probability for that single token."""
    return math.exp(logprob)


def decide_response(confidence: float, evidence_score: float, threshold: float = 0.7) -> str:
    """Turn an uncertainty estimate into a concrete system action.

    Abstention keys on evidence, not on raw token confidence. A high confidence
    with no supporting evidence still routes to ``abstain``, because token
    likelihood is not a check that a claim is real; genuine support with an
    adequate confidence answers; anything else asks for more evidence.

    Args:
        confidence: A calibrated confidence in [0, 1] for the candidate answer.
        evidence_score: Support from retrieval or a verifier in [0, 1].
        threshold: Minimum confidence required to answer directly.

    Returns:
        One of ``"answer"``, ``"retrieve"``, or ``"abstain"``.
    """
    if evidence_score >= 0.5 and confidence >= threshold:
        return "answer"
    if confidence >= threshold and evidence_score < 0.5:
        return "abstain"
    return "retrieve"


def simulate_decode(prompt_tokens: int, output_tokens: int, *,
                    prefill_ms_per_token: float = 0.4, decode_ms_per_token: float = 18.0,
                    jitter_ms: float = 1.5, seed: int = 0) -> list[float]:
    """Simulate the arrival times of one streaming generation.

    Prefill processes the whole prompt in parallel to produce the first token
    (its cost scales with prompt length); decode then emits tokens serially with
    small per-step jitter. Returning arrival times lets the same TTFT/TPOT
    formulas used on a real stream run offline and deterministically.

    Args:
        prompt_tokens: Number of input tokens, driving prefill time.
        output_tokens: Number of tokens to emit.
        prefill_ms_per_token: Prefill cost charged per prompt token.
        decode_ms_per_token: Mean serial cost per output token.
        jitter_ms: Symmetric per-token noise around the decode mean.
        seed: Seed for the jitter, so runs reproduce.

    Returns:
        Arrival time in milliseconds of each output token, first to last.
    """
    rng = random.Random(seed)
    ttft = 8.0 + prompt_tokens * prefill_ms_per_token
    times = [ttft]
    for _ in range(1, output_tokens):
        times.append(times[-1] + decode_ms_per_token + rng.uniform(-jitter_ms, jitter_ms))
    return times


def tpot_ms(token_times_ms: list[float]) -> float:
    """Compute time per output token from a decode loop's arrival times.

    Args:
        token_times_ms: Arrival time in milliseconds of each output token.

    Returns:
        The request-level TPOT, (t_N - t_1) / (N - 1); 0.0 for a single token.
    """
    if len(token_times_ms) < 2:
        return 0.0
    return (token_times_ms[-1] - token_times_ms[0]) / (len(token_times_ms) - 1)


@dataclass(frozen=True)
class Rung:
    """One rung of the authority ladder and the per-task overhead it adds.

    The fields are deliberately countable — calls, retries, review minutes, and
    the number of records a failure can touch — because a budget stated in
    countable units is the only kind you can enforce or promote against.
    """

    name: str
    extra_calls: float
    expected_retries: float
    review_minutes: float
    blast_radius_records: int


def autonomy_costs(rungs: list[Rung], call_cost: float = 0.02,
                   reviewer_cost_per_minute: float = 1.0) -> list[tuple[str, float, float]]:
    """Cost each authority increment in the dollars used to justify the project.

    Every rung is charged for the model calls and retries it adds and for the
    operator minutes it consumes; the running total shows that autonomy is
    bought one action at a time, each increment with an explicit price.

    Args:
        rungs: The ladder rungs in ascending order of authority.
        call_cost: Dollar cost of one model call (and of one retry).
        reviewer_cost_per_minute: Dollar cost of one operator-review minute.

    Returns:
        One ``(name, incremental_cost, cumulative_cost)`` triple per rung.
    """
    rows, cumulative = [], 0.0
    for rung in rungs:
        incremental = (rung.extra_calls + rung.expected_retries) * call_cost \
            + rung.review_minutes * reviewer_cost_per_minute
        cumulative += incremental
        rows.append((rung.name, incremental, cumulative))
    return rows
