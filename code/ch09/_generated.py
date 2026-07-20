# Auto-generated from chapters/09-inference-behavior.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import math

def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Turn logits into a temperature-scaled categorical distribution.

    Implements @eq-ch09-temperature with two numerical guards: at
    ``temperature <= 0`` it returns a one-hot argmax (the greedy limit, with
    ties broken by lowest index) rather than dividing by zero, and it subtracts
    the largest scaled logit before exponentiating so ``exp`` never overflows.
    Neither guard changes the resulting probabilities.

    Args:
        logits: One real score per vocabulary token.
        temperature: Positive scale; smaller sharpens, larger flattens. Values
            at or below zero select the argmax deterministically.

    Returns:
        A probability distribution the same length as ``logits`` summing to 1.
    """
    if temperature <= 0:
        winner = max(range(len(logits)), key=logits.__getitem__)
        return [float(i == winner) for i in range(len(logits))]
    scaled = [z / temperature for z in logits]
    peak = max(scaled)
    exps = [math.exp(s - peak) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


import math

def truncate(probs: list[float], method: str = "full", value: float = 1.0) -> list[float]:
    """Mask a distribution to a candidate set and renormalize the survivors.

    Supports four policies that trade fixed against adaptive support: ``top_k``
    (the ``value`` highest tokens), ``top_p`` (the smallest prefix reaching
    cumulative mass ``value``, including the crossing token), ``min_p`` (tokens
    at least ``value`` times the peak probability), and ``typical`` (tokens
    whose surprisal is closest to the entropy until mass ``value``). Every
    policy keeps at least one token, so the result is always a valid draw.

    Args:
        probs: A probability distribution (already temperature-scaled).
        method: One of ``full``, ``top_k``, ``top_p``, ``min_p``, ``typical``.
        value: The policy parameter (``k``, ``p``, ``alpha``, or mass budget).

    Returns:
        A distribution over the same support with dropped tokens at 0.0 and the
        kept tokens renormalized to sum to 1.

    Raises:
        ValueError: If ``method`` is not a known policy.
    """
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    if method == "full":
        keep = set(order)
    elif method == "top_k":
        keep = set(order[: max(1, int(value))])
    elif method == "top_p":
        keep, mass = set(), 0.0
        for i in order:
            keep.add(i); mass += probs[i]
            if mass >= value:
                break
    elif method == "min_p":
        cutoff = value * probs[order[0]]
        keep = {i for i in order if probs[i] >= cutoff} or {order[0]}
    elif method == "typical":
        entropy = -sum(p * math.log(p) for p in probs if p)
        ranked = sorted(order, key=lambda i: abs(-math.log(probs[i]) - entropy) if probs[i] else 1e9)
        keep, mass = set(), 0.0
        for i in ranked:
            keep.add(i); mass += probs[i]
            if mass >= value:
                break
    else:
        raise ValueError(f"unknown truncation method: {method}")
    total = sum(probs[i] for i in keep)
    return [probs[i] / total if i in keep else 0.0 for i in range(len(probs))]


def process_logits(logits: list[float], counts: list[int], repetition: float = 1.0,
                   frequency: float = 0.0, presence: float = 0.0,
                   bias: dict[int, float] | None = None) -> list[float]:
    """Rewrite logits from token history using a sign-aware repetition rule.

    Applies @eq-ch09-processor plus a multiplicative repetition penalty that is
    sign-aware: a positive seen logit is divided by ``repetition`` and a
    negative one is multiplied by it, so a repeated token's probability always
    falls. Frequency and presence penalties and an explicit per-token bias are
    additive, so their strength scales as ``1/T`` if a later temperature divide
    follows — which is why processor order is part of the policy.

    Args:
        logits: The model's raw logits.
        counts: How many times each token already appears in the context.
        repetition: Sign-aware multiplicative penalty (1.0 disables it).
        frequency: Additive penalty per prior occurrence.
        presence: Additive penalty applied once if the token appeared at all.
        bias: Optional explicit per-index logit offsets.

    Returns:
        The rewritten logit list, same length as ``logits``.
    """
    out = logits[:]
    for i, count in enumerate(counts):
        if count and repetition != 1.0:
            out[i] = out[i] / repetition if out[i] > 0 else out[i] * repetition
        out[i] -= frequency * count + presence * float(count > 0)
        out[i] += (bias or {}).get(i, 0.0)
    return out


def batch_invariance_probe() -> dict[str, float]:
    """Show that two legal reduction orders can flip a greedy argmax.

    A logit is assembled from terms that catastrophically cancel. Summed left to
    right the small term is lost and the logit is 0.0; summed as two partial
    sums the large terms cancel first and the logit is 1.0. Against a rival at
    0.5 the two orders pick different winners, so a purely numerical reduction
    change — not randomness — moves the argmax and the top probability.

    Returns:
        A dict with both reduction sums, whether the greedy winner flipped, and
        the maximum probability drift between the two orders.
    """
    parts = [1e16, 1.0, -1e16, 0.0]
    serial = 0.0
    for term in parts:
        serial += term                      # left-to-right loses the 1.0
    partitioned = (parts[0] + parts[2]) + (parts[1] + parts[3])   # cancels large terms first
    single = softmax([serial, 0.5])
    batched = softmax([partitioned, 0.5])
    return {"serial_sum": serial, "partitioned_sum": partitioned,
            "greedy_flip": float(single.index(max(single)) != batched.index(max(batched))),
            "max_drift": max(abs(a - b) for a, b in zip(single, batched))}


def support_fraction(claims: list[tuple[str, bool]]) -> float:
    """Fraction of atomic claims that are supported by the reference.

    Long-form factuality is scored per atomic claim, not per answer, so a fluent
    response with one false sentence is not credited as correct. ``claims`` pairs
    each atomic statement with whether the reference (world state or source set)
    supports it.

    Args:
        claims: ``(claim_text, is_supported)`` pairs from one answer.

    Returns:
        Supported claims divided by total claims, in [0, 1]; 0.0 if empty.
    """
    if not claims:
        return 0.0
    return sum(1 for _, ok in claims if ok) / len(claims)


import random

def qa_fixture(n: int = 400, seed: int = 0) -> list[dict]:
    """Build an illustrative QA fixture with two competing confidence signals.

    Each item has a ground-truth ``correct`` label plus two confidence scores on
    the same question: ``logprob_conf``, a token-probability-style signal that is
    monotone in latent skill but systematically overconfident, and
    ``verbal_conf``, a coarse rounded "how sure are you" number. The two are
    deliberately miscalibrated in different ways so calibration and abstention
    have something to fix. Values are synthetic, not measured on any model.

    Args:
        n: Number of items to generate.
        seed: Seed for the internal RNG, so the fixture is reproducible.

    Returns:
        A list of dicts with ``correct``, ``logprob_conf``, and ``verbal_conf``.
    """
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        skill = rng.random()
        correct = int(rng.random() < skill)
        logprob_conf = min(0.99, max(0.5, 0.5 + 0.5 * skill ** 0.5))
        verbal_conf = min(0.95, max(0.5, round((0.4 + 0.6 * skill) * 10) / 10))
        items.append({"correct": correct, "logprob_conf": logprob_conf, "verbal_conf": verbal_conf})
    return items


def brier(items: list[dict], key: str) -> float:
    """Mean squared error between a confidence score and the binary outcome."""
    return sum((it[key] - it["correct"]) ** 2 for it in items) / len(items)

def ece(items: list[dict], key: str, bins: int = 10) -> float:
    """Expected calibration error: accuracy-weighted gap between confidence and accuracy.

    Predictions are grouped into ``bins`` equal-width confidence buckets; each
    bucket contributes the absolute difference between its mean confidence and
    its empirical accuracy, weighted by its share of the data.

    Args:
        items: Scored items with a ``correct`` field.
        key: Which confidence field to grade.
        bins: Number of equal-width confidence buckets.

    Returns:
        The weighted calibration gap, in [0, 1]; lower is better.
    """
    buckets = [[] for _ in range(bins)]
    for it in items:
        buckets[min(bins - 1, int(it[key] * bins))].append(it)
    error = 0.0
    for bucket in buckets:
        if bucket:
            confidence = sum(it[key] for it in bucket) / len(bucket)
            accuracy = sum(it["correct"] for it in bucket) / len(bucket)
            error += len(bucket) / len(items) * abs(confidence - accuracy)
    return error


import math, re

def normalize_answer(answer: str) -> tuple:
    """Canonicalize an answer to a content-word key for meaning-equivalence.

    Lowercases, keeps alphanumeric words, drops a small stopword set, and sorts
    the remainder, so "Paris", "The city is Paris", and "paris." map to the same
    key. This is a deterministic stand-in for the entailment-based clustering a
    production semantic-entropy estimator would use.

    Args:
        answer: One sampled answer string.

    Returns:
        A sorted tuple of content words used as the cluster identity.
    """
    words = re.findall(r"[a-z0-9]+", answer.lower())
    stop = {"the", "a", "an", "is", "was", "it", "of", "in", "city", "capital", "named"}
    return tuple(sorted(w for w in words if w not in stop))

def semantic_entropy(samples: list[tuple[str, float]]) -> dict:
    """Compare surface-form entropy with entropy over meaning clusters.

    Groups ``(answer, probability)`` samples by ``normalize_answer`` and computes
    entropy before and after grouping (@eq-ch09-semantic). Paraphrases of one
    answer collapse into a single cluster, so semantic entropy is at most surface
    entropy and is lower whenever surface variation is meaning-preserving.

    Args:
        samples: Sampled answers with their probabilities.

    Returns:
        A dict with surface and semantic entropy (nats) and the cluster masses.
    """
    surface, clusters = {}, {}
    for answer, prob in samples:
        surface[answer] = surface.get(answer, 0.0) + prob
        key = normalize_answer(answer)
        clusters[key] = clusters.get(key, 0.0) + prob
    entropy = lambda ps: -sum(p * math.log(p) for p in ps if p > 0)
    return {"surface_nats": entropy(surface.values()), "semantic_nats": entropy(clusters.values()),
            "clusters": {(" ".join(k) or "<empty>"): round(v, 3) for k, v in clusters.items()}}


import math

def fit_temperature(items: list[dict], key: str) -> float:
    """Fit one scalar calibration temperature by minimizing held-out log loss.

    Rescales each confidence in logit space by ``1/tau`` and grid-searches ``tau``
    over [0.30, 5.0] for the value minimizing binary NLL. Because the map is
    monotone it changes only how confidence values map to probabilities, never
    the order of examples — so it improves calibration without changing which
    items a threshold answers first.

    Args:
        items: Scored items with a ``correct`` field.
        key: Which confidence field to calibrate.

    Returns:
        The temperature minimizing held-out log loss.
    """
    def nll(tau: float) -> float:
        total = 0.0
        for it in items:
            q = it[key]
            calibrated = 1 / (1 + math.exp(-math.log(q / (1 - q)) / tau))
            p = calibrated if it["correct"] else 1 - calibrated
            total -= math.log(max(p, 1e-12))
        return total / len(items)
    return min((0.30 + i * 0.01 for i in range(471)), key=nll)

def calibrate(q: float, tau: float) -> float:
    """Apply a fitted temperature to one confidence in logit space."""
    return 1 / (1 + math.exp(-math.log(q / (1 - q)) / tau))


def risk_curve(items: list[dict], key: str) -> list[dict]:
    """Trace coverage, selective risk, and marginal error as the threshold sweeps.

    For each distinct confidence value used as a threshold, reports the fraction
    answered (coverage), the error rate among answered items (selective risk,
    @eq-ch09-risk), and the wrong-answer rate over all items (marginal error).
    A useful confidence ranking lowers selective risk as coverage falls.

    Args:
        items: Scored items with a ``correct`` field.
        key: Which confidence field to threshold on.

    Returns:
        One row per threshold, from most selective to least.
    """
    rows = []
    for t in sorted({it[key] for it in items}, reverse=True):
        answered = [it for it in items if it[key] >= t]
        errors = sum(1 - it["correct"] for it in answered)
        rows.append({"threshold": t, "coverage": len(answered) / len(items),
                     "selective_risk": errors / len(answered), "marginal_error": errors / len(items)})
    return rows


import math

def crc_threshold(items: list[dict], key: str, alpha: float) -> float:
    """Lowest threshold whose released wrong-answer rate satisfies the CRC bound.

    Scans thresholds upward and returns the first where the conformal correction
    ``(errors + 1) / (n + 1) <= alpha`` holds, controlling expected marginal loss
    up to the finite-sample correction under exchangeability. The lowest such
    threshold maximizes coverage.

    Args:
        items: Calibration items with a ``correct`` field.
        key: Which confidence field to threshold on.
        alpha: Target marginal wrong-answer rate.

    Returns:
        The selected threshold, or ``inf`` if none qualifies.
    """
    for t in sorted({it[key] for it in items}):
        errors = sum(it[key] >= t and not it["correct"] for it in items)
        if (errors + 1) / (len(items) + 1) <= alpha:
            return t
    return math.inf


import math, random

def multiclass_fixture(n_cal: int = 200, n_test: int = 200) -> tuple:
    """Build illustrative calibration and test sets of per-class probabilities.

    Each item has a probability over four answer options and the index of the
    true one; the true-class probability varies with a latent skill so answer
    sets range from confident singletons to genuine multi-answer ambiguity.
    Synthetic, for exercising the conformal machinery, not measured from a model.

    Returns:
        A ``(calibration, test)`` pair of item lists, each item a dict with
        ``probs`` and ``true_index``.
    """
    rng = random.Random(7)
    def make(n):
        out = []
        for _ in range(n):
            skill = rng.random()
            true_p = 0.30 + 0.55 * skill
            others = sorted((rng.random() for _ in range(3)), reverse=True)
            scale = (1 - true_p) / sum(others)
            probs = [true_p] + [o * scale for o in others]
            order = list(range(4)); rng.shuffle(order)
            shuffled = [0.0] * 4
            for src, dst in enumerate(order):
                shuffled[dst] = probs[src]
            out.append({"probs": shuffled, "true_index": order[0]})
        return out
    return make(n_cal), make(n_test)

def split_conformal(calibration: list[dict], test: list[dict], alpha: float) -> dict:
    """Split-conformal answer sets with guaranteed marginal true-label coverage.

    Scores nonconformity ``1 - p(true)`` on the calibration set, takes the
    rank-``ceil((n+1)(1-alpha))`` value as the set threshold, and returns every
    answer within it. Under exchangeability the true label is in the set with
    probability at least ``1 - alpha``. A singleton set is a confident answer; a
    larger set is an abstention signal.

    Args:
        calibration: Items with ``probs`` and ``true_index``.
        test: Held-out items scored the same way.
        alpha: Target miscoverage; coverage is guaranteed at ``1 - alpha``.

    Returns:
        A dict with the threshold ``qhat`` and coverage / singleton statistics.
    """
    scores = sorted(1 - it["probs"][it["true_index"]] for it in calibration)
    rank = min(len(scores), math.ceil((len(scores) + 1) * (1 - alpha))) - 1
    qhat = scores[rank]
    contained = singleton = errors = 0
    for it in test:
        answer_set = [i for i, p in enumerate(it["probs"]) if 1 - p <= qhat + 1e-12]
        contained += int(it["true_index"] in answer_set)
        if len(answer_set) == 1:
            singleton += 1
            errors += int(answer_set[0] != it["true_index"])
    return {"qhat": qhat, "set_coverage": contained / len(test),
            "singleton_coverage": singleton / len(test),
            "singleton_risk": errors / singleton if singleton else 0.0}


import codecs

def stream_incremental(pieces: list[bytes], stop: str = "") -> str:
    """Decode a byte-piece stream into safe text, buffering across boundaries.

    Runs a stateful UTF-8 decoder so a multibyte character split across pieces is
    emitted only once complete, and withholds the last ``len(stop) - 1``
    characters so a stop sequence straddling two pieces is caught before any of
    it reaches the user. Returns text up to the stop if one is found.

    Args:
        pieces: Successive byte fragments (e.g. token ``token_bytes``).
        stop: Optional stop string to detect across piece boundaries.

    Returns:
        The safely decoded text, truncated at the stop if present.
    """
    decoder = codecs.getincrementaldecoder("utf-8")()
    pending, emitted = "", []
    for piece in pieces:
        pending += decoder.decode(piece)
        if stop and stop in pending:
            emitted.append(pending.split(stop, 1)[0])
            return "".join(emitted)
        hold = max(0, len(stop) - 1)
        if len(pending) > hold:
            emitted.append(pending[:-hold] if hold else pending)
            pending = pending[-hold:] if hold else ""
    return "".join(emitted) + pending + decoder.decode(b"", final=True)
