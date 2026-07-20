# Auto-generated from chapters/05-scaling-laws-pretraining-data.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import numpy as np
from dataclasses import dataclass

PARAM_REF, TOKEN_REF = 1.0e4, 1.0e5   # reference units; conditioning only


@dataclass
class ScalingLaw:
    """A fitted joint law L(N, D) = E + A(N/N0)^-alpha + B(D/D0)^-beta.

    The five fitted numbers are the whole model of the ladder: ``floor`` is the
    irreducible loss E, ``param_coeff``/``data_coeff`` scale each resource term,
    and ``param_exponent``/``data_exponent`` are how fast loss falls with model
    size and with tokens. Their ratio, not their absolute size, decides how a
    compute budget should be split between N and D.
    """

    floor: float
    param_coeff: float
    data_coeff: float
    param_exponent: float
    data_exponent: float

    def loss(self, params, tokens):
        """Predict loss for absolute parameter and token counts.

        Args:
            params: Non-embedding parameter count(s) N, scalar or array.
            tokens: Training-token count(s) D, scalar or array.

        Returns:
            The predicted cross-entropy under @eq-ch05-joint.
        """
        n = np.asarray(params, dtype=float) / PARAM_REF
        d = np.asarray(tokens, dtype=float) / TOKEN_REF
        return (self.floor + self.param_coeff * n ** (-self.param_exponent)
                + self.data_coeff * d ** (-self.data_exponent))

    def compute_optimal(self, compute):
        """Return the loss-minimizing (N, D, loss) on the budget C = 6ND.

        Args:
            compute: A training-compute budget C in FLOPs.

        Returns:
            A tuple ``(N, D, loss)``: the parameter and token counts that
            minimize @eq-ch05-joint subject to ``6 * N * D == compute``, and the
            loss they achieve. The split follows the closed form of
            @sec-ch05-chinchilla.
        """
        c = compute / (6.0 * PARAM_REF * TOKEN_REF)
        ratio = (self.param_exponent * self.param_coeff) / (self.data_exponent * self.data_coeff)
        n = (ratio * c ** self.data_exponent) ** (1.0 / (self.param_exponent + self.data_exponent))
        d = c / n
        return n * PARAM_REF, d * TOKEN_REF, float(self.loss(n * PARAM_REF, d * TOKEN_REF))


from scipy.optimize import curve_fit


def fit_scaling_law(params, tokens, losses):
    """Fit @eq-ch05-joint to measured ladder points by bounded least squares.

    The fit is done in loss units with positivity bounds on every coefficient
    and exponent, and the floor E is bounded below the smallest observed loss so
    the reducible terms explain the *variation* rather than absorbing the floor.

    Args:
        params: Sequence of non-embedding parameter counts N.
        tokens: Sequence of training-token counts D.
        losses: Sequence of measured losses, one per (N, D).

    Returns:
        The fitted :class:`ScalingLaw`.
    """
    n = np.asarray(params, dtype=float)
    d = np.asarray(tokens, dtype=float)
    y = np.asarray(losses, dtype=float)

    def surface(inputs, floor, a, b, alpha, beta):
        nn, dd = inputs
        return floor + a * (nn / PARAM_REF) ** (-alpha) + b * (dd / TOKEN_REF) ** (-beta)

    lower = [0.0, 1e-4, 1e-4, 0.02, 0.02]
    upper = [float(y.min()), 80.0, 80.0, 2.0, 2.0]
    guess = [float(y.min()) * 0.5, 1.0, 1.0, 0.3, 0.3]
    popt, _ = curve_fit(surface, (n, d), y, p0=guess, bounds=(lower, upper), maxfev=40000)
    return ScalingLaw(*popt)


def extrapolation_interval(params, tokens, losses, *, multiplier=100.0, samples=120, seed=0):
    """Residual-bootstrap the compute-optimal loss at ``multiplier`` times the
    ladder's largest compute.

    The point estimate comes from the fit on the real ladder; the interval comes
    from refitting on residual-resampled copies and taking the 5th/95th
    percentiles of the extrapolated loss. It quantifies finite-ladder noise
    only — not model misspecification or distribution shift.

    Args:
        params: Sequence of parameter counts N.
        tokens: Sequence of token counts D.
        losses: Sequence of measured losses.
        multiplier: How far past the largest observed compute to extrapolate.
        samples: Number of bootstrap refits.
        seed: Seed for the residual resampler.

    Returns:
        A dict with the target compute, the compute-optimal ``(opt_params,
        opt_tokens)``, the point ``loss``, its ``p05``/``p95`` bounds, and the
        count of ``valid`` bootstrap fits.
    """
    params = np.asarray(params, dtype=float)
    tokens = np.asarray(tokens, dtype=float)
    losses = np.asarray(losses, dtype=float)
    law = fit_scaling_law(params, tokens, losses)
    target = float((6.0 * params * tokens).max() * multiplier)
    opt_params, opt_tokens, point = law.compute_optimal(target)
    base = np.asarray(law.loss(params, tokens))
    residuals = losses - base
    residuals = residuals - residuals.mean()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        noisy = base + rng.choice(residuals, size=len(losses))
        try:
            _, _, predicted = fit_scaling_law(params, tokens, noisy).compute_optimal(target)
        except Exception:
            continue
        if np.isfinite(predicted):
            draws.append(predicted + float(rng.choice(residuals)))
    low, high = np.quantile(draws, (0.05, 0.95))
    return {"target_compute": target, "opt_params": opt_params, "opt_tokens": opt_tokens,
            "loss": point, "p05": float(low), "p95": float(high), "valid": len(draws)}


import hashlib, re
from dataclasses import dataclass as _dataclass

WORD = re.compile(r"\w+", re.UNICODE)


@_dataclass(frozen=True)
class Document:
    """One extracted record with the provenance a policy decision needs.

    ``source`` and ``rights`` are kept distinct on purpose: a crawler being able
    to fetch a page (source) is not the same as a license to train on it
    (rights), and collapsing them into a single ``safe`` flag is how corpora
    quietly admit text they had no right to.
    """

    doc_id: str
    url: str
    source: str
    language: str
    rights: str
    text: str


def extract_records(raw_records):
    """Turn raw crawl records into :class:`Document` values with stable IDs.

    Args:
        raw_records: Iterable of dicts with ``url``, ``source``, ``rights``,
            ``lang``, and ``text`` keys (the fields a WET record and its headers
            supply).

    Returns:
        A list of :class:`Document`, one per record, each ``doc_id`` a short hash
        of the URL so the same page always extracts to the same identifier.
    """
    documents = []
    for record in raw_records:
        doc_id = hashlib.sha1(record["url"].encode()).hexdigest()[:10]
        documents.append(Document(doc_id, record["url"], record["source"],
                                   record["lang"], record["rights"], record["text"].strip()))
    return documents


from collections import Counter, defaultdict


@_dataclass(frozen=True)
class FilterPolicy:
    """Auditable thresholds for a transparent quality filter.

    Each field is a single interpretable knob so a rejection can always be
    explained by one named reason rather than an opaque score.
    """

    allowed_rights: tuple = ("licensed", "public-domain", "permission")
    min_words: int = 40
    min_alpha_ratio: float = 0.55
    max_repeated_line_fraction: float = 0.5


def document_features(text):
    """Measure interpretable surface features without treating them as truth.

    Args:
        text: The document body.

    Returns:
        A tuple ``(word_count, alpha_ratio, repeated_line_fraction)`` — the three
        quantities the filter thresholds.
    """
    words = WORD.findall(text)
    visible = [c for c in text if not c.isspace()]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    repeated = 0.0 if not lines else 1.0 - len(set(lines)) / len(lines)
    alpha = sum(c.isalpha() for c in visible) / max(1, len(visible))
    return len(words), alpha, repeated


def filter_documents(documents, policy=FilterPolicy()):
    """Apply rights and quality gates, recording a reason for each rejection.

    Args:
        documents: Iterable of :class:`Document`.
        policy: The :class:`FilterPolicy` thresholds to apply.

    Returns:
        A tuple ``(kept, rejected)`` where ``rejected`` items are ``{doc_id,
        reason}`` dicts, so the drop at this gate is always explainable.
    """
    kept, rejected = [], []
    for doc in documents:
        words, alpha, repeated = document_features(doc.text)
        reason = None
        if doc.rights not in policy.allowed_rights:
            reason = "rights"
        elif words < policy.min_words:
            reason = "too-short"
        elif alpha < policy.min_alpha_ratio:
            reason = "low-alpha"
        elif repeated > policy.max_repeated_line_fraction:
            reason = "repetition"
        if reason is None:
            kept.append(doc)
        else:
            rejected.append({"doc_id": doc.doc_id, "reason": reason})
    return kept, rejected


def word_shingles(text, width=5, cap=200):
    """Return a bounded set of hashed word shingles for a document.

    Args:
        text: The document body.
        width: Shingle length in words (5 is a common choice).
        cap: Keep only the ``cap`` lowest-hashing shingles (bottom-k sampling),
            so long and short documents get comparable-size signatures.

    Returns:
        A set of shingle strings.
    """
    words = WORD.findall(text.casefold())
    shingles = {" ".join(words[i:i + width]) for i in range(max(1, len(words) - width + 1))}
    return set(sorted(shingles, key=lambda s: hashlib.blake2b(s.encode(), digest_size=8).digest())[:cap])


def jaccard(a, b):
    """Return the Jaccard similarity of two shingle sets."""
    return len(a & b) / max(1, len(a | b))


def minhash_signature(shingles, permutations=32, prime=(1 << 61) - 1):
    """Return a MinHash signature whose agreement rate estimates Jaccard.

    Args:
        shingles: A set of shingle strings.
        permutations: Signature length; more permutations sharpen the estimate.
        prime: Modulus for the (a*h + b) mod p hash family.

    Returns:
        A tuple of ``permutations`` minima, one per hash function.
    """
    if not shingles:
        return (0,) * permutations
    hashed = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in shingles]
    return tuple(min(((2 * k + 1) * h + 2654435761 * (k + 1)) % prime for h in hashed)
                 for k in range(permutations))


def near_deduplicate(documents, threshold=0.7, permutations=32, bands=16):
    """Cluster near-duplicate documents and keep one exemplar from each cluster.

    LSH proposes candidate pairs from shared signature bands; each candidate is
    confirmed by an exact Jaccard check above ``threshold`` before its documents
    are merged with union-find. The longest document in a cluster is kept, so
    boilerplate that is merely *shared* across otherwise-distinct documents does
    not collapse them.

    Args:
        documents: Iterable of :class:`Document`.
        threshold: Minimum confirmed Jaccard similarity to merge a pair.
        permutations: MinHash signature length.
        bands: Number of LSH bands; ``permutations`` must divide evenly by it.

    Returns:
        A tuple ``(kept, clusters)`` where ``kept`` is the surviving documents in
        input order and ``clusters`` lists each cluster's member IDs.
    """
    docs = list(documents)
    shingles = [word_shingles(d.text) for d in docs]
    signatures = [minhash_signature(s, permutations) for s in shingles]
    parent = list(range(len(docs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    width = permutations // bands
    buckets = defaultdict(list)
    for i, sig in enumerate(signatures):
        for band in range(bands):
            buckets[(band, sig[band * width:(band + 1) * width])].append(i)
    candidates = set()
    for members in buckets.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                candidates.add((members[x], members[y]))
    for i, j in sorted(candidates):
        if jaccard(shingles[i], shingles[j]) >= threshold:
            union(i, j)

    clusters = defaultdict(list)
    for i in range(len(docs)):
        clusters[find(i)].append(i)
    kept = sorted(max(members, key=lambda i: (len(docs[i].text), -i)) for members in clusters.values())
    return [docs[i] for i in kept], [[docs[i].doc_id for i in members] for members in clusters.values()]


def decontaminate(documents, evaluation_strings, min_chars=20):
    """Remove documents containing a normalized evaluation substring.

    Args:
        documents: Iterable of :class:`Document`.
        evaluation_strings: Known evaluation prompts/answers to exclude.
        min_chars: Ignore needles shorter than this after normalization, so
            trivially short strings do not match everything.

    Returns:
        A tuple ``(kept, removed)`` where ``removed`` items are ``{doc_id}``
        dicts — the documents pulled for touching the evaluation set.
    """
    def normalize(text):
        return " ".join(WORD.findall(text.casefold()))
    needles = [normalize(s) for s in evaluation_strings if len(normalize(s)) >= min_chars]
    kept, removed = [], []
    for doc in documents:
        body = normalize(doc.text)
        if any(needle in body for needle in needles):
            removed.append({"doc_id": doc.doc_id})
        else:
            kept.append(doc)
    return kept, removed


import math


def mix_documents(documents, weights, total):
    """Select an exact per-source quota from documents by target weights.

    Weights are turned into integer quotas by the largest-remainder method, and
    a zero-weight source contributes nothing — the mixture never quietly fills a
    shortfall from another source, so the realized proportions are inspectable.

    Args:
        documents: Iterable of :class:`Document`.
        weights: Mapping from source name to non-negative weight.
        total: Total documents to select.

    Returns:
        The selected documents, grouped by source in name order.
    """
    positive = {s: w for s, w in weights.items() if w > 0}
    eligible = [d for d in documents if d.source in positive]
    total_weight = sum(positive.values())
    exact = {s: total * w / total_weight for s, w in positive.items()}
    quota = {s: math.floor(v) for s, v in exact.items()}
    for s in sorted(positive, key=lambda s: -(exact[s] - quota[s])):
        if sum(quota.values()) >= total:
            break
        quota[s] += 1
    pools = defaultdict(list)
    for d in sorted(eligible, key=lambda d: d.doc_id):
        pools[d.source].append(d)
    selected = []
    for s in sorted(quota):
        selected.extend(pools[s][:quota[s]])
    return selected


def measure_fertility(samples, encode):
    """Measure tokenizer fertility (tokens per character) per language sample.

    Args:
        samples: Mapping from language name to a sample string.
        encode: A tokenizer's ``encode`` function returning a token-ID list.

    Returns:
        A list of ``{language, characters, tokens, tokens_per_char}`` dicts,
        one per sample — the per-character cost the same content incurs.
    """
    rows = []
    for language, text in samples.items():
        tokens = len(encode(text))
        rows.append({"language": language, "characters": len(text), "tokens": tokens,
                     "tokens_per_char": tokens / max(1, len(text))})
    return rows


def recursive_generations(generations=8, sample_size=20, mode="replace", seed=0):
    """Simulate one lineage of a distribution learned from its own samples.

    Each generation fits a Gaussian (maximum-likelihood mean and spread) to the
    current data and draws a fresh sample from it. Under ``"replace"`` the fresh
    sample becomes the next generation's data; under ``"accumulate"`` it is
    pooled with the original real data before resampling, so the real tails keep
    returning.

    Args:
        generations: Number of generations to simulate.
        sample_size: Points per generation (small sizes collapse faster).
        mode: ``"replace"`` or ``"accumulate"``.
        seed: Seed for the generator.

    Returns:
        A tuple ``(spread_by_generation, final_sample)``: the standard deviation
        after each generation and the final generation's samples.
    """
    rng = np.random.default_rng(seed)
    real = rng.normal(0.0, 1.0, sample_size)
    data = real.copy()
    spread = [float(data.std())]
    for _ in range(generations):
        fresh = rng.normal(data.mean(), data.std(), sample_size)
        data = fresh if mode == "replace" else rng.choice(np.concatenate([real, fresh]), sample_size, replace=False)
        spread.append(float(data.std()))
    return spread, data
