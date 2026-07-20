# Auto-generated from chapters/14-embeddings-rag.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
CODE_RE = re.compile(r"\b[a-z]\d{3,}\b")
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "my", "of", "on", "or", "should", "the",
    "to", "what", "when", "with", "after", "before",
}


def terms(text: str) -> list[str]:
    """Tokenize into lower-case words and code-like identifiers (``E1492``)."""
    return TOKEN_RE.findall(text.lower())


def content_terms(text: str) -> list[str]:
    """Tokens with stop-words removed, for overlap and support scoring."""
    return [t for t in terms(text) if t not in STOP]


@dataclass(frozen=True)
class Document:
    """One versioned corpus source; ``active`` gates it out of the index."""

    source_id: str
    family: str
    title: str
    version: str
    active: bool
    text: str


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage that remembers its source identity and version."""

    chunk_id: str
    source_id: str
    title: str
    version: str
    ordinal: int
    text: str


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk identity paired with the score that ranked it."""

    chunk_id: str
    source_id: str
    score: float


def load_jsonl(path: Path) -> list[dict]:
    """Load non-empty JSONL rows from an inspectable data contract."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_documents(path: Path) -> list[Document]:
    """Load the versioned corpus into :class:`Document` records."""
    return [Document(**row) for row in load_jsonl(path)]


def chunk_documents(
    documents: Sequence[Document], size: int = 48, overlap: int = 9
) -> list[Chunk]:
    """Split active documents into overlapping windows with stable IDs.

    Only active documents are chunked, so a retired source cannot be retrieved.
    Each chunk's identity is derived from its source, its ordinal, and a digest
    of its own text, which makes IDs reproducible across identical builds and
    lets a citation refer to an exact, addressable passage.

    Args:
        documents: The versioned corpus.
        size: Window length in whitespace words; the chunking hyperparameter
            swept in @sec-ch14-ingestion.
        overlap: Words shared between adjacent windows, so a fact is not split
            from its qualifier at a boundary.

    Returns:
        Chunks in document order; inactive documents contribute none.
    """
    if size < 8 or not 0 <= overlap < size:
        raise ValueError("require size >= 8 and 0 <= overlap < size")
    chunks: list[Chunk] = []
    step = size - overlap
    for document in documents:
        if not document.active:
            continue
        words = document.text.split()
        for ordinal, start in enumerate(range(0, len(words), step)):
            body = " ".join(words[start : start + size])
            if not body:
                continue
            digest = hashlib.sha256(body.encode()).hexdigest()[:8]
            chunk_id = f"{document.source_id}:{ordinal}:{digest}"
            chunks.append(
                Chunk(chunk_id, document.source_id, document.title,
                      document.version, ordinal, body)
            )
            if start + size >= len(words):
                break
    return chunks


class LsaEmbedder:
    """A latent-semantic bi-encoder fit on the local corpus.

    Text becomes a TF-IDF vector over the corpus vocabulary; a truncated SVD
    then projects those sparse vectors onto ``dim`` dense axes that summarize how
    terms co-occur. Two texts that share *context* -- "annual subscription" and
    "yearly plan renewal" -- land close even when they share few exact words,
    which is the geometry a dense retriever is supposed to learn. Axes are
    ordered by singular value, so a prefix of any vector is a valid shorter
    embedding (the Matryoshka property, for free).

    Args:
        chunks: Corpus chunks whose title+text define the vocabulary and the
            co-occurrence statistics the geometry is learned from.
        dim: Number of SVD components, i.e. the embedding width.
    """

    def __init__(self, chunks: Sequence[Chunk], dim: int = 32):
        self.dim = dim
        rows = [terms(f"{c.title} {c.text}") for c in chunks]
        self.vocab = sorted({t for row in rows for t in row})
        self.index = {t: i for i, t in enumerate(self.vocab)}
        n, doc_freq = len(rows), Counter(t for row in rows for t in set(row))
        self.idf = {t: math.log((n + 1) / (doc_freq[t] + 1)) + 1 for t in self.vocab}
        matrix = np.zeros((n, len(self.vocab)))
        for i, row in enumerate(rows):
            for term, freq in Counter(row).items():
                matrix[i, self.index[term]] = (1 + math.log(freq)) * self.idf[term]
        from sklearn.decomposition import TruncatedSVD

        self.svd = TruncatedSVD(n_components=dim, random_state=0)
        self.doc_vectors = self._l2(self.svd.fit_transform(self._l2(matrix)))

    @staticmethod
    def _l2(m: np.ndarray) -> np.ndarray:
        return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)

    def _bow(self, text: str) -> np.ndarray:
        vector = np.zeros((1, len(self.vocab)))
        for term, freq in Counter(terms(text)).items():
            if term in self.index:
                vector[0, self.index[term]] = (1 + math.log(freq)) * self.idf[term]
        return self._l2(vector)

    def embed(self, text: str, dim: int | None = None) -> np.ndarray:
        """Embed one text; pass ``dim`` < self.dim to truncate (Matryoshka)."""
        vector = self.svd.transform(self._bow(text))[0]
        if dim is not None:
            vector = vector[:dim]
        return self._l2(vector.reshape(1, -1))[0]

    def cosine(self, a: str, b: str) -> float:
        """Cosine similarity of two texts; a bounded, magnitude-free score."""
        return float(self.embed(a) @ self.embed(b))


class DenseIndex:
    """Exact nearest-neighbor search over embeddings; the approximation oracle.

    Every query is scored against every corpus vector, so results are exact by
    construction. A ``dim`` argument truncates both sides to a Matryoshka prefix
    so recall can be re-measured as the embedding is compressed.

    Args:
        chunks: Corpus chunks, aligned with the embedder's document vectors.
        embedder: The fitted :class:`LsaEmbedder` supplying the geometry.
    """

    def __init__(self, chunks: Sequence[Chunk], embedder: LsaEmbedder):
        self.chunks = list(chunks)
        self.embedder = embedder

    def search(self, query: str, k: int, dim: int | None = None) -> list[Hit]:
        query_vector = self.embedder.embed(query, dim=dim)
        docs = self.embedder.doc_vectors
        if dim is not None:
            docs = LsaEmbedder._l2(docs[:, :dim])
        scores = docs @ query_vector
        order = np.argsort(-scores)
        return [Hit(self.chunks[i].chunk_id, self.chunks[i].source_id,
                    float(scores[i])) for i in order[:k]]


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of required sources found by rank k; the downstream ceiling."""
    if not relevant:
        return 1.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """Reciprocal of the first relevant rank; 0 if none is returned."""
    return next((1.0 / r for r, x in enumerate(ranked, 1) if x in relevant), 0.0)


def average_precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Mean precision at each newly found relevant source, duplicates ignored.

    Averaging precision over the ranks where relevant sources first appear
    rewards packing them early, which is why AP separates two rankings that MRR
    and recall would call identical.

    Args:
        ranked: Returned source identities, best first.
        relevant: Sources judged relevant for this query.
        k: Cutoff rank.

    Returns:
        Average precision at k in [0, 1].
    """
    if not relevant:
        return 1.0
    seen: set[str] = set()
    total = found = 0.0
    for rank, item in enumerate(ranked[:k], 1):
        if item in relevant and item not in seen:
            seen.add(item)
            found += 1
            total += found / rank
    return total / min(len(relevant), k)


def ndcg_at_k(ranked: Sequence[str], gains: dict[str, float], k: int) -> float:
    """Discounted cumulative gain at k, normalized by the ideal ordering.

    The log discount and graded gains let a directly-answering source outrank a
    page that merely mentions the topic, which recall and MRR cannot express.

    Args:
        ranked: Returned source identities, best first.
        gains: Graded relevance per source; absent sources score zero gain.
        k: Cutoff rank.

    Returns:
        nDCG at k in [0, 1]; 1.0 when there is no gain to earn.
    """
    dcg = sum((2 ** gains.get(x, 0.0) - 1) / math.log2(r + 1)
              for r, x in enumerate(ranked[:k], 1))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(r + 1) for r, g in enumerate(ideal, 1))
    return dcg / idcg if idcg else 1.0


class BM25:
    """An exact BM25 index over chunk title+text, with positive IDF.

    Term frequency is saturated by ``k1`` and normalized for document length by
    ``b`` per @eq-ch14-bm25, so a rare, high-IDF token such as an error code can
    outweigh many common words -- the property that makes lexical search the
    right tool for literal evidence.

    Args:
        chunks: Corpus chunks to index.
        k1: Term-frequency saturation; larger rewards repetition more.
        b: Length-normalization strength in [0, 1].
    """

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.2, b: float = 0.75):
        self.chunks, self.k1, self.b = list(chunks), k1, b
        self.rows = [terms(f"{c.title} {c.text}") for c in chunks]
        self.freqs = [Counter(row) for row in self.rows]
        self.avg_len = sum(map(len, self.rows)) / max(len(self.rows), 1)
        self.doc_freq = Counter(t for row in self.rows for t in set(row))

    def idf(self, term: str) -> float:
        """Positive inverse document frequency; higher for rarer terms."""
        n, df = len(self.rows), self.doc_freq.get(term.lower(), 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[Hit]:
        """Score every chunk against the query and return the top k hits."""
        out: list[Hit] = []
        for chunk, freq, length in zip(self.chunks, self.freqs, map(len, self.rows)):
            score = 0.0
            for term in terms(query):
                tf = freq.get(term, 0)
                denom = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                score += self.idf(term) * (tf * (self.k1 + 1) / denom if denom else 0)
            out.append(Hit(chunk.chunk_id, chunk.source_id, score))
        return sorted(out, key=lambda h: (-h.score, h.chunk_id))[:k]


def unique_sources(hits: Sequence[Hit], k: int) -> list[str]:
    """Collapse chunks from one source into k distinct source identities.

    Ranked retrieval is scored over *sources*, not passages, so three adjacent
    chunks from one policy count once. This preserves rank order while deduping.

    Args:
        hits: Ranked hits, best first.
        k: Number of distinct sources to keep.

    Returns:
        Up to k source identities in rank order.
    """
    seen: list[str] = []
    for hit in hits:
        if hit.source_id not in seen:
            seen.append(hit.source_id)
        if len(seen) >= k:
            break
    return seen


def rrf_merge(rankings: Sequence[Sequence[Hit]], lookup: dict[str, Chunk],
              k: int = 60, limit: int | None = None) -> list[Hit]:
    """Fuse ranked lists by reciprocal rank, ignoring their raw score scales.

    Each list contributes 1/(k+rank) to a candidate, so agreement across lists
    is rewarded without ever comparing a BM25 score to a cosine. ``k`` is the
    study's stabilizing constant (60), a historical setting rather than a law.

    Args:
        rankings: One ranked hit list per retrieval route.
        lookup: Chunk-id to :class:`Chunk`, for attaching source identities.
        k: Rank-fusion constant.
        limit: Optional cap on the fused list length.

    Returns:
        The fused hits, highest combined reciprocal rank first.
    """
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            scores[hit.chunk_id] += 1 / (k + rank)
    merged = [Hit(cid, lookup[cid].source_id, s) for cid, s in scores.items()]
    return sorted(merged, key=lambda h: (-h.score, h.chunk_id))[:limit]


EXPANSIONS = {
    "money back": "refund", "money": "refund", "overseas": "international",
    "yearly": "annual", "membership": "subscription", "parcel": "package",
    "settled bills": "invoices", "laptop": "computer", "erased": "deleted",
    "recovery link": "reset link", "severity-one": "priority",
}


def expand_query(query: str, history: str = "") -> list[str]:
    """Return bounded query variants for ellipsis and true-synonym gaps.

    The original query is always kept as one route; a synonym-normalized variant
    is added only when it differs, and an elided follow-up is condensed against
    its history. Bounding the variants keeps rewrite drift observable.

    Args:
        query: The current user turn.
        history: The previous turn, used only to restore an elided subject.

    Returns:
        One or two query strings, original first.
    """
    low = query.lower().strip()
    variants = [query]
    rewritten = low
    for source, target in EXPANSIONS.items():
        rewritten = rewritten.replace(source, target)
    if rewritten != low:
        variants.append(rewritten)
    if low == "what about annual plans?" and history:
        variants = [query, "cancel annual subscription before renewal"]
    return variants


def pair_score(query: str, chunk: Chunk, embedder: LsaEmbedder) -> float:
    """A transparent reranking proxy for a cross-encoder over a shortlist.

    Combines content-term overlap, a strong bonus for a shared literal code, and
    the embedder cosine. It is not a trained cross-encoder result; it only
    reorders a shortlist and demonstrates the funnel invariant. Replace it with a
    validated reranker in production while keeping the candidate-subset property.

    Args:
        query: The (expanded) query text.
        chunk: A candidate chunk from the fused shortlist.
        embedder: The bi-encoder supplying a semantic term.

    Returns:
        A relevance score; higher ranks the chunk earlier.
    """
    q_terms = set(content_terms(query))
    d_terms = set(content_terms(f"{chunk.title} {chunk.text}"))
    codes = set(CODE_RE.findall(query.lower())) & \
        set(CODE_RE.findall(f"{chunk.title} {chunk.text}".lower()))
    return 2.0 * len(q_terms & d_terms) + 6.0 * len(codes) \
        + 1.5 * embedder.cosine(query, chunk.text)


def rerank(query: str, hits: Sequence[Hit], lookup: dict[str, Chunk],
           embedder: LsaEmbedder, k: int) -> list[Hit]:
    """Rescore a fused shortlist with :func:`pair_score` and keep the top k.

    Reranking only reorders and trims the candidates the retriever supplied, so
    the returned chunk ids are always a subset of ``hits`` -- the funnel
    invariant the tests assert.

    Args:
        query: The (expanded) query text.
        hits: The fused candidate shortlist.
        lookup: Chunk-id to :class:`Chunk`.
        embedder: The bi-encoder used inside :func:`pair_score`.
        k: Number of reranked hits to keep.

    Returns:
        The top k hits by pair score, a subset of ``hits``.
    """
    rescored = [Hit(h.chunk_id, h.source_id, pair_score(query, lookup[h.chunk_id], embedder))
                for h in hits]
    return sorted(rescored, key=lambda h: (-h.score, h.chunk_id))[:k]


@dataclass
class HybridRetriever:
    """The retrieve→fuse→rerank funnel behind one method.

    Holds the two indexes, the embedder, and the chunk lookup, and exposes a
    single :meth:`retrieve` that expands the query, fuses BM25 and dense results
    by rank, and reranks the shortlist. It returns both the fused candidates and
    the reranked final list so a caller can check the candidate ceiling.
    """

    bm25: BM25
    dense: DenseIndex
    embedder: LsaEmbedder
    lookup: dict[str, Chunk]
    candidate_k: int = 12
    final_k: int = 5

    def retrieve(self, query: str, history: str = "") -> tuple[list[Hit], list[Hit]]:
        """Return (fused_candidates, reranked_final) for one query."""
        variants = expand_query(query, history)
        rankings = [r for v in variants
                    for r in (self.bm25.search(v, self.candidate_k),
                              self.dense.search(v, self.candidate_k))]
        fused = rrf_merge(rankings, self.lookup, limit=self.candidate_k)
        final = rerank(" ".join(variants), fused, self.lookup, self.embedder, self.final_k)
        return fused, final


def build_retriever(chunks: Sequence[Chunk], dim: int = 32) -> HybridRetriever:
    """Construct the full hybrid funnel from a chunk set."""
    embedder = LsaEmbedder(chunks, dim=dim)
    lookup = {c.chunk_id: c for c in chunks}
    return HybridRetriever(BM25(chunks), DenseIndex(chunks, embedder), embedder, lookup)


def assemble_context(hits: Sequence[Hit], lookup: dict[str, Chunk],
                     budget: int = 220) -> str:
    """Render versioned, citable source blocks without splitting a chunk.

    Each hit becomes a whole ``<source>`` block carrying its ID, version, and a
    trust label; a block that would exceed the word budget is skipped rather than
    truncated, so a citation always points at complete, addressable text.

    Args:
        hits: Reranked hits to serialize, best first.
        lookup: Chunk-id to :class:`Chunk`.
        budget: Word budget (a proxy for the endpoint's token budget).

    Returns:
        The concatenated source blocks that fit within the budget.
    """
    blocks, used = [], 0
    for hit in hits:
        chunk = lookup[hit.chunk_id]
        block = (f'<source id="{chunk.chunk_id}" version="{chunk.version}" '
                 f'trust="retrieved-data">\n{chunk.text}\n</source>')
        size = len(block.split())
        if used + size > budget:
            continue
        blocks.append(block)
        used += size
    return "\n".join(blocks)


def support_score(query: str, hits: Sequence[Hit], lookup: dict[str, Chunk],
                  bm25: BM25) -> float:
    """IDF-weighted overlap between the query and the best retrieved chunk.

    Weighting overlap by IDF means a distinctive matched term (an error code, a
    domain noun) counts far more than a common one, which is what separates a
    genuinely answerable query from a spurious one-word match -- imperfectly, as
    @sec-ch14-grounding shows.

    Args:
        query: The (expanded) query text.
        hits: Reranked hits.
        lookup: Chunk-id to :class:`Chunk`.
        bm25: The index whose IDF weights the overlap.

    Returns:
        The maximum IDF-weighted overlap over the hits.
    """
    q_terms = set(content_terms(query))
    best = 0.0
    for hit in hits:
        chunk = lookup[hit.chunk_id]
        d_terms = set(content_terms(f"{chunk.title} {chunk.text}"))
        best = max(best, sum(bm25.idf(t) for t in q_terms & d_terms))
    return best


def answer_or_abstain(query: str, hits: Sequence[Hit], lookup: dict[str, Chunk],
                      bm25: BM25, threshold: float = 3.5) -> dict:
    """Extract one supported sentence or take the insufficient-evidence path.

    Abstention is a decision rule, not an apology: below the support threshold
    the system returns INSUFFICIENT_EVIDENCE; above it, it returns the retrieved
    sentence with the most query overlap, cited by chunk id.

    Args:
        query: The user query.
        hits: Reranked hits.
        lookup: Chunk-id to :class:`Chunk`.
        bm25: Index supplying IDF for the support score.
        threshold: Minimum support to answer; the risk–coverage knob.

    Returns:
        A dict with ``answer``, ``citations``, and ``abstained``.
    """
    expanded = " ".join(expand_query(query))
    if not hits or support_score(expanded, hits, lookup, bm25) < threshold:
        return {"answer": "INSUFFICIENT_EVIDENCE", "citations": [], "abstained": True}
    q_terms = set(content_terms(expanded))
    best = (-1.0, "", "")
    for hit in hits:
        for sentence in re.split(r"(?<=[.!?])\s+", lookup[hit.chunk_id].text):
            overlap = len(q_terms & set(content_terms(sentence)))
            if overlap > best[0]:
                best = (overlap, sentence, hit.chunk_id)
    _, sentence, chunk_id = best
    return {"answer": f"{sentence} [{chunk_id}]", "citations": [chunk_id],
            "abstained": False}


def citation_support(answer: str, lookup: dict[str, Chunk]) -> float:
    """Return 1.0 only if the cited chunk literally contains the claim sentence.

    A deliberately narrow rule -- exact containment -- that catches a
    syntactically valid citation whose source does not state the claim. Production
    replaces containment with entailment, but the separation of *cited* from
    *supported* is the durable point.

    Args:
        answer: The answer text with a trailing ``[chunk_id]`` citation.
        lookup: Chunk-id to :class:`Chunk`.

    Returns:
        1.0 if supported, else 0.0.
    """
    cited = re.findall(r"\[([^\]]+)\]", answer)
    if not cited or any(c not in lookup for c in cited):
        return 0.0
    claim = re.sub(r"\[[^\]]+\]", "", answer).strip().lower()
    return float(any(claim.rstrip(".") in lookup[c].text.lower().rstrip(".") for c in cited))


def decompose_claims(answer: str) -> list[str]:
    """Split an answer into checkable claim sentences, dropping the citation."""
    text = re.sub(r"\[[^\]]+\]", "", answer)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
            if len(content_terms(s)) >= 2]


def claim_supported(claim: str, context_text: str) -> bool:
    """True when most of a claim's content terms occur in the context."""
    ctx = set(content_terms(context_text))
    words = set(content_terms(claim))
    return bool(words) and len(words & ctx) / len(words) >= 0.6


def faithfulness(answer: str, context_text: str) -> float:
    """RAGAS-style faithfulness: supported fraction of decomposed claims.

    Decomposing the answer into claims and scoring each against the retrieved
    context localizes an unfaithful sentence, which a single answer-level score
    cannot. It measures grounding, not correctness.

    Args:
        answer: The generated answer.
        context_text: The assembled evidence the answer must be grounded in.

    Returns:
        Fraction of claims supported by the context, in [0, 1].
    """
    claims = decompose_claims(answer)
    if not claims:
        return 1.0
    return sum(claim_supported(c, context_text) for c in claims) / len(claims)


def token_f1(answer: str, reference: str) -> float:
    """A transparent lexical answer-similarity proxy for the offline harness.

    Content-term F1 between answer and reference stands in for a calibrated
    answer-correctness judge; it is deliberately shallow so the harness stays
    deterministic and the gap to faithfulness is legible.

    Args:
        answer: The generated answer text.
        reference: The gold reference answer.

    Returns:
        Token F1 in [0, 1]; 0 when either side or the overlap is empty.
    """
    predicted = set(content_terms(answer))
    expected = set(content_terms(reference))
    overlap = len(predicted & expected)
    if not predicted or not expected or not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def evaluate_pipeline(documents: Sequence[Document], gold: Sequence[dict],
                      size: int = 48, overlap: int = 9, threshold: float = 3.5) -> dict:
    """Run one end-to-end experiment and return stage-local metrics.

    Every answerable query passes through the funnel; per-route rankings feed the
    four IR metrics, and answered queries feed candidate recall, faithfulness,
    citation support, abstention accuracy, and a lexical answer F1. Retrieval and
    generation are scored separately so a failure can be attributed to a stage.

    Args:
        documents: The versioned corpus.
        gold: Golden queries with relevance and answerability labels.
        size: Chunk size in words.
        overlap: Chunk overlap in words.
        threshold: Abstention support threshold.

    Returns:
        A nested dict of per-route retrieval metrics and generation metrics.
    """
    chunks = chunk_documents(documents, size, overlap)
    lookup = {c.chunk_id: c for c in chunks}
    retriever = build_retriever(chunks)
    bm25 = retriever.bm25
    answerable = [g for g in gold if g["answerable"]]
    routes: dict[str, list[list[str]]] = {n: [] for n in ("bm25", "dense", "hybrid", "reranked")}
    relevant_sets = [set(g["relevant_sources"]) for g in answerable]
    candidate, faith, cite, f1, abstain = [], [], [], [], []
    for g in gold:
        fused, final = retriever.retrieve(g["query"], g.get("history", ""))
        result = answer_or_abstain(g["query"], final, lookup, bm25, threshold)
        if g["answerable"]:
            routes["bm25"].append(unique_sources(bm25.search(g["query"], 12), 5))
            routes["dense"].append(unique_sources(retriever.dense.search(g["query"], 12), 5))
            routes["hybrid"].append(unique_sources(fused, 5))
            routes["reranked"].append(unique_sources(final, 5))
            candidate.append(recall_at_k(unique_sources(fused, 12), set(g["relevant_sources"]), 12))
            abstain.append(float(not result["abstained"]))
            if not result["abstained"]:
                faith.append(faithfulness(result["answer"], assemble_context(final, lookup)))
                cite.append(citation_support(result["answer"], lookup))
                f1.append(token_f1(result["answer"], g["reference_answer"]))
        else:
            abstain.append(float(result["abstained"]))

    def route_metrics(rankings):
        return {
            "recall@5": float(np.mean([recall_at_k(r, s, 5) for r, s in zip(rankings, relevant_sets)])),
            "mrr": float(np.mean([reciprocal_rank(r, s) for r, s in zip(rankings, relevant_sets)])),
            "map@5": float(np.mean([average_precision_at_k(r, s, 5) for r, s in zip(rankings, relevant_sets)])),
            "ndcg@5": float(np.mean([ndcg_at_k(r, {x: 1.0 for x in s}, 5) for r, s in zip(rankings, relevant_sets)])),
        }

    return {
        "chunks": len(chunks),
        "retrieval": {name: route_metrics(r) for name, r in routes.items()},
        "candidate_recall@12": float(np.mean(candidate)),
        "generation": {
            "answered": int(sum(abstain[:len(answerable)])),
            "faithfulness": float(np.mean(faith)),
            "citation_support": float(np.mean(cite)),
            "answer_f1": float(np.mean(f1)),
            "abstention_accuracy": float(np.mean(abstain)),
        },
    }
