"""Deterministic tests for Chapter 14's embeddings-and-RAG build.

The chapter's teaching code is authored inline in ``chapters/14-embeddings-rag.qmd``
and tangled to ``code/ch14/_generated.py``; we import that module under a unique
name so it cannot collide with another chapter's generated module.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "ch14" / "corpus.jsonl"
GOLD = ROOT / "data" / "ch14" / "gold.jsonl"


def _load_generated():
    path = ROOT / "code" / "ch14" / "_generated.py"
    spec = importlib.util.spec_from_file_location("ch14_generated", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ch14_generated"] = module
    spec.loader.exec_module(module)
    return module


ch14 = _load_generated()


def _fixture(size: int = 48, overlap: int = 9):
    documents = ch14.load_documents(CORPUS)
    chunks = ch14.chunk_documents(documents, size, overlap)
    return documents, chunks


def test_ranked_metrics_have_distinct_meanings() -> None:
    ranked, relevant = ["c1", "c2", "c3", "c4"], {"c1", "c3"}
    assert ch14.recall_at_k(ranked, relevant, 2) == 0.5
    assert ch14.recall_at_k(ranked, relevant, 4) == 1.0
    assert ch14.reciprocal_rank(ranked, relevant) == 1.0
    assert ch14.average_precision_at_k(ranked, relevant, 4) == pytest.approx((1 + 2 / 3) / 2)
    assert ch14.ndcg_at_k(ranked, {"c1": 2.0, "c3": 1.0}, 4) < 1.0


def test_bm25_surfaces_the_rare_error_code() -> None:
    _, chunks = _fixture()
    bm25 = ch14.BM25(chunks)
    hits = bm25.search("What does error E1492 mean?", 3)
    assert hits[0].source_id == "error-e1492-v5"
    assert bm25.idf("e1492") > bm25.idf("the")


def test_lsa_embedding_learns_a_semantic_geometry() -> None:
    _, chunks = _fixture()
    embedder = ch14.LsaEmbedder(chunks, dim=32)
    related = embedder.cosine("cancel my annual subscription before it renews",
                              "stop automatic renewal of a yearly plan")
    unrelated = embedder.cosine("cancel my annual subscription before it renews",
                                "the payment gateway timed out during checkout")
    assert related > 0.5
    assert related > unrelated + 0.4


def test_dense_retrieval_bridges_a_paraphrase() -> None:
    _, chunks = _fixture()
    dense = ch14.DenseIndex(chunks, ch14.LsaEmbedder(chunks, dim=32))
    sources = {h.source_id for h in dense.search("stop my yearly membership from renewing", 3)}
    assert "cancel-annual-v4" in sources


def test_matryoshka_prefix_holds_recall_at_half_width() -> None:
    _, chunks = _fixture()
    gold = ch14.load_jsonl(GOLD)
    answerable = [g for g in gold if g["answerable"]]
    dense = ch14.DenseIndex(chunks, ch14.LsaEmbedder(chunks, dim=32))

    def recall_at_dim(dim):
        rc = [ch14.recall_at_k(ch14.unique_sources(dense.search(g["query"], 12, dim=dim), 5),
                               set(g["relevant_sources"]), 5) for g in answerable]
        return sum(rc) / len(rc)

    full, half, quarter_of_quarter = recall_at_dim(32), recall_at_dim(16), recall_at_dim(4)
    assert half == pytest.approx(full)          # half the width is lossless here
    assert quarter_of_quarter < full            # aggressive truncation degrades


def test_binary_quantization_is_lossy_and_measured() -> None:
    import numpy as np

    _, chunks = _fixture()
    embedder = ch14.LsaEmbedder(chunks, dim=32)
    gold = ch14.load_jsonl(GOLD)
    answerable = [g for g in gold if g["answerable"]]
    signs = np.sign(embedder.doc_vectors)
    overlaps = []
    for g in answerable:
        exact = set(np.argsort(-(embedder.doc_vectors @ embedder.embed(g["query"])))[:5])
        q_sign = np.sign(embedder.svd.transform(embedder._bow(g["query"]))[0])
        binary = set(np.argsort((signs != q_sign).sum(axis=1))[:5])
        overlaps.append(len(exact & binary) / 5)
    ann_recall = sum(overlaps) / len(overlaps)
    assert 0.0 < ann_recall < 1.0               # a real, lossy compression


def test_rerank_respects_the_candidate_ceiling_and_improves_order() -> None:
    _, chunks = _fixture()
    retriever = ch14.build_retriever(chunks)
    fused, final = retriever.retrieve("Can I get my money back six weeks after buying in the US?")
    assert final[0].source_id == "refund-us-v3"
    assert {h.chunk_id for h in final}.issubset({h.chunk_id for h in fused})


def test_retired_policy_absent_and_chunk_ids_stable() -> None:
    documents = ch14.load_documents(CORPUS)
    first = ch14.chunk_documents(documents, 32, 6)
    second = ch14.chunk_documents(documents, 32, 6)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert all(c.source_id != "refund-us-v2" for c in first)


def test_rrf_depends_on_rank_not_raw_score_scale() -> None:
    _, chunks = _fixture()
    lookup = {c.chunk_id: c for c in chunks}
    one, two = chunks[0], chunks[1]
    a = [ch14.Hit(one.chunk_id, one.source_id, 1_000_000), ch14.Hit(two.chunk_id, two.source_id, -9)]
    b = [ch14.Hit(two.chunk_id, two.source_id, 0.01), ch14.Hit(one.chunk_id, one.source_id, 0.001)]
    merged = ch14.rrf_merge([a, b], lookup)
    assert merged[0].score == pytest.approx(merged[1].score)


def test_query_expansion_condenses_the_ellipsis() -> None:
    variants = ch14.expand_query("What about annual plans?", "How can I cancel before renewal?")
    assert variants[-1] == "cancel annual subscription before renewal"
    assert ch14.expand_query("How do I stop my yearly membership from renewing?")[-1] != \
        "How do I stop my yearly membership from renewing?"


def test_context_is_budgeted_at_source_boundaries_with_provenance() -> None:
    _, chunks = _fixture()
    retriever = ch14.build_retriever(chunks)
    _, final = retriever.retrieve("What does E1492 mean?")
    context = ch14.assemble_context(final, retriever.lookup, budget=70)
    assert 'trust="retrieved-data"' in context
    assert context.count("<source ") == context.count("</source>")
    assert len(context.split()) <= 70


def test_citation_syntax_does_not_prove_support() -> None:
    _, chunks = _fixture()
    retriever = ch14.build_retriever(chunks)
    _, final = retriever.retrieve("What does error E1492 mean?")
    real = ch14.answer_or_abstain("What does error E1492 mean?", final, retriever.lookup, retriever.bm25)
    assert ch14.citation_support(real["answer"], retriever.lookup) == 1.0
    fake = f"Refunds are instant and unlimited. [{final[0].chunk_id}]"
    assert ch14.citation_support(fake, retriever.lookup) == 0.0


def test_abstention_trades_coverage_against_false_answers() -> None:
    _, chunks = _fixture()
    retriever = ch14.build_retriever(chunks)
    gold = ch14.load_jsonl(GOLD)
    negatives = [g for g in gold if not g["answerable"]]
    # At the operating threshold every unanswerable query abstains.
    for g in negatives:
        _, final = retriever.retrieve(g["query"], g.get("history", ""))
        result = ch14.answer_or_abstain(g["query"], final, retriever.lookup, retriever.bm25, 3.5)
        assert result["abstained"], g["id"]
    # A lower threshold lets at least one negative through: no clean separator.
    leaked = 0
    for g in negatives:
        _, final = retriever.retrieve(g["query"], g.get("history", ""))
        if not ch14.answer_or_abstain(g["query"], final, retriever.lookup, retriever.bm25, 2.5)["abstained"]:
            leaked += 1
    assert leaked >= 1


def test_faithfulness_decomposes_claims() -> None:
    grounded = "Error code E1492 means the payment gateway timed out before returning a result."
    context = "Error code E1492 means the payment gateway timed out before returning a final result."
    assert ch14.faithfulness(grounded, context) == pytest.approx(1.0)
    mixed = grounded + " Refunds are instant and unlimited."
    assert ch14.faithfulness(mixed, context) == pytest.approx(0.5)


def test_evaluate_pipeline_reports_stages_and_is_deterministic() -> None:
    documents = ch14.load_documents(CORPUS)
    gold = ch14.load_jsonl(GOLD)
    first = ch14.evaluate_pipeline(documents, gold)
    second = ch14.evaluate_pipeline(documents, gold)
    assert first == second
    assert first["candidate_recall@12"] == pytest.approx(1.0)
    assert first["retrieval"]["reranked"]["recall@5"] == pytest.approx(1.0)
    assert first["retrieval"]["reranked"]["mrr"] == pytest.approx(1.0)
    assert first["retrieval"]["reranked"]["mrr"] >= first["retrieval"]["hybrid"]["mrr"]
    assert first["generation"]["faithfulness"] == pytest.approx(1.0)
    # Faithful is not the same as complete: the lexical answer F1 stays well below 1.
    assert first["generation"]["answer_f1"] < 0.6
