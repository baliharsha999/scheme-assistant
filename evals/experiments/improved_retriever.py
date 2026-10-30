"""
evals/experiments/improved_retriever.py

Improved Hybrid Retriever implementing:
1. Target Collection: 'welfare_schemes_v2' (contains 900-char context-enriched chunks)
2. Expanded Candidate Retrieval: Retrieves top-20 dense candidates and top-20 BM25 candidates
   (vs. top-10 in baseline) to ensure high recall prior to reranking.
3. Reciprocal Rank Fusion (top_n=20, k=60)
4. BAAI/bge-reranker-v2-m3 Cross-Encoder Reranking down to top-k (default 5).
"""

import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import chromadb
from pathlib import Path

from src.retrieval.hybrid_search import reranker, reciprocal_rank_fusion
from src.retrieval.dense_retriever import get_embeddings_model
from evals.experiments.ingest_improved import get_improved_collection, DB_PATH

def tokenize_text(text: str) -> List[str]:
    """Tokenize alphanumeric words, stripping punctuation cleanly."""
    return re.findall(r"\w+", text.lower())


def get_improved_bm25_index():
    collection = get_improved_collection()
    results = collection.get(include=["documents", "metadatas"])

    documents = []
    doc_list = results.get("documents") or []
    ids_list = results.get("ids") or []
    meta_list = results.get("metadatas") or []

    for i, text in enumerate(doc_list):
        documents.append({
            "id": ids_list[i],
            "text": text,
            "metadata": meta_list[i] if i < len(meta_list) else {}
        })

    if not documents:
        return None, []

    tokenized_docs = [
        tokenize_text(doc["text"])
        for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_docs)
    return bm25, documents


def improved_bm25_search(query: str, k: int = 20) -> List[Dict[str, Any]]:
    bm25, documents = get_improved_bm25_index()
    if bm25 is None or not documents:
        return []

    query_tokens = tokenize_text(query)
    if not query_tokens:
        return []
    scores = bm25.get_scores(query_tokens)

    sorted_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []
    for i in sorted_indices:
        if scores[i] <= 0:
            continue
        results.append({
            "id": documents[i]["id"],
            "text": documents[i]["text"],
            "metadata": documents[i]["metadata"],
            "bm25_score": float(scores[i])
        })
        if len(results) >= k:
            break

    return results


def improved_dense_search(query: str, k: int = 20) -> List[Dict[str, Any]]:
    collection = get_improved_collection()
    embed_model = get_embeddings_model()
    query_embeddings = embed_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embeddings,
        include=["documents", "metadatas", "distances"],
        n_results=k
    )

    documents = []
    if results and results.get("documents") and len(results["documents"]) > 0:
        for i, text in enumerate(results["documents"][0]):
            documents.append({
                "id": results["ids"][0][i],
                "text": text,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

    return documents


def improved_retrieve_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Improved hybrid search pipeline:
    BM25 (k=20) + Dense (k=20) on welfare_schemes_v2 -> RRF (top 20) -> BGE Reranker (top 5).
    """
    bm25_results = improved_bm25_search(query, k=20)
    dense_results = improved_dense_search(query, k=20)
    fused_results = reciprocal_rank_fusion([bm25_results, dense_results], top_n=20)
    reranked_results = reranker(query, fused_results, n=top_k)
    return reranked_results
