from collections import defaultdict
from FlagEmbedding import FlagReranker

reranker_model = None


def get_reranker_model():
    """Lazily load the cross-encoder reranker to prevent heavy memory allocation at import time."""
    global reranker_model
    if reranker_model is None:
        reranker_model = FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=False,
            low_cpu_mem_usage=True
        )
    return reranker_model


def reciprocal_rank_fusion(result_lists, k=60, top_n=10):
    scores = defaultdict(float)
    documents = {}
    for result in result_lists:
        for rank, doc in enumerate(result, start=1):
            doc_id = doc["id"]
            scores[doc_id] += 1 / (k + rank)
            documents[doc_id] = doc

    ranked_ids = sorted(
        scores,
        key=lambda doc_id: scores[doc_id],
        reverse=True
    )
    fused_results = []
    for doc_id in ranked_ids[:top_n]:
        fused_results.append({
            **documents[doc_id],
            "rrf_score": scores[doc_id]
        })
    return fused_results


def reranker(query, fused_results, n=5):
    if not fused_results:
        return []

    try:
        model = get_reranker_model()
        pairs = [
            [query, doc["text"]]
            for doc in fused_results
        ]

        scores = model.compute_score(pairs)

        if isinstance(scores, (int, float)):
            scores = [scores]

        reranked_results = []
        for doc, score in zip(fused_results, scores):
            reranked_results.append({
                **doc,
                "rerank_score": float(score)
            })

        reranked_results.sort(
            key=lambda doc: doc["rerank_score"],
            reverse=True
        )
        return reranked_results[:n]

    except Exception as e:
        print(f"[Reranker Warning] Falling back to RRF rankings due to system memory limit: {e}")
        return fused_results[:n]
