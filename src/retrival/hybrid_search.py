from collections import defaultdict
from FlagEmbedding import FlagReranker

def reciprocal_rank_fusion(result_lists , k = 60 , top_n =10):
    scores = defaultdict(float)
    documents = {}
    for result in result_lists:
        for rank , doc in enumerate(result , start = 1):
            doc_id = doc["id"]
            scores[doc_id] += 1/(k + rank)
            documents[doc_id] = doc

    ranked_ids = sorted(
        scores,
        key = lambda doc_id : scores[doc_id],
        reverse = True
    )
    fused_results = []
    for doc_id in ranked_ids[:top_n]:
        fused_results.append({
            **documents[doc_id],
            "rrf_score":scores[doc_id]
        })
    return fused_results

# to give the reranker we must combine the query and doc text to compare and rank
def reranker(query,fused_results,n = 5):
    reranker = FlagReranker(
        "BAAI/bge-reranker-v2-m3",
        use_fp16 = True
    )
    pairs = [[query,doc["text"]] for doc in fused_results]
    scores = reranker.compute_score(pairs)

    for doc,score in zip(fused_results , scores):
        doc["rerank_score"] = score

    reranked_results = sorted(
        fused_results,
        key = lambda doc:doc["rerank_score"],
        reverse = True
    )
    top_documents = reranked_results[:n]
    # we need to return only the top_documents for c rag as it will check the relevant scores unlike the basic rag

    return top_documents
