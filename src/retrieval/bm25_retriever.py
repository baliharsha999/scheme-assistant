from rank_bm25 import BM25Okapi
from src.Ingestion.vector_store import get_collection

def build_bm25_index():

    collection = get_collection()

    results = collection.get(
        include=["documents", "metadatas"]
    )

    documents = []

    for i, text in enumerate(results["documents"] or []):

        documents.append({
            "id": results["ids"][i],
            "text": text,
            "metadata": (results["metadatas"] or [])[i]
        })

    if not documents:
        return None, []

    tokenized_docs = [
        document["text"].lower().split()
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_docs)

    return bm25, documents

def bm25_search(query, k=10):

    bm25, documents = build_bm25_index()

    if bm25 is None:
        return []

    query_tokens = query.lower().split()

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
