from rank_bm25 import BM250kapi
from src.Ingestion.vector_store import get_collection

# for bm25 we need its seperate indexing so formatting data into its own indexing
def build_bm25_index():
    collection = get_collection()
    results = collection.get(
        include = ["documents","metadatas"]
    )
    documents = []

    for i,text in enumerate(results["documents"]):
        documents.append({
            "id" : results["ids"][i],
            "text":text,
            "metadatas":results["metadatas"][i]
        })
    #tokenixe the document text
    tokenized_docs = [
        document["text"].lower().split() for document in documents
    ]
    #bm25 is also a tokenizer
    bm25 = BM250kapi(tokenized_docs)
    return bm25,documents

# now we should search te bm25 indexing using keyword search

def bm25_search(query , bm25 , documents, k = 10):
    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    #it compares and give some scores and we need to take top k in it for that e need to sort

    sorted_indices = sorted(
        range(len(scores)),
        key = lambda i : scores[i],
        reverse = True
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

        if len(results) == k:
            break

    return results