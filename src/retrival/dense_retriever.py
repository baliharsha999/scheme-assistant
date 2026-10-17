from sentence_transformers import SentenceTransformer
from src.Ingestion.vector_store import get_collection

embeddings_model = SentenceTransformer("BAAI/bge-m3")

def dense_search(query , k = 10):
    # to search in vector database we need to get collection
    collection = get_collection()
    # we need to convert the query into embeddins
    query_embeddings = embeddings_model.encode(
        query,
        normalize_embeddings = True
    ).tolist()
    # getting the relevant documents
    results = collection.query(
        query_embeddings = query_embeddings,
        include = ["documents","metadatas","distances"],
        n_results = k
    )

    documents = []

    for i,text in enumerate(results["documents"][0]):
        documents.append({
            "id":results["ids"][0][i],
            "text" : text,
            "metadata":results["metadatas"][0][i],
            "distance":results["distances"][0][i]
        })

    return documents