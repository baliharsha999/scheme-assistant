from sentence_transformers import SentenceTransformer
from src.Ingestion.vector_store import get_collection

embeddings_model = None


def get_embeddings_model():
    """Lazily load the dense embedding model to prevent memory spikes during module imports."""
    global embeddings_model
    if embeddings_model is None:
        embeddings_model = SentenceTransformer("BAAI/bge-m3")
    return embeddings_model


def dense_search(query, k=10):
    collection = get_collection()
    model = get_embeddings_model()
    query_embeddings = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embeddings,
        include=["documents", "metadatas", "distances"],
        n_results=k
    )

    documents = []
    for i, text in enumerate(results["documents"][0]):
        documents.append({
            "id": results["ids"][0][i],
            "text": text,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })

    return documents
