import chromadb
from src.Ingestion.embeddings import embeddings

def get_collection():
    client = chromadb.PersistentClient(
        path="./data/ChromaDB"
    )

    collection = client.get_or_create_collection(
        name="welfare_schemes",
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def store_embeddings():
    vectors, ids, texts, metadatas = embeddings()

    collection = get_collection()
    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=vectors
    )
    return collection

if __name__ == "__main__":
    collection = store_embeddings()