import chromadb
from pathlib import Path

from src.Ingestion.embeddings import embeddings


# Create one shared ChromaDB client
DB_PATH = Path("./data/ChromaDB").resolve()

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)


def get_collection():
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
    print("Documents stored:", collection.count())