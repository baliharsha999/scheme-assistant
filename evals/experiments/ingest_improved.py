"""
evals/experiments/ingest_improved.py

Embeds context-enriched chunks and stores them into an isolated ChromaDB collection:
"welfare_schemes_v2".

Strictly isolates experimental vector embeddings from the original "welfare_schemes"
production collection.
Uses memory-safe batch encoding (batch_size=8, max_seq_length=512, torch threads=4)
to prevent CPU RAM exhaustion on Windows.
"""

import argparse
from pathlib import Path
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from evals.experiments.improved_chunking import build_improved_chunks

# Optimize CPU thread allocation to prevent thread thrashing and memory exhaustion
torch.set_num_threads(4)

DB_PATH = Path("./data/ChromaDB").resolve()
IMPROVED_COLLECTION_NAME = "welfare_schemes_v2"


def get_improved_collection(client: chromadb.PersistentClient = None):
    if client is None:
        client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(
        name=IMPROVED_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def ingest_improved_chunks(batch_size: int = 8, force: bool = False):
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = get_improved_collection(client)

    current_count = collection.count()
    if current_count > 0 and not force:
        print(f"Collection '{IMPROVED_COLLECTION_NAME}' already contains {current_count} documents.")
        print("Use --force to re-embed and overwrite.")
        return collection

    if force and current_count > 0:
        print(f"Deleting existing collection '{IMPROVED_COLLECTION_NAME}' for fresh ingestion...")
        client.delete_collection(IMPROVED_COLLECTION_NAME)
        collection = get_improved_collection(client)

    print("Generating context-enriched chunks (chunk_size=900, overlap=150)...")
    ids, texts, metadatas = build_improved_chunks()
    total = len(texts)
    print(f"Prepared {total} chunks. Generating BAAI/bge-m3 dense embeddings...")

    embed_model = SentenceTransformer("BAAI/bge-m3")
    embed_model.max_seq_length = 512

    vectors = embed_model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    print(f"Upserting {len(vectors)} vectors into '{IMPROVED_COLLECTION_NAME}'...")
    upsert_batch = 250
    for i in range(0, total, upsert_batch):
        collection.upsert(
            ids=ids[i:i + upsert_batch],
            documents=texts[i:i + upsert_batch],
            metadatas=metadatas[i:i + upsert_batch],
            embeddings=vectors[i:i + upsert_batch]
        )

    print(f"Successfully stored {collection.count()} chunks in '{IMPROVED_COLLECTION_NAME}'.")
    return collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest improved context-enriched chunks")
    parser.add_argument("--force", action="store_true", help="Force re-indexing")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding batch size")
    args = parser.parse_args()

    ingest_improved_chunks(batch_size=args.batch_size, force=args.force)
