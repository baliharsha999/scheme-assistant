import chromadb
from embeddings import embeddings,ids,texts,metadatas

def get_collection():
    client = chromadb.PersistentClient(
        path="./data/ChromaDB"
    )
    collection = client.get_or_create_collection(
        name="welfare_schemes",
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def store_embeddings(ids, texts, metadatas, embeddings):
    collection = get_collection()

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"Total records stored: {collection.count()}")

    return collection

collection = store_embeddings(
    ids = ids,
    embeddings = embeddings,
    texts = texts,
    metadatas = metadatas
)