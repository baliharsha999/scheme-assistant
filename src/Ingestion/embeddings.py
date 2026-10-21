from sentence_transformers import SentenceTransformer
from src.Ingestion.chunking import chunked_documents


def get_embedding_model():
    return SentenceTransformer("BAAI/bge-m3")

def prepare_data(chunked_docs):
    ids = []
    texts = []
    metadatas = []

    for document in chunked_docs:
        text = document["text"]
        metadata = document["metadata"]

        source = metadata["source"]
        page = metadata["page"]
        scheme = metadata["scheme"]
        content_type = metadata["content_type"]
        chunk_index = metadata["chunk_index"]
        table_index = metadata.get("table_index")

        if table_index is not None:
            chunk_id = (
                f"{source}_page{page}_{content_type}_{scheme}"
                f"_table{table_index}_chunk{chunk_index}"
            )
        else:
            chunk_id = (
                f"{source}_page{page}_{content_type}_{scheme}"
                f"_chunk{chunk_index}"
            )

        ids.append(chunk_id)
        texts.append(text)
        metadatas.append(metadata.copy())

    return ids, texts, metadatas

def embeddings():
    chunk_docs = chunked_documents()

    ids, texts, metadatas = prepare_data(chunk_docs)

    embedding_model = get_embedding_model()

    vectors = embedding_model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    return vectors, ids, texts, metadatas