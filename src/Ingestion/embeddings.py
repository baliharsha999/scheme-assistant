from sentence_transformers import SentenceTransformer
from chunking import chunked_documents


def get_embedding_model():
    return SentenceTransformer("BAAI/bge-m3")

def prepare_data(chunked_documents):
    ids = []
    texts = []
    metadatas = []

    for document in chunked_documents:
        text = document["text"]
        metadata = document["metadata"]

        source = metadata["source"]
        page = metadata["page"]
        scheme = metadata["scheme"]
        content_type = metadata["content_type"]
        chunk_index = metadata["chunk_index"]
        table_index = metadata.get("table_index")
        scheme = metadata["scheme"]

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

        # Preserve ALL original metadata
        metadatas.append(metadata.copy())

    return ids, texts, metadatas

ids, texts,metadatas = prepare_data(chunked_documents)

embedding_model = get_embedding_model()

embeddings = embedding_model.encode(
    texts,
    batch_size=16,
    normalize_embeddings=True,
    show_progress_bar=True
).tolist()
print("all good")