"""
evals/experiments/improved_chunking.py

Implements improved chunking strategy for welfare scheme documents:
1. Context-Enriched Metadata Injection:
   Prepends every chunk with human-readable and model-parseable metadata:
   "[Scheme: {scheme} | Document: {source} | Page: {page}]\n"
   This allows both BM25 keyword search and dense embedding retrieval to correctly
   associate isolated paragraphs with their originating scheme.
2. Semantic Sizing (900 chunk size, 150 overlap):
   Replaces the previous 500-char chunk size with 900 chars, ensuring complex eligibility
   criteria, tabular rows, and application steps are not fragmented mid-condition.
"""

from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.Ingestion.load_documents import load_documents


def build_improved_chunks(
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Loads raw documents from PDFs and splits them with context-enriched headers.
    Returns (ids, texts, metadatas).
    """
    raw_documents = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    ids = []
    texts = []
    metadatas = []

    for doc in raw_documents:
        metadata = doc["metadata"]
        raw_text = doc["text"]

        source = metadata.get("source", "unknown")
        page = metadata.get("page", 0)
        scheme = metadata.get("scheme", "unknown")
        content_type = metadata.get("content_type", "text")
        table_index = metadata.get("table_index")

        chunks = text_splitter.split_text(raw_text)

        for chunk_index, chunk in enumerate(chunks):
            # Header injection for context enrichment
            header = f"[Scheme: {scheme} | Document: {source} | Page: {page}]\n"
            enriched_text = f"{header}{chunk}"

            if table_index is not None:
                chunk_id = (
                    f"{source}_page{page}_{content_type}_{scheme}"
                    f"_table{table_index}_chunk{chunk_index}_v2"
                )
            else:
                chunk_id = (
                    f"{source}_page{page}_{content_type}_{scheme}"
                    f"_chunk{chunk_index}_v2"
                )

            chunk_meta = {
                **metadata,
                "chunk_index": chunk_index,
                "chunk_id": chunk_id,
                "strategy": "context_enriched_900_150",
            }

            ids.append(chunk_id)
            texts.append(enriched_text)
            metadatas.append(chunk_meta)

    return ids, texts, metadatas


if __name__ == "__main__":
    ids, texts, metas = build_improved_chunks()
    print(f"Generated {len(ids)} improved chunks.")
    if texts:
        print("Sample Chunk 0:\n" + texts[0][:300])
