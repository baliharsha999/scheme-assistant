from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.Ingestion.load_documents import document_loaders

def split_documents(documents):
    chunked_documents = []

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200
    )


    for document in documents:
        metadata = document["metadata"]
        text = document["text"]

        chunks = text_splitter.split_text(text)

        for chunk_index , chunk in enumerate(chunks):
            chunked_documents.append({
                "text":chunk,
                "metadata":{
                    **metadata,
                    "chunk_index":chunk_index
                }
            })
    return chunked_documents

chunked_documents = split_documents(document_loaders)