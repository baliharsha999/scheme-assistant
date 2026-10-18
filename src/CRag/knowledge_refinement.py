from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.CRag.schemas import RetrievalEvalution

def split_into_passages(
    document: dict,
) -> list[dict]:
    """
    Split an ordinary Python dictionary's content into passages.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )

    passages = splitter.split_text(
        document["content"]
    )

    refined_passages = []

    for passage in passages:
        refined_passages.append({
            "content": passage,
            "source": document.get("source", "Unknown"),
            "page": document.get("page"),
            "section": document.get("section"),
        })

    return refined_passages