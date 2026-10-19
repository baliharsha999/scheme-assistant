from typing import TypedDict


class Cragstate(TypedDict, total=False):
    query: str # done
    retrieved_documents: list[dict]# done
    relevant_documents: list[dict]# done
    ambiguous_documents: list[dict]# done
    irrelevant_documents: list[dict]# done
    refined_documents: list[dict]# done
    evidence_documents: list[dict]
    evidence_sufficient: bool
    evidence_reason: str
    missing_information: list[str]
    rewritten_query: str
    web_documents: list[dict]
    retry_count: int
    max_retries: int
    answer: str
    citations: list[dict]
    abstained: bool
    abstention_reason: str
    bm25_results = list[dict]# done
    dense_results = list[dict]# done
    reranked_results = list[dict]# done