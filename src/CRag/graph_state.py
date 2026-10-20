from typing import TypedDict


class Cragstate(TypedDict, total=False):
    query: str # done
    retrieved_documents: list[dict]# done
    relevant_documents: list[dict]# done
    ambiguous_documents: list[dict]# done
    irrelevant_documents: list[dict]# done
    refined_documents: list[dict]# done
    evidence_documents: list[dict]#done
    evidence_sufficient: bool#done
    evidence_reason: str#done
    missing_information: list[str]#done
    rewritten_query: str#done
    web_documents: list[dict]#done
    retry_count: int
    max_retries: int
    answer: str
    citations: list[dict]
    abstained: bool
    abstention_reason: str
    bm25_results : list[dict]# done
    dense_results : list[dict]# done
    reranked_results : list[dict]# done
    relevant_web_documents : list[dict]
    ambiguous_web_documents : list[dict]
    irrelevant_web_documents : list[dict]
    web_evidence_sufficient: bool
    web_evidence_reason: str
    web_missing_information: list[str]
    web_evidence_documents: list[dict] 
    refined_web_docs : list[dict]