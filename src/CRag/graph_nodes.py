from src.CRag.graph_state import Cragstate
from src.retrival.bm25_retriever import bm25_search
from src.retrival.dense_retriever import dense_search
from src.retrival.hybrid_search import (reciprocal_rank_fusion,reranker)
from src.CRag.retrieval_evaluator import evaluate_retrieval
from src.llm.model import llm
from src.CRag.knowledge_refinement import refine_ambiguous_documents
from src.CRag.evidence_evaluator import evaluate_evidence
from src.CRag.query_rewriter import rewrite_query
from src.CRag.web_search import search_web
# bm25 node
def bm25_search_node(state: Cragstate):
    query = state["query"]

    bm25_results = bm25_search(query)

    return {
        "bm25_results": bm25_results
    }
# dense search node
def dense_search_node(state:Cragstate):
    query = state["query"]

    dense_results = dense_search(query)

    return {
        "dense_results" : dense_results
    }
# rrf node
def rrf_node(state: Cragstate):
    bm25_results = state["bm25_results"]
    dense_results = state["dense_results"]

    result_lists = [
        bm25_results,
        dense_results
    ]

    fused_results = reciprocal_rank_fusion(result_lists)

    return {
        "rrf_results": fused_results
    }
# reranking node
def rerank_node(state: Cragstate):
    query = state["query"]
    fused_results = state["rrf_results"]
    reranked_results = reranker(query, fused_results)
    return {
        "reranked_results": reranked_results
    }
# if we connect all these then we will get the top k documents
def evaluation_retrieval_node(state:Cragstate):
    query = state["query"]
    documents = state["reranked_results"]
    evaluation = evaluate_retrieval(llm,query,documents)

    return {
        "relevant_documents": evaluation["relevant"],
        "ambiguous_documents": evaluation["ambiguous"],
        "irrelevant_documents": evaluation["irrelevant"]
    }
# knowledge refinement node
def knowledge_refine_node(state:Cragstate):
    query = state["query"]
    ambiguous_documents = state["ambiguous_documents"]

    refined_documents = refine_ambiguous_documents(llm,query,ambiguous_documents,0.7)

    return {
        "refinef_documents" : refined_documents
    }
# Evidence sufficiency evaluation
def evidence_evaluation_node(state: Cragstate):
    query = state["query"]
    documents = state["evidence_documents"]

    evidence_evaluation_results = evaluate_evidence(llm,query,documents)

    return {
        "evidence_reason": evidence_evaluation_results.reason,
        "evidence_sufficient": evidence_evaluation_results.sufficient,
        "missing_information": evidence_evaluation_results.missing_information
    }
# query rewritting node for web search
def query_rewritten_node(state : Cragstate):
    query = state["query"]
    missing_information = state["missing_information"]

    rewritten_query = rewrite_query(llm,query,missing_information)
    return {
        "rewritten_query":rewritten_query.rewritten_query
    }
# tavily web search node
def web_search_node(state:Cragstate):
    rewritten_query = state["rewritten_query"]
    web_search_results = search_web(rewritten_query)

    return {
        "web_documents" : web_search_results
    }
# generating answer node
def generate_answer_node(state: Cragstate):
    query = state["query"]
    documents = state["evidence_documents"]

    # Construct a context from the evidence.
    context = "\n\n".join(
        doc.get("content", doc.get("text", ""))
        for doc in documents
    )

    prompt = f"""
    You are a government welfare scheme assistant.

    Answer the user's question using only the provided context.
    Do not invent facts.
    If a claim is unsupported by the context, do not include it.
    Cite the provided sources where appropriate.

    Question:
    {query}

    Context:
    {context}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "abstained": False
    }
# if the sufficient documents are not found 
def abstain_node(state: Cragstate):
    reason = state.get(
        "evidence_reason",
        "The available evidence is insufficient."
    )

    return {
        "answer": (
            "I'm sorry, but I couldn't find sufficient reliable "
            "information to answer your question accurately. "
            "Please check the relevant official government website "
            "or contact the responsible department."
        ),
        "citations": [],
        "abstained": True,
        "abstention_reason": reason
    }