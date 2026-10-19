from src.CRag.graph_state import Cragstate
from src.retrival.bm25_retriever import bm25_search
from src.retrival.dense_retriever import dense_search
from src.retrival.hybrid_search import (reciprocal_rank_fusion,reranker)
from src.CRag.retrieval_evaluator import evaluate_retrieval
from src.llm.model import llm
from src.CRag.knowledge_refinement import refine_ambiguous_documents

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
# evidence sufficiency evalution
def evidence_evalution_node(state:Cragstate):
    query = state["query"]
    