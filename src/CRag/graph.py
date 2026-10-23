from typing import TypedDict
from src.CRag.graph_nodes import (bm25_search_node,
                                  dense_search_node,
                                  rrf_node,
                                  rerank_node,
                                  evaluation_retrieval_node,
                                  knowledge_refine_node,
                                  evidence_evaluation_node,
                                  query_rewritten_node,
                                  web_search_node,
                                  generate_answer_node,
                                  abstain_node,
                                  assemble_evidence_node,
                                  evaluate_web_documents_node,
                                  web_assemble_evidence_node,
                                  refine_web_documents_node,
                                  web_evidence_evaluation_node)
from langgraph.graph import StateGraph, START, END
from src.CRag.graph_state import Cragstate

#routing function that is requierd to direct from ambigous to knowledge refinement
# and combine with relevant and add to evidence sufficient evaluator
def route_after_retrieval_evaluation(state: Cragstate):
    if state.get("ambiguous_documents"):
        return "knowledge_refine"

    return "assemble_evidence"
# for web documents
def route_after_web_retrieval_evaluation(state: Cragstate):
    if state.get("ambiguous_web_documents"):
        return "refine_web_documents"

    return "web_assemble_evidence"

#routing after the evidence sufficiency if evidence is sufficent then we will
#go the generate answer or pipeline that is to be followed after insufficient ecidence
def route_after_evidence_evalution(state:Cragstate):
    if state.get("evidence_sufficient"):
        return "generate_answer"
    return "query_rewritten"

#for web documents sufficiency
def web_evidence_evalution_route(state: Cragstate):
    if state.get("web_evidence_sufficient", False):
        return "generate_answer"

    return "abstain"

builder = StateGraph(Cragstate)

# adding all the nodes
builder.add_node("bm25_search",bm25_search_node)
builder.add_node("dense_search",dense_search_node)
builder.add_node("rrf",rrf_node)
builder.add_node("rerank",rerank_node)
builder.add_node("evaluation_retrieval",evaluation_retrieval_node)
builder.add_node("assemble_evidence", assemble_evidence_node)
builder.add_node("knowledge_refine",knowledge_refine_node)
builder.add_node("evidence_evaluation",evidence_evaluation_node)
builder.add_node("query_rewritten",query_rewritten_node)
builder.add_node("web_search",web_search_node)
builder.add_node("generate_answer",generate_answer_node)
builder.add_node("abstain",abstain_node)
builder.add_node("evaluate_web_documents",evaluate_web_documents_node)
builder.add_node("web_assemble_evidence",web_assemble_evidence_node)
builder.add_node("refine_web_documents",refine_web_documents_node)
builder.add_node(
    "web_evidence_evaluation",
    web_evidence_evaluation_node
)
# adding some of the edges
builder.add_edge(START,"bm25_search")
builder.add_edge(START,"dense_search")
builder.add_edge("dense_search","rrf")
builder.add_edge("bm25_search","rrf")
builder.add_edge("rrf","rerank")
builder.add_edge("rerank","evaluation_retrieval")
builder.add_conditional_edges(
    "evaluation_retrieval",
    route_after_retrieval_evaluation,
    {
        "knowledge_refine": "knowledge_refine",
        "assemble_evidence": "assemble_evidence"
    }
)
builder.add_edge("knowledge_refine","assemble_evidence")
builder.add_edge("assemble_evidence","evidence_evaluation")
builder.add_conditional_edges(
    "evidence_evaluation",
    route_after_evidence_evalution,
    {
        "generate_answer":"generate_answer",
        "query_rewritten":"query_rewritten"
    }
)
builder.add_edge("query_rewritten","web_search")
builder.add_edge("web_search", "evaluate_web_documents")
builder.add_conditional_edges(
    "evaluate_web_documents",
    route_after_web_retrieval_evaluation,
    {
        "refine_web_documents": "refine_web_documents",
        "web_assemble_evidence": "web_assemble_evidence"
    }
)
builder.add_edge("web_assemble_evidence","web_evidence_evaluation")
builder.add_edge("refine_web_documents","web_assemble_evidence")
builder.add_conditional_edges(
    "web_evidence_evaluation",
    web_evidence_evalution_route,
    {
        "generate_answer": "generate_answer",
        "abstain": "abstain"
    }
)

builder.add_edge("generate_answer", END)
builder.add_edge("abstain", END)