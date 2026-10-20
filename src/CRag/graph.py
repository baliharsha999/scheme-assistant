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
                                  abstain_node)
from langgraph import StateGraph ,START,END

class graph_state(TypedDict):
    result : str
    query : str

builder = StateGraph(graph_state)

# adding all the nodes
builder.add_node("bm25_search",bm25_search_node)
builder.add_node("dense_search",dense_search_node)
builder.add_node("rrf",rrf_node)
builder.add_node("rerank",rerank_node)
builder.add_node("evaluation_retrieval",evaluation_retrieval_node)
builder.add_node("knowledge_refine",knowledge_refine_node)
builder.add_node("evidence_evaluation",evidence_evaluation_node)
builder.add_node("query_rewritten",query_rewritten_node)
builder.add_node("web_search",web_search_node)
builder.add_node("generate_answer",generate_answer_node)
builder.add_node("abstain",abstain_node)

# adding some of the edges
builder.add_edge(START,"bm25_search")
builder.add_edge(START,"dense_search")
builder.add_edge("dense_search","rrf")
builder.add_edge("bm25_search","rrf")
builder.add_edge("rrf","rerank")
builder.add_edge("rerank","evaluation_retrieval")