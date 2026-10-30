from src.CRag.graph_state import Cragstate
from src.retrieval.bm25_retriever import bm25_search
from src.retrieval.dense_retriever import dense_search
from src.retrieval.hybrid_search import (reciprocal_rank_fusion,reranker)
from src.CRag.retrieval_evaluator import evaluate_retrieval
from src.llm.model import llm
from src.CRag.knowledge_refinement import refine_ambiguous_documents
from src.CRag.evidence_evaluator import evaluate_evidence
from src.CRag.query_rewriter import rewrite_query
from src.CRag.web_search import search_web
import asyncio
# bm25 node ----> 1
async def bm25_search_node(state: Cragstate):
    query = state["query"]

    bm25_results = await asyncio.to_thread(bm25_search, query)

    return {
        "bm25_results": bm25_results
    }
# dense search node ---->1
async def dense_search_node(state: Cragstate):
    query = state["query"]

    dense_results = await asyncio.to_thread(
        dense_search,
        query
    )

    return {
        "dense_results": dense_results
    }
# rrf node ------> 2
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
# reranking node ------> 3
async def rerank_node(state: Cragstate):
    query = state["query"]
    fused_results = state["rrf_results"]

    reranked_results = await asyncio.to_thread(
        reranker,
        query,
        fused_results
    )

    return {
        "reranked_results": reranked_results
    }
# if we connect all these then we will get the top k documents ------->4
async def evaluation_retrieval_node(state:Cragstate):
    query = state["query"]
    documents = state["reranked_results"]
    evaluation = await evaluate_retrieval(llm,query,documents)

    return {
        "relevant_documents": evaluation["relevant"],
        "ambiguous_documents": evaluation["ambiguous"],
        "irrelevant_documents": evaluation["irrelevant"]
    }
# knowledge refinement node  ------> 5
async def knowledge_refine_node(state:Cragstate):
    query = state["query"]
    ambiguous_documents = state["ambiguous_documents"]

    refined_documents = await refine_ambiguous_documents(llm,query,ambiguous_documents,0.7)

    return {
        "refined_documents" : refined_documents
    }
# Evidence sufficiency evaluation ------> 7
def evidence_evaluation_node(state: Cragstate):
    query = state["query"]
    documents = state["evidence_documents"]

    evidence_evaluation_results = evaluate_evidence(llm,query,documents)

    return {
        "evidence_reason": evidence_evaluation_results.reason,
        "evidence_sufficient": evidence_evaluation_results.sufficient,
        "missing_information": evidence_evaluation_results.missing_information
    }
# query rewritting node for web search -------->8
def query_rewritten_node(state : Cragstate):
    query = state["query"]
    missing_information = state["missing_information"]

    rewritten_query = rewrite_query(llm,query,missing_information)
    return {
        "rewritten_query":rewritten_query.rewritten_query
    }
# tavily web search node  -------->9
async def web_search_node(state: Cragstate):

    rewritten_query = state.get("rewritten_query", "").strip()

    if not rewritten_query:
        rewritten_query = state["query"].strip()

    if not rewritten_query:
        raise ValueError("Both original and rewritten queries are empty.")

    print("Rewritten query:", repr(rewritten_query))

    web_search_results = await asyncio.to_thread(search_web, rewritten_query)

    print("TAVILY RESULTS:", len(web_search_results))
    print("FIRST WEB RESULT:", web_search_results[:1])

    return {
        "rewritten_query": rewritten_query,
        "web_documents": web_search_results,
    }
# generating answer node ----->14
async def generate_answer_node(state: Cragstate):
    query = state["query"]

    if state["evidence_sufficient"]:
        documents = state.get("evidence_documents", [])

    elif state["web_evidence_sufficient"]:
        documents = state.get("web_evidence_documents", [])

    else:
        documents = []

    # Construct context with source information
    context_parts = []

    for doc in documents:
        text = doc.get("content") or doc.get("text", "")
        metadata = doc.get("metadata") or {}

        source = metadata.get(
            "source",
            doc.get("source", "Unknown source")
        )

        page = metadata.get(
            "page",
            doc.get("page", "Unknown")
        )

        if text.strip():
            context_parts.append(
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content: {text}"
            )

    context = "\n\n".join(context_parts)

    prompt = f"""
    You are a helpful and accurate government welfare scheme assistant.

    Your task is to answer the user's question using only the evidence provided.

    Instructions:

    1. Answer the user's question directly in clear, simple language.

    2. Synthesize information from the relevant passages into a coherent answer.
       Do not simply copy entire passages from the documents.

    3. Include only information relevant to the user's question.

    4. Use bullet points for eligibility criteria, benefits, required documents,
       and application procedures when appropriate.

    5. Do not invent facts, eligibility conditions, benefits, dates, or procedures.

    6. If a requested detail is missing from the evidence, explicitly say that
       the available documents do not specify it.

    7. Cite sources using the document filename and page number provided
       in the context, for example: [Source: scheme_guidelines.pdf, Page 12].

    8. Do not invent or guess source names or page numbers.

    9. Do not mention internal technical terms such as chunks, embeddings,
       retrievers, or RAG.

    User question:

    {query}

    Evidence:

    {context}

    Provide a clear, concise answer for an ordinary citizen.
    """
    response = await llm.ainvoke(prompt)

    return {
        "answer": response.content,
        "abstained": False
    }
# if the sufficient documents are not found -----> 14
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

# assemble node to assemble ambigoues and relevant
# Evidence assembly node ---------->6
def assemble_evidence_node(state: Cragstate):
    relevant_documents = state.get("relevant_documents", [])
    refined_documents = state.get("refined_documents", [])
    evidence_documents = []

    for item in relevant_documents:
        document = item.get("document", item)
        evidence_documents.append(document)

    for document in refined_documents:

        # Converting Pydantic objects into dictionaries
        if hasattr(document, "model_dump"):
            document = document.model_dump()

        # Normalize the passage text
        if "text" in document and "content" not in document:
            document = {
                **document,
                "content": document["text"]
            }

        evidence_documents.append(document)

    return {
        "evidence_documents": evidence_documents
    }
# the refined documents are then assemebled -------->12
def web_assemble_evidence_node(state: Cragstate):
    relevant_documents = state.get("relevant_web_documents", [])
    refined_documents = state.get("refined_web_docs", [])
    evidence_documents = []

    for item in relevant_documents:
        document = item.get("document", item)
        evidence_documents.append(document)

    for document in refined_documents:

        # Converting Pydantic objects into dictionaries
        if hasattr(document, "model_dump"):
            document = document.model_dump()

        # Normalize the passage text
        if "text" in document and "content" not in document:
            document = {
                **document,
                "content": document["text"]
            }

        evidence_documents.append(document)

    return {
        "web_evidence_documents": evidence_documents
    }

# the documents that we are getting from web are evaluated ---->10
async def evaluate_web_documents_node(state: Cragstate):
    query = state["rewritten_query"]
    documents = state.get("web_documents", [])

    evaluation = await evaluate_retrieval(
        llm,
        query,
        documents
    )
    print("WEB DOCUMENTS RECEIVED:", len(documents))
    print("RELEVANT WEB DOCUMENTS:", len(evaluation["relevant"]))
    print("AMBIGUOUS WEB DOCUMENTS:", len(evaluation["ambiguous"]))
    print("IRRELEVANT WEB DOCUMENTS:", len(evaluation["irrelevant"]))

    return {
        "relevant_web_documents": evaluation["relevant"],
        "ambiguous_web_documents": evaluation["ambiguous"],
        "irrelevant_web_documents": evaluation["irrelevant"]

    }

# the assembeled web documents are then evaluated for sufficiency -------> 13
def web_evidence_evaluation_node(state: Cragstate):
    query = state["query"]
    documents = state.get("web_evidence_documents", [])

    evaluation = evaluate_evidence(
        llm,
        query,
        documents
    )
    print("FINAL WEB EVIDENCE COUNT:", len(documents))
    print("WEB EVIDENCE SUFFICIENT:", evaluation.sufficient)
    print("WEB EVIDENCE REASON:", evaluation.reason)
    print("MISSING INFORMATION:", evaluation.missing_information)

    return {
        "web_evidence_sufficient": evaluation.sufficient,
        "web_evidence_reason": evaluation.reason,
        "web_missing_information": evaluation.missing_information
    }
# the documents from web are then refined -------> 11
async def refine_web_documents_node(state: Cragstate):
    query = state["rewritten_query"]
    ambiguous_documents = state.get("ambiguous_web_documents", [])

    print("AMBIGUOUS DOCUMENTS ENTERING REFINEMENT:", len(ambiguous_documents))

    refined_documents = await refine_ambiguous_documents(
        llm=llm,
        query=query,
        ambiguous_documents=ambiguous_documents,
        threshold=0.7
    )

    print("DOCUMENTS AFTER REFINEMENT:", len(refined_documents))

    return {
        "refined_web_docs": refined_documents
    }