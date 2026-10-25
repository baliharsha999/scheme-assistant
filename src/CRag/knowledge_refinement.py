from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import asyncio

from src.CRag.schemas import PassageEvaluation, RefinedDocuments


def split_into_passages(document: dict) -> list[dict]:
    """
    Split a retrieved document into smaller passages while
    preserving its source metadata.
    """
    text = document.get("content") or document.get("text", "")

    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )

    passages = splitter.split_text(text)

    refined_passages = []

    metadata = document.get("metadata") or {}

    source = document.get(
        "source",
        metadata.get("source", "Unknown")
    )

    page = document.get(
        "page",
        metadata.get("page")
    )

    section = document.get(
        "section",
        metadata.get("section")
    )

    for passage in passages:
        refined_passages.append({
            "content": passage,
            "source": source,
            "page": page,
            "section": section,
        })

    return refined_passages


def create_passage_evaluator(llm):
    structured_llm = llm.with_structured_output(
        PassageEvaluation,
        method="json_mode"
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You evaluate passages for a government welfare scheme assistant.

            Score relevance from 0.0 to 1.0.

            0.0-0.29: Irrelevant.
            0.30-0.69: Partially relevant or ambiguous.
            0.70-1.0: Strongly relevant and useful evidence.

            Evaluate the actual information, not just keyword overlap.
            Do not invent facts.

            Return valid JSON with exactly these fields:
            - score: a number between 0.0 and 1.0
            - reason: a concise explanation
            """
        ),
        (
            "human",
            """
            User query:
            {query}

            Passage:
            {content}
            """
        ),
    ])

    return prompt | structured_llm

async def evaluate_passage(evaluator, query: str, passage: dict) -> dict:
    result = await evaluator.ainvoke({
        "query": query,
        "content": passage["content"],
    })

    return {
        "passage": passage,
        "score": result.score,
        "reason": result.reason,
    }

# Limit concurrent passage evaluations within this process.
PASSAGE_EVALUATION_LIMIT = 5


async def refine_ambiguous_documents(
    llm,
    query: str,
    ambiguous_documents: list[dict],
    threshold: float = 0.70,
) -> list[RefinedDocuments]:

    if not ambiguous_documents:
        return []

    evaluator = create_passage_evaluator(llm)

    passages_to_evaluate = []

    for item in ambiguous_documents:

        document = item.get("document", item)

        if not isinstance(document, dict):
            if hasattr(document, "model_dump"):
                document = document.model_dump()
            else:
                continue

        passages = split_into_passages(document)
        passages_to_evaluate.extend(passages)

    if not passages_to_evaluate:
        return []

    semaphore = asyncio.Semaphore(
        PASSAGE_EVALUATION_LIMIT
    )

    async def evaluate_with_limit(passage):

        async with semaphore:
            return await evaluate_passage(
                evaluator=evaluator,
                query=query,
                passage=passage,
            )

    results = await asyncio.gather(
        *[
            evaluate_with_limit(passage)
            for passage in passages_to_evaluate
        ]
    )

    refined_evidence = []

    for result in results:

        if result["score"] >= threshold:

            passage = result["passage"]

            refined_document = RefinedDocuments(
                text=passage["content"],
                source=passage["source"],
                page=passage["page"],
                section=passage["section"],
                relevance_score=result["score"],
            )

            refined_evidence.append(refined_document)

    return refined_evidence
