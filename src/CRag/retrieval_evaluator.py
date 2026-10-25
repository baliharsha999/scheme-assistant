from typing import Literal
import asyncio
from langchain_core.prompts import ChatPromptTemplate

from src.CRag.schemas import RetrievalEvalution


LOWER_BOUNDARY = 0.30
UPPER_BOUNDARY = 0.70


RetrievalClassification = Literal[
    "correct",
    "ambiguous",
    "incorrect",
]


def classify_retrieval(
    score: float,
) -> RetrievalClassification:
    """
    Classify an individual document based on its relevance score.
    """

    if score >= UPPER_BOUNDARY:
        return "correct"

    if score >= LOWER_BOUNDARY:
        return "ambiguous"

    return "incorrect"


def create_retrieval_evaluator(llm):
    """
    Create an evaluator that scores one document at a time.
    """

    structured_llm = llm.with_structured_output(
        RetrievalEvalution
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a document relevance evaluator for a
                government welfare scheme assistant.

                Evaluate how well ONE document supports answering
                the user's query.

                Assign a relevance score between 0.0 and 1.0.

                Scoring guidance:

                0.00 - 0.29:
                The document is irrelevant or provides almost
                no useful evidence.

                0.30 - 0.69:
                The document is partially relevant, ambiguous,
                or incomplete.

                0.70 - 1.00:
                The document is strongly relevant.

                Consider:

                1. Does the document address the actual question?
                2. Does it contain useful supporting information?
                3. Does it concern the correct welfare scheme?
                4. Does it provide evidence relevant to the
                   requested information?
                5. Is it merely matching keywords without
                   actually supporting the answer?

                Evaluate only the provided document.
                Do not assume facts that are not present.

                Return a score and a concise explanation.
                """
            ),
            (
                "human",
                """
                User Query:
                {query}

                Document Content:
                {content}

                Source:
                {source}

                Page:
                {page}
                """
            ),
        ]
    )

    return prompt | structured_llm


async def evaluate_document(
    evaluator,
    query: str,
    document: dict,
) -> dict:

    result: RetrievalEvalution = await evaluator.ainvoke(
        {
            "query": query,
            "content": document.get(
                "content",
                document.get("text", "")
            ),
            "source": document.get("source", "Unknown"),
            "page": document.get("page", "Unknown"),
        }
    )

    classification = classify_retrieval(result.score)

    return {
        "document": document,
        "score": result.score,
        "reason": result.reason,
        "classification": classification,
    }

async def evaluate_retrieval(
    llm,
    query: str,
    documents: list[dict],
) -> dict:

    if not documents:
        return {
            "results": [],
            "relevant": [],
            "ambiguous": [],
            "irrelevant": [],
            "average_score": 0.0,
        }

    evaluator = create_retrieval_evaluator(llm)

    results = await asyncio.gather(
        *[
            evaluate_document(
                evaluator=evaluator,
                query=query,
                document=document,
            )
            for document in documents
        ]
    )

    relevant = []
    ambiguous = []
    irrelevant = []

    for result in results:

        if result["classification"] == "correct":
            relevant.append(result)

        elif result["classification"] == "ambiguous":
            ambiguous.append(result)

        else:
            irrelevant.append(result)

    average_score = (
        sum(result["score"] for result in results)
        / len(results)
    )

    return {
        "results": results,
        "relevant": relevant,
        "ambiguous": ambiguous,
        "irrelevant": irrelevant,
        "average_score": average_score,
    }