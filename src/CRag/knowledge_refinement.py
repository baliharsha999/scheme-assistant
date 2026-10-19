from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.CRag.schemas import RetrievalEvalution
from langchain_core.prompts import ChatPromptTemplate
from src.CRag.schemas import RefinedDocuments

def split_into_passages(document: dict) -> list[dict]:
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

def create_passage_evaluator(llm):
    """
    create an evaluator that assigns scores one passage at a time
    """
    structured_llm = llm.with_structured_output(
        RefinedDocuments
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are evaluating passages for a government
            welfare scheme assistant.

            Evaluate how relevant the passage is to the user's query.

            Assign a score between 0.0 and 1.0.

            Scoring guidelines:

            0.0 - 0.29:
            The passage is irrelevant to the query.

            0.30 - 0.69:
            The passage is partially relevant or ambiguous.

            0.70 - 1.0:
            The passage is strongly relevant and provides
            useful supporting evidence.

            Evaluate the actual information in the passage.
            Do not reward keyword overlap alone.

            Do not invent facts that are not present.

            Return a score and a concise explanation.
            """           
        ),
        (
            "human",
            """
            User Query:
            {query}

            Passage:
            {content}
            """
        ),
    ])

    return prompt|structured_llm

def evaluate_passage(evaluator, query: str, passage: dict) -> dict:
    """
    Evaluate one passage and preserve its source metadata.
    """

    result: RetrievalEvalution = evaluator.invoke({
        "query": query,
        "content": passage["content"],
    })

    return {
        "passage": passage,
        "score": result.score,
        "reason": result.reason,
    }

def refine_ambiguous_documents(
    llm,
    query: str,
    ambiguous_documents: list[dict],
    threshold: float = 0.70,
) -> list[RefinedDocuments]:
    """
    Split ambiguous documents into passages, evaluate each
    passage, and get sufficiently relevant evidence.
    """

    evaluator = create_passage_evaluator(llm)

    refined_evidence = []

    for item in ambiguous_documents:

        # Extract the original retrieved document.
        document = item["document"]

        # Split it into smaller passages.
        passages = split_into_passages(document)

        # Evaluate every passage independently.
        for passage in passages:

            result = evaluate_passage(
                evaluator=evaluator,
                query=query,
                passage=passage,
            )

            # Retain passages that pass the threshold.
            if result["score"] >= threshold:

                refined_document = RefinedDocuments(
                    text=passage["content"],
                    source=passage["source"],
                    page=passage["page"],
                    section=passage["section"],
                    relevance_score=result["score"],
                )

                refined_evidence.append(refined_document)

    return refined_evidence