from langchain_core.prompts import ChatPromptTemplate
from src.CRag.schemas import EvidenceEvalution

def create_evidence_evaluator(llm):
    """
    create an evaluator that determines wheather the combined docs are
    sufficent for answering the users query
    """
    structured_llm = llm.with_structured_output(
        EvidenceEvalution
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an evidence sufficiency evaluator for a
                grounded government welfare scheme assistant.

                Your task is to determine whether the provided
                evidence is sufficient to answer the user's query
                accurately and completely.

                IMPORTANT DISTINCTION:

                Relevant evidence is not necessarily sufficient
                evidence.

                Evaluate the combined evidence, not each passage
                independently.

                Follow these rules:

                1. Identify the main information requested by
                   the user.

                2. Check whether the evidence supports each
                   important part of the question.

                3. Identify missing information, if any.

                4. Do not assume facts that are not supported
                   by the evidence.

                5. Do not mark the evidence sufficient merely
                   because it discusses the correct scheme.

                6. For questions about government welfare schemes,
                   pay attention to eligibility requirements,
                   benefits, application procedures, deadlines,
                   locations, and financial amounts when relevant
                   to the query.

                7. If important requested information is missing,
                   set sufficient to false and explain what
                   information is missing.

                8. If the evidence adequately supports the
                   requested answer, set sufficient to true.

                Return:
                - sufficient
                - reason
                - missing_information

                If the evidence is sufficient, return an empty
                missing_information list.
                """
            ),
            (
                "human",
                """
                User Query:
                {query}

                Available Evidence:
                {evidence}
                """
            ),
        ]
    )
    return prompt | structured_llm

def evaluate_evidence(llm,query: str,documents : list ) -> EvidenceEvalution:
    """
    evaluate whether the given evidence are sufficient
    """
    if not documents:
        return EvidenceEvalution(
            sufficient = False,
            reason = "no evidence is available to answer the query",
            missing_information = [
                "reliable information relevant to the users query"
            ]
        )
    evidence_parts = []

    for index , document in enumerate(documents,start = 1):
        # Support ordinary Python dictionaries.
        text = document.get(
            "text",
            document.get("content", ""),
        )

        source = document.get("source", "Unknown")
        page = document.get("page")
        section = document.get("section")

        if not text or not text.strip():
            continue;

        evidence_parts.append(
            f"""
            Evidence {index}:
            Source : {source}
            Page: {page if page is not None else "Unknown"}
            Section: {section if section else "Unknown"}
            Content: {text}
            """
        )
        
    if not evidence_parts:
        return EvidenceEvalution(
            sufficient=False,
            reason="No usable evidence text is available.",
            missing_information=[
                "Non-empty evidence relevant to the user's query."
            ],
        )

    evidence_text = "\n\n".join(evidence_parts)

    evaluator = create_evidence_evaluator(llm)

    result: EvidenceEvalution = evaluator.invoke(
        {
            "query": query,
            "evidence": evidence_text,
        }
    )

    return result