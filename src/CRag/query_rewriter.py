from src.CRag.schemas import (RewrittenQuery)
from langchain_core.prompts import ChatPromptTemplate

def create_query_rewriter(llm):
    """
    create an llm that rerwrites the queries based on search engine for websearch
    """
    structured_llm = llm.with_structured_output(
        RewrittenQuery
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a search-query optimization expert for a
                citation-grounded government welfare scheme assistant.

                Your task is to rewrite the user's query so that a
                search engine can retrieve reliable information
                addressing the user's actual information need.

                Follow these rules:

                1. Preserve the user's original intent.

                2. Use the missing-information details to focus
                   the search on what is not yet established.

                3. Preserve important scheme names, government
                   departments, locations, eligibility conditions,
                   financial amounts, dates, and other relevant details.

                4. If the query concerns a government welfare scheme,
                   prefer wording that helps locate official government
                   notifications, scheme guidelines, department pages,
                   or other authoritative sources.

                5. Do not invent scheme names, locations, dates,
                   eligibility requirements, or other facts.

                6. Do not make the query unnecessarily long.

                7. Produce one clear, standalone search query.

                8. Do not answer the user's question. Only rewrite
                   the query for searching.

                Return:
                - rewritten_query
                - reason
                """
            ),
            (
                "human",
                """
                Original user query:
                {query}

                Missing information:
                {missing_information}

                Rewrite the query to retrieve reliable evidence
                addressing the missing information while preserving
                the user's original intent.
                """
            ),
        ]
    )

    return prompt | structured_llm

def rewrite_query(llm,query: str,missing_information: list[str],) -> RewrittenQuery:
    """
    Rewrite the original query using the missing information
    identified by the evidence sufficiency evaluator.
    """

    rewriter = create_query_rewriter(llm)

    result: RewrittenQuery = rewriter.invoke(
        {
            "query": query,
            "missing_information": "\n".join(
                f"{item}" for item in missing_information
            ),
        }
    )

    return result