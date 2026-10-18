def search_web(rewritten_query: str,tavily_client,) -> list[dict]:
    """
    Search the web using Tavily and return normalized results
    each result is represented as a dictionary so that it can be compatible with the evidence
    sufficiency pipeline
    """

    if not rewritten_query.strip():
        raise ValueError("the rewritten query cannot be empty")

    response = tavily_client.search(
        query = rewritten_query,
        max_results = 5,
        search_depth = "advanced",
        include_answer = False
    )
    normalized_results = []

    for result in response.get("results",[]):
        content = result.get("content","").strip()
        source = result.get("url","").strip()
        if not content or not source:
            continue

        normalized_results.append({
            "content": content,
            "source": source,
            "page": None,
            "section": None,
            "title": result.get("title", ""),
            "source_type": "web",
        })

    return normalized_results
    