from src.CRag.graph import builder

graph = builder.compile()

result = graph.invoke({
    "query": "What are the eligibility criteria for a scholarship scheme?"
})

print(result)