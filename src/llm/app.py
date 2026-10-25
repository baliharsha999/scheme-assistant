import traceback
import asyncio 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.CRag.graph import builder

app = FastAPI()

graph = builder.compile()


class QueryRequest(BaseModel):
    query: str


@app.post("/")
def home():
    return {
        "message": "This is the home endpoint of the Government Scheme Chatbot API"
    }


@app.post("/chat")
async def chat(request: QueryRequest):
    print("\n========== CHAT ENDPOINT CALLED ==========")
    print("User query:", request.query)

    try:
        print("Starting CRAG graph execution...")

        result = await graph.ainvoke({
            "query": request.query
        })

        print("CRAG GRAPH EXECUTION COMPLETED")
        print("Answer:", result.get("answer"))
        print("Abstained:", result.get("abstained"))

        return {
            "answer": result.get("answer", ""),
            "abstained": result.get("abstained", False),
            "abstention_reason": result.get("abstention_reason")
        }

    except Exception as e:
        print("\n========== CRAG GRAPH ERROR ==========\n")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
