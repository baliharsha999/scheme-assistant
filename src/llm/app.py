from src.CRag.graph import builder
from fastapi import FastAPI
from pydantic import BaseModel

class State(BaseModel,total = False):
    query : str
    result : list[dict]
    answer : str

app = FastAPI()



@app.post("/chat")
def rag_bot(state:State):
    query = State[query]

    result = builder.invoke({
        "query":query
    })

    return {
        "answer":result["answer"]
    }