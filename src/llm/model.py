import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)
