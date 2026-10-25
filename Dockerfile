FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir --default-timeout=300 --retries=10 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.llm.app:app", "--host", "0.0.0.0", "--port", "8000"]