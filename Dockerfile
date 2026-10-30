# ==========================================
# Dockerfile for Government Scheme Assistant
# Built for simple, student-friendly deployment
# ==========================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory inside the container
WORKDIR /app

# Install essential system dependencies (curl for health check, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (for faster layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command starts FastAPI (can be overridden by docker-compose)
CMD ["uvicorn", "src.llm.app:app", "--host", "0.0.0.0", "--port", "8000"]
