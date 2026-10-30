# 🏛️ Government Welfare Scheme Assistant (CRAG)

A Corrective RAG (CRAG) chatbot designed to help citizens understand and navigate Indian government welfare schemes with grounded, verified answers and exact source citations.

---

## 📖 Overview

When citizens ask questions about government schemes, hallucinated or inaccurate answers can cause confusion. This project implements a **Corrective Retrieval-Augmented Generation (CRAG)** pipeline:
1. **Hybrid Retrieval**: Combines sparse keyword search (**BM25**) with dense semantic search (**BGE-M3** in **ChromaDB**).
2. **Rank Fusion & Reranking**: Merges candidates using **Reciprocal Rank Fusion (RRF)** and scores them with **BGE-Reranker-v2-m3**.
3. **CRAG Quality Checks**:
   - **Document Relevance Grading**: Evaluates whether retrieved documents are relevant, ambiguous, or irrelevant.
   - **Knowledge Refinement**: Chunks and re-grades ambiguous passages to extract key information.
   - **Evidence Sufficiency Evaluation**: Checks if the assembled local evidence is sufficient to answer the question.
   - **Web Search Fallback**: If local evidence is insufficient, rewrites the query and retrieves real-time info via **Tavily Web Search**.
   - **Graceful Abstention**: If neither local documents nor web search provide sufficient facts, politely abstains instead of making things up.
4. **Citizen-Friendly Answer Synthesis**: Generates clear, concise bullet points citing exact document filenames and page numbers (e.g., `[Source: PM-KISAN.pdf, Page 12]`).

---

## 📑 Indexed Welfare Schemes

The vector database includes 2,300+ indexed chunks across 6 official operational guidelines:
- **Mission Saksham Anganwadi & Poshan 2.0**
- **PM-KISAN** (Pradhan Mantri Kisan Samman Nidhi)
- **PM-KMY** (Pradhan Mantri Kisan Maan Dhan Yojana)
- **PMAY-U 2.0** (Pradhan Mantri Awas Yojana - Urban)
- **PMAJAY** (Pradhan Mantri Anusuchit Jaati Abhyuday Yojana)
- **PMMVY** (Pradhan Mantri Matru Vandana Yojana)

---

## 🛠️ Architecture Workflow

```
                   USER QUERY
                       |
                       v
        +--------------+--------------+
        |                             |
        v                             v
   BM25 Search                   Dense Search
  (Keyword match)             (BGE-M3 embeddings)
        |                             |
        +--------------+--------------+
                       |
                       v
             Reciprocal Rank Fusion (RRF)
                       |
                       v
             Cross-Encoder Reranker
              (bge-reranker-v2-m3)
                       |
                       v
            Document-Wise Relevance
           (Relevant / Ambiguous / Discard)
                       |
        +--------------+--------------+
        |                             |
        v                             v
     Relevant                    Ambiguous
     (Keep)                      (Refine & Re-score)
        |                             |
        +--------------+--------------+
                       |
                       v
            Evidence Sufficiency Check
                       |
            +----------+----------+
            |                     |
            v                     v
       Sufficient            Insufficient
            |                     |
            |                     v
            |              Query Rewriting
            |                     |
            |                     v
            |             Tavily Web Search
            |                     |
            |                     v
            |             Web Relevance Check
            |                     |
            |                     v
            |             Web Sufficiency Check
            |                     |
            |             +-------+-------+
            |             |               |
            |             v               v
            |         Sufficient     Insufficient
            |             |               |
            +------+------+               v
                   |                   Abstain
                   v
             Generate Answer
            (with Citations)
                   |
                   v
             FINAL RESPONSE
```

---

## 📂 Project Structure

```text
scheme-assistant/
├── data/
│   ├── raw/                 # Original PDF scheme guidelines
│   └── ChromaDB/            # Persistent vector database (2,306 chunks)
├── src/
│   ├── Ingestion/           # PDF extraction, chunking, embeddings, vector store
│   │   ├── load_documents.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   ├── retrieval/           # BM25, dense retrieval, RRF, reranking
│   │   ├── bm25_retriever.py
│   │   ├── dense_retriever.py
│   │   └── hybrid_search.py
│   ├── CRag/                # LangGraph nodes, state, evaluators, web search
│   │   ├── graph.py         # StateGraph workflow definition
│   │   ├── graph_nodes.py   # Step execution nodes
│   │   ├── graph_state.py   # State dictionary schema
│   │   ├── schemas.py       # Pydantic schemas for LLM outputs
│   │   ├── retrieval_evaluator.py
│   │   ├── knowledge_refinement.py
│   │   ├── evidence_evaluator.py
│   │   ├── query_rewriter.py
│   │   └── web_search.py
│   └── llm/                 # Model init and FastAPI server
│       ├── model.py         # Groq LLM setup
│       └── app.py           # FastAPI endpoints & web chat UI
├── evals/                   # Evaluation suite (DeepEval & Groq)
│   ├── dataset/             # Golden evaluation dataset (JSON)
│   ├── config.py            # Evaluation settings and thresholds
│   ├── groq_model.py        # Groq wrapper for DeepEval
│   ├── evaluate_retriever.py
│   ├── evaluate_generator.py
│   ├── report_generator.py
│   └── run_evaluation.py    # Main evaluation runner CLI
├── evaluation_results/      # Generated evaluation reports and JSON logs
├── tests/                   # Pytest unit tests
│   ├── test_api.py
│   ├── test_retrieval.py
│   └── test_chroma.py
├── Dockerfile               # Container definition
├── docker-compose.yml       # Docker Compose setup
├── .dockerignore
├── requirements.txt         # Core dependencies
├── requirements_evals.txt   # Evaluation dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Groq API Key ([https://console.groq.com](https://console.groq.com))
- Tavily API Key ([https://tavily.com](https://tavily.com))

### 2. Clone and Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd scheme-assistant

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_evals.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

---

## 💻 Running the Application

### Start the FastAPI Server

```bash
uvicorn src.llm.app:app --host 0.0.0.0 --port 8000 --reload
```

Once started:
- **Interactive Web Chat UI**: Open your browser at [http://localhost:8000](http://localhost:8000)
- **API Documentation**: Open [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### Testing via API (cURL)

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Which ministry issued the guidelines for Mission Saksham Anganwadi and Poshan 2.0?"}'
```

**Response Example:**
```json
{
  "answer": "The guidelines for Mission Saksham Anganwadi and Poshan 2.0 were issued by the Ministry of Women and Child Development, Government of India. [Source: Mission Saksham Anganwadi and Poshan 2.0 scheme guidelines.pdf, Page 1]",
  "abstained": false,
  "abstention_reason": null
}
```

---

## 🧪 Running Tests & Evaluation

### Run Unit Tests (Pytest)

```bash
pytest tests/ -v
```

### Run Evaluation Suite (DeepEval + Groq)

To evaluate the hybrid retriever and generator components:

```bash
# Run evaluation on a sample of test cases
python -m evals.run_evaluation --sample-size 5

# Or evaluate only the retriever
python -m evals.run_evaluation --retriever --sample-size 5

# Or evaluate only the generator
python -m evals.run_evaluation --generator --sample-size 5

# Rebuild Markdown summary reports from existing results
python -m evals.run_evaluation --report-only
```

Evaluation results are saved into `evaluation_results/`:
- `summary_report.md` - Executive summary with Component Health Scorecard
- `retriever_report.md` - Contextual Recall, Precision, and Hit Rate@5 breakdown
- `generator_report.md` - Faithfulness, Relevancy, and Correctness breakdown

---

## 🐳 Docker Deployment

The application is fully containerized using Docker and Docker Compose.

### Option A: Using Docker Compose (Recommended)

Make sure your `.env` file exists with your `GROQ_API_KEY` and `TAVILY_API_KEY`, then run:

```bash
# Build and run the container
docker compose up --build

# Run in background (detached mode)
docker compose up -d

# Stop the container
docker compose down
```

### Option B: Using Plain Docker

```bash
# 1. Build the Docker image
docker build -t scheme-assistant .

# 2. Run the Docker container
docker run -d -p 8000:8000 --env-file .env --name scheme-assistant-app scheme-assistant

# 3. Check logs
docker logs -f scheme-assistant-app

# 4. Stop container
docker stop scheme-assistant-app
```

Then visit [http://localhost:8000](http://localhost:8000) in your browser.

---

## 📚 Student Learning Notes & Design Choices

1. **No Cache Complexity**: To keep the system simple and reliable during learning, no complex caching mechanisms (such as Redis or module-level memoization dicts) are used. Each retrieval and inference step executes directly and transparently.
2. **Component Separation**: Evaluating the retriever independently from the generator prevents "blaming the LLM" when the retriever missed the relevant document.
3. **Corrective Flow**: Instead of blindly generating answers from whatever chunks are retrieved, the LLM actively checks if the retrieved text is actually sufficient. If not, it uses web search fallback or gracefully abstains.
