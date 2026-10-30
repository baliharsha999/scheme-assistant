import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_PDF_DIR = DATA_DIR / "raw"
CHROMA_DB_DIR = DATA_DIR / "ChromaDB"
EVALS_DIR = PROJECT_ROOT / "evals"
DATASET_DIR = EVALS_DIR / "dataset"
GOLDEN_DATASET_FILE = DATASET_DIR / "golden_dataset.json"

RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
RETRIEVER_RESULTS_FILE = RESULTS_DIR / "retriever_results.json"
GENERATOR_RESULTS_FILE = RESULTS_DIR / "generator_results.json"
SUMMARY_RESULTS_FILE = RESULTS_DIR / "evaluation_summary.json"

CHANGES_DOC_FILE = PROJECT_ROOT / "evaluation_changes.txt"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Model configurations
# Production model used by existing application
PRODUCTION_MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

# Judge model used by DeepEval for evaluation metrics
# Defaults to EVAL_GROQ_MODEL if set, otherwise GROQ_MODEL, otherwise openai/gpt-oss-120b
EVAL_JUDGE_MODEL_NAME = os.getenv(
    "EVAL_GROQ_MODEL",
    os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
).strip()

# Evaluation hyperparameters
EVAL_TEMPERATURE = float(os.getenv("EVAL_TEMPERATURE", "0.0"))
EVAL_REQUEST_TIMEOUT = int(os.getenv("EVAL_REQUEST_TIMEOUT", "60"))
EVAL_MAX_RETRIES = int(os.getenv("EVAL_MAX_RETRIES", "3"))
EVAL_RATE_LIMIT_DELAY = float(os.getenv("EVAL_RATE_LIMIT_DELAY", "1.0"))

# Metric thresholds (0.0 to 1.0)
CONTEXTUAL_RECALL_THRESHOLD = float(os.getenv("CONTEXTUAL_RECALL_THRESHOLD", "0.5"))
CONTEXTUAL_PRECISION_THRESHOLD = float(os.getenv("CONTEXTUAL_PRECISION_THRESHOLD", "0.5"))
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.5"))
ANSWER_RELEVANCY_THRESHOLD = float(os.getenv("ANSWER_RELEVANCY_THRESHOLD", "0.5"))
ANSWER_CORRECTNESS_THRESHOLD = float(os.getenv("ANSWER_CORRECTNESS_THRESHOLD", "0.5"))
