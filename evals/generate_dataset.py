import os
import json
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_groq import ChatGroq


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_DIRECTORY = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIRECTORY = PROJECT_ROOT / "evals" / "dataset"
OUTPUT_FILE = OUTPUT_DIRECTORY / "golden_dataset.json"

load_dotenv(PROJECT_ROOT / ".env")

# Set this in your .env to a model currently available to you.
MODEL_NAME = os.getenv("GROQ_MODEL", "").strip()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Start small. Increase after verifying the output.
MAX_CHUNKS = 30
QUESTIONS_PER_CHUNK = 2


# =========================================================
# VALIDATION
# =========================================================

def validate_configuration():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY is missing from your project .env file."
        )

    if not MODEL_NAME:
        raise ValueError(
            "GROQ_MODEL is missing from your .env file.\n"
            "Set GROQ_MODEL to a model ID available to your Groq account."
        )

    if not PDF_DIRECTORY.exists():
        raise FileNotFoundError(
            f"PDF directory does not exist: {PDF_DIRECTORY}"
        )

    pdf_files = sorted(PDF_DIRECTORY.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {PDF_DIRECTORY}"
        )

    return pdf_files


# =========================================================
# INITIALIZE AND TEST GROQ
# =========================================================

def initialize_llm():
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # Test connectivity before processing hundreds of chunks.
    try:
        response = llm.invoke(
            "Reply with exactly the word OK."
        )
        print(f"Groq connectivity test passed: {response.content}")

    except Exception as exc:
        raise RuntimeError(
            f"Groq connectivity test failed for model '{MODEL_NAME}'. "
            "Check the model ID, API key, and account access. "
            f"Original error: {exc}"
        ) from exc

    return llm


# =========================================================
# EXTRACT AND CHUNK PDF TEXT
# =========================================================

def extract_pdf_chunks(pdf_files):
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for pdf_path in pdf_files:
        print(f"Reading PDF: {pdf_path.name}")

        try:
            reader = PdfReader(str(pdf_path))

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = " ".join(text.split())

                if not text:
                    continue

                for start in range(0, len(text), step):
                    chunk_text = text[start:start + CHUNK_SIZE].strip()

                    if not chunk_text:
                        continue

                    chunks.append({
                        "source": pdf_path.name,
                        "page": page_number,
                        "chunk_id": (
                            f"{pdf_path.stem}_p{page_number}_"
                            f"c{start // step}"
                        ),
                        "text": chunk_text,
                    })

        except Exception as exc:
            raise RuntimeError(
                f"Failed to read PDF '{pdf_path.name}': {exc}"
            ) from exc

    if not chunks:
        raise RuntimeError(
            "No extractable text was found in the PDFs. "
            "Scanned PDFs may require OCR."
        )

    print(f"\nTotal extracted chunks: {len(chunks)}")

    # Reproducible sample for initial testing.
    selected_chunks = chunks[:MAX_CHUNKS]

    print(
        f"Using {len(selected_chunks)} chunks for this run. "
        "Increase MAX_CHUNKS after validating the results."
    )

    return selected_chunks


# =========================================================
# GENERATE QUESTION-ANSWER EXAMPLES
# =========================================================

def generate_examples(llm, chunks):
    goldens = []

    for index, chunk in enumerate(chunks, start=1):
        print(
            f"Generating examples from chunk "
            f"{index}/{len(chunks)}..."
        )

        prompt = f"""
You are generating evaluation examples for an Indian
government welfare-scheme RAG chatbot.

Create up to {QUESTIONS_PER_CHUNK} useful question-answer
examples based ONLY on the source text below.

Rules:
1. Questions must be answerable from the source text.
2. Expected answers must be supported by the text.
3. Do not invent amounts, dates, eligibility rules, or benefits.
4. Avoid questions whose answers are not established by the text.
5. Return valid JSON only, without Markdown fences.

Required format:
{{
  "examples": [
    {{
      "input": "Question",
      "expected_output": "Answer"
    }}
  ]
}}

Source file: {chunk["source"]}
Page: {chunk["page"]}

Source text:
{chunk["text"]}
"""

        try:
            response = llm.invoke(prompt)
            content = response.content

            if not isinstance(content, str):
                content = str(content)

            content = content.strip()

            # Handle responses wrapped in Markdown code fences.
            if content.startswith("```"):
                lines = content.splitlines()
                content = "\n".join(
                    line for line in lines
                    if not line.strip().startswith("```")
                )

            data = json.loads(content)

            examples = data.get("examples", [])

            if not isinstance(examples, list):
                raise ValueError(
                    "The JSON 'examples' field must be a list."
                )

            for example in examples:
                question = str(
                    example.get("input", "")
                ).strip()

                answer = str(
                    example.get("expected_output", "")
                ).strip()

                if not question or not answer:
                    continue

                goldens.append({
                    "input": question,
                    "expected_output": answer,
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "chunk_id": chunk["chunk_id"],
                    "retrieval_context": [chunk["text"]],
                })

        except Exception as exc:
            # Stop rather than silently losing hundreds of examples.
            raise RuntimeError(
                f"Failed at chunk {index}/{len(chunks)} "
                f"from {chunk['source']}, page {chunk['page']}. "
                f"Original error: {exc}"
            ) from exc

    return goldens


# =========================================================
# REMOVE DUPLICATE QUESTIONS
# =========================================================

def remove_duplicate_questions(goldens):
    unique = []
    seen = set()

    for golden in goldens:
        normalized_question = golden["input"].strip().casefold()

        if normalized_question not in seen:
            seen.add(normalized_question)
            unique.append(golden)

    return unique


# =========================================================
# SAVE DATASET
# =========================================================

def save_dataset(goldens):
    if not goldens:
        raise RuntimeError(
            "No examples were generated. Check the PDF content "
            "and the model's response."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = {
        "dataset_name": "Welfare Scheme RAG Golden Dataset",
        "model_used": MODEL_NAME,
        "total_examples": len(goldens),
        "examples": goldens,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            dataset,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nDataset generated successfully!")
    print(f"Total examples: {len(goldens)}")
    print(f"Output file: {OUTPUT_FILE}")


# =========================================================
# MAIN
# =========================================================

def main():
    pdf_files = validate_configuration()

    print(f"Found {len(pdf_files)} PDF files.")

    llm = initialize_llm()

    chunks = extract_pdf_chunks(pdf_files)

    goldens = generate_examples(llm, chunks)

    goldens = remove_duplicate_questions(goldens)

    save_dataset(goldens)


if __name__ == "__main__":
    main()