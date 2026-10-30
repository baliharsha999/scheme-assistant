# RAG Generator Component Evaluation Report

- **Production Model Evaluated**: `openai/gpt-oss-120b`
- **Judge Model**: `openai/gpt-oss-120b`
- **Total Test Cases**: 5

## 1. Controlled-Context vs. End-to-End Comparison

| Metric | Test A: Controlled-Context (Pure Generator) | Test B: End-to-End (Retriever + Generator) | Threshold |
| :--- | :---: | :---: | :---: |
| **Faithfulness (Groundedness)** | `0.9000` (Pass: 100.0%) | `1.0000` (Pass: 100.0%) | $\ge 0.5$ |
| **Answer Relevancy** | `0.8800` (Pass: 80.0%) | `1.0000` (Pass: 100.0%) | $\ge 0.5$ |
| **Answer Correctness (GEval)** | `0.7600` (Pass: 80.0%) | `0.6800` (Pass: 60.0%) | $\ge 0.5$ |

## 2. Metric Explanations

- **Faithfulness**: Measures whether claims in the generated response are strictly grounded in the provided context passages. Penalizes hallucinations and unsupported assertions.
- **Answer Relevancy**: Evaluates whether the generated response directly answers the user's question without drifting or adding irrelevant filler.
- **Answer Correctness**: Evaluates semantic and factual agreement between the generated answer and the ground-truth expected answer.

## 3. Controlled Generation Weakness Analysis (1 cases below threshold)

### Case #1: "What are the main sections listed in the contents of the Saksham Anganwadi and Poshan 2.0 guidelines?"
- **Expected Answer**: The contents list the following main sections: 1. INTRODUCTION, 1.1 Vision, 1.2 Objectives, 1.3 Important components of Saksham Anganwadi and Poshan 2.0, 2. INSTITUTIONAL FRAMEWORK FOR IMPLEMENTATION, 2.1 Framework at Central Level, 2.2 Roles and responsibilities of State Administration, 2.3 Roles and responsibilities of District Administration, 2.4 Roles and Responsibilities of Anganwadi Workers.
- **Generated Answer**: **Main sections shown in the contents of the Saksham Anganwadi and Poshan 2.0 guidelines**

- **1. INTRODUCTION**  
  (includes Vision, Objectives, and Important components)

- **2. INSTITUTIONAL FRAMEWORK FOR IMPLEMENTATION**  
  (covers the central‑level framework and the roles and responsibilities of State, District, and Anganwadi workers)

These are the primary headings listed in the contents page of the guidelines.  

[Source: Mission Saksham Anganwadi and Poshan 2.0 scheme guidelines.pdf, Page 2]
- **Scores**: Faithfulness: `1.0` | Relevancy: `0.4` | Correctness: `0.3`
- **Faithfulness Reason**: The score is 1.00 because there are no contradictions, indicating perfect faithfulness.
