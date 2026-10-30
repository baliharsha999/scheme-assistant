# RAG Retriever Component Evaluation Report

- **Model Used for Evaluation**: `openai/gpt-oss-120b`
- **Total Test Cases Evaluated**: 5
- **Retrieval Depth (Top-k)**: 5

## 1. Aggregate Metric Summary

| Metric | Average Score | Pass Rate | Passed / Total | Threshold |
| :--- | :---: | :---: | :---: | :---: |
| **Contextual Recall** | `0.8000` | 80.0% | 4 / 5 | $\ge 0.5$ |
| **Contextual Precision** | `0.6978` | 80.0% | 4 / 5 | $\ge 0.5$ |
| **Hit Rate@k (Provenance Match)** | `0.8000` | 80.0% | 4 / 5 | N/A (Diagnostic) |

## 2. Metric Explanations

- **Contextual Recall**: Evaluates whether the retrieved context contains all facts required to answer the query accurately, compared to the ground-truth expected answer.
- **Contextual Precision**: Evaluates whether relevant chunks are prioritized at top ranks rather than diluted by irrelevant or ambiguous documents.
- **Hit Rate@k**: Diagnostic tracking whether the ground-truth PDF document and page number appear anywhere within the top-k retrieved chunks.

## 3. Failure Analysis (1 Failed Cases)

### Case #1: eval_case_1 - "What is the name of the scheme mentioned in the document?"
- **Source**: `Mission Saksham Anganwadi and Poshan 2.0 scheme guidelines.pdf` (Page 1)
- **Expected Answer**: The scheme is called "Mission Saksham Anganwadi and Poshan 2.0".
- **Contextual Recall**: `0.0` (Reason: The score is 0.00 because sentence 1 cannot be linked to any node(s) in retrieval context, as none mention “Mission Saksham Anganwadi and Poshan 2.0.”)
- **Contextual Precision**: `0.0` (Reason: The score is 0.00 because all five retrieved nodes are irrelevant to the query – node 1 says "The text refers to the “Interest Subsidy Scheme” (ISS) under PMAY‑U 2.0, which is unrelated to “Mission Saksham Anganwadi and Poshan 2.0"", node 2 notes "This excerpt discusses house ownership rules under a scheme, but does not mention “Mission Saksham Anganwadi and Poshan 2.0"", node 3 states "It talks about beneficiary validation for PMAY‑U 2.0, not the required scheme name.", node 4 mentions "The passage lists activities for a ‘Grants‑in‑Aid’ component, again unrelated to the target scheme.", and node 5 reports "This document describes the Pradhan Mantri Anusuchit Jaati Abhyuday Yojana (PM‑AJAY), not “Mission Saksham Anganwadi and Poshan 2.0"". Since irrelevant nodes occupy the top ranks and there are no relevant nodes, the contextual precision cannot be higher than 0.00.)
- **Hit Detected**: `False` (Matched Rank: -1)

**Retrieved Chunks Top-3 Snippets:**
- **Rank 1** (`Operational-Guidelines-of-PMAY-U-2.pdf`, p.64, rerank: -1.9850490093231201): the context shall mean and include their successors, assignees and administrators) on the SECOND 
PART;
WHEREAS, the Ministry of Housing and Urban Affairs (MoHUA), Government of India is 
implementing...
- **Rank 2** (`Operational-Guidelines-of-PMAY-U-2.pdf`, p.16, rerank: -3.7890684604644775): timeline by signing a Memorandum of Agreement (MoA) with MoHUA as per the format 
at Annexure-1A. 
1.8 
The houses constructed/ acquired/purchased with Central Assistance under the Scheme 
should be i...
- **Rank 3** (`Operational-Guidelines-of-PMAY-U-2.pdf`, p.18, rerank: -4.819698333740234): on unified Web-portal. The beneficiaries will be validated by States/UTs/ULBs as per the 
eligibility criteria of the scheme guidelines. States/UTs/ULBs should verify and exclude the 
beneficiary fami...
