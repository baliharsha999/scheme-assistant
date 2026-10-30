"""
evals/experiments/compare_retrievers.py

Side-by-side A/B comparative evaluation script for the RAG Retriever:
- Baseline Retriever:
    Collection: 'welfare_schemes' (500-char chunks, no headers)
    Candidate pool: BM25 (top 10) + Dense (top 10) -> RRF (top 10) -> BGE Reranker (top 5)
- Improved Retriever:
    Collection: 'welfare_schemes_v2' (900-char chunks with context-enriched metadata headers)
    Candidate pool: BM25 (top 20) + Dense (top 20) -> RRF (top 20) -> BGE Reranker (top 5)

Evaluates Contextual Recall, Contextual Precision, and Hit Rate@5 using DeepEval and Groq.
Generates comprehensive comparative JSON and Markdown reports.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

from evals.config import (
    CONTEXTUAL_RECALL_THRESHOLD,
    CONTEXTUAL_PRECISION_THRESHOLD,
    EVAL_RATE_LIMIT_DELAY,
)
from evals.groq_model import GroqJudgeModel
from evals.dataset_loader import EvaluationDatasetLoader
from evals.evaluate_retriever import retrieve_chunks as baseline_retrieve_chunks, calculate_hit_rate
from evals.experiments.improved_retriever import improved_retrieve_chunks


def run_comparative_retriever_eval(
    test_cases: List[Dict[str, Any]],
    judge_model: Optional[GroqJudgeModel] = None,
    top_k: int = 5,
    delay_between_calls: float = EVAL_RATE_LIMIT_DELAY,
) -> Dict[str, Any]:
    """
    Evaluates both baseline and improved retrievers side-by-side on the same test cases.
    """
    if judge_model is None:
        judge_model = GroqJudgeModel()

    recall_metric = ContextualRecallMetric(
        threshold=CONTEXTUAL_RECALL_THRESHOLD,
        model=judge_model,
        async_mode=False,
        include_reason=True,
    )
    precision_metric = ContextualPrecisionMetric(
        threshold=CONTEXTUAL_PRECISION_THRESHOLD,
        model=judge_model,
        async_mode=False,
        include_reason=True,
    )

    baseline_results = []
    improved_results = []
    case_comparisons = []

    total = len(test_cases)
    print(f"\n=======================================================")
    print(f"Starting Retriever A/B Evaluation on {total} test cases")
    print(f"Top-K: {top_k} | Judge Model: {judge_model.get_model_name()}")
    print(f"=======================================================\n")

    for i, test_case in enumerate(test_cases, start=1):
        case_id = test_case.get("case_id", f"case_{i}")
        query = test_case["input"]
        expected_output = test_case["expected_output"]
        expected_context = test_case.get("expected_retrieval_context", [])
        source_doc = test_case.get("source_doc") or test_case.get("source")
        source_page = test_case.get("source_page") or test_case.get("page")
        source_chunk_id = test_case.get("source_chunk_id") or test_case.get("chunk_id")

        print(f"[{i}/{total}] Evaluating Case '{case_id}': {query[:75]}...")

        # ---------------- 1. Baseline Retrieval & Evaluation ----------------
        print("  -> Querying Baseline Retriever ('welfare_schemes')...")
        t0 = time.time()
        base_retrieved = baseline_retrieve_chunks(query, top_k=top_k)
        base_retrieval_time = round(time.time() - t0, 3)

        base_retrieval_context = [c.get("text", "") for c in base_retrieved]
        base_hit_diag = calculate_hit_rate(source_doc, source_page, source_chunk_id, base_retrieved)

        base_test_case = LLMTestCase(
            input=query,
            actual_output=expected_output,
            expected_output=expected_output,
            retrieval_context=base_retrieval_context if base_retrieval_context else ["No documents retrieved."],
        )

        time.sleep(delay_between_calls)
        recall_metric.measure(base_test_case)
        base_recall_score = float(recall_metric.score or 0.0)
        base_recall_reason = recall_metric.reason

        time.sleep(delay_between_calls)
        precision_metric.measure(base_test_case)
        base_precision_score = float(precision_metric.score or 0.0)
        base_precision_reason = precision_metric.reason

        base_eval_case = {
            "case_id": case_id,
            "input": query,
            "expected_output": expected_output,
            "retrieved_chunks": base_retrieved,
            "hit_diagnostics": base_hit_diag,
            "contextual_recall": {"score": round(base_recall_score, 4), "reason": base_recall_reason},
            "contextual_precision": {"score": round(base_precision_score, 4), "reason": base_precision_reason},
            "retrieval_time_s": base_retrieval_time,
        }
        baseline_results.append(base_eval_case)

        # ---------------- 2. Improved Retrieval & Evaluation ----------------
        print("  -> Querying Improved Retriever ('welfare_schemes_v2')...")
        t0 = time.time()
        imp_retrieved = improved_retrieve_chunks(query, top_k=top_k)
        imp_retrieval_time = round(time.time() - t0, 3)

        imp_retrieval_context = [c.get("text", "") for c in imp_retrieved]
        imp_hit_diag = calculate_hit_rate(source_doc, source_page, source_chunk_id, imp_retrieved)

        imp_test_case = LLMTestCase(
            input=query,
            actual_output=expected_output,
            expected_output=expected_output,
            retrieval_context=imp_retrieval_context if imp_retrieval_context else ["No documents retrieved."],
        )

        time.sleep(delay_between_calls)
        recall_metric.measure(imp_test_case)
        imp_recall_score = float(recall_metric.score or 0.0)
        imp_recall_reason = recall_metric.reason

        time.sleep(delay_between_calls)
        precision_metric.measure(imp_test_case)
        imp_precision_score = float(precision_metric.score or 0.0)
        imp_precision_reason = precision_metric.reason

        imp_eval_case = {
            "case_id": case_id,
            "input": query,
            "expected_output": expected_output,
            "retrieved_chunks": imp_retrieved,
            "hit_diagnostics": imp_hit_diag,
            "contextual_recall": {"score": round(imp_recall_score, 4), "reason": imp_recall_reason},
            "contextual_precision": {"score": round(imp_precision_score, 4), "reason": imp_precision_reason},
            "retrieval_time_s": imp_retrieval_time,
        }
        improved_results.append(imp_eval_case)

        # ---------------- 3. Delta Comparison for this Case ----------------
        delta_recall = round(imp_recall_score - base_recall_score, 4)
        delta_precision = round(imp_precision_score - base_precision_score, 4)

        print(f"  [Result {case_id}]")
        print(f"    Recall:    Baseline={base_recall_score:.4f} -> Improved={imp_recall_score:.4f} (Δ {delta_recall:+.4f})")
        print(f"    Precision: Baseline={base_precision_score:.4f} -> Improved={imp_precision_score:.4f} (Δ {delta_precision:+.4f})")
        print(f"    Hit@5:     Baseline={'YES' if base_hit_diag['hit'] else 'NO'} -> Improved={'YES' if imp_hit_diag['hit'] else 'NO'}")

        case_comparisons.append({
            "case_id": case_id,
            "input": query,
            "source_doc": source_doc,
            "source_page": source_page,
            "baseline": {
                "recall": round(base_recall_score, 4),
                "precision": round(base_precision_score, 4),
                "hit": base_hit_diag["hit"],
                "doc_hit": base_hit_diag["doc_hit"],
                "matched_rank": base_hit_diag["matched_rank"],
                "recall_reason": base_recall_reason,
                "precision_reason": base_precision_reason,
            },
            "improved": {
                "recall": round(imp_recall_score, 4),
                "precision": round(imp_precision_score, 4),
                "hit": imp_hit_diag["hit"],
                "doc_hit": imp_hit_diag["doc_hit"],
                "matched_rank": imp_hit_diag["matched_rank"],
                "recall_reason": imp_recall_reason,
                "precision_reason": imp_precision_reason,
            },
            "delta": {
                "recall": delta_recall,
                "precision": delta_precision,
                "hit_improved": imp_hit_diag["hit"] and not base_hit_diag["hit"],
            }
        })

    # Summary aggregations
    def _avg(arr):
        return sum(arr) / len(arr) if arr else 0.0

    b_avg_recall = round(_avg([c["baseline"]["recall"] for c in case_comparisons]), 4)
    i_avg_recall = round(_avg([c["improved"]["recall"] for c in case_comparisons]), 4)
    recall_abs_diff = round(i_avg_recall - b_avg_recall, 4)
    recall_pct_gain = round((recall_abs_diff / b_avg_recall * 100), 2) if b_avg_recall > 0 else 0.0

    b_avg_precision = round(_avg([c["baseline"]["precision"] for c in case_comparisons]), 4)
    i_avg_precision = round(_avg([c["improved"]["precision"] for c in case_comparisons]), 4)
    prec_abs_diff = round(i_avg_precision - b_avg_precision, 4)
    prec_pct_gain = round((prec_abs_diff / b_avg_precision * 100), 2) if b_avg_precision > 0 else 0.0

    b_hit_rate = round(sum(1 for c in case_comparisons if c["baseline"]["hit"]) / total, 4)
    i_hit_rate = round(sum(1 for c in case_comparisons if c["improved"]["hit"]) / total, 4)
    hit_abs_diff = round(i_hit_rate - b_hit_rate, 4)
    hit_pct_gain = round((hit_abs_diff / b_hit_rate * 100), 2) if b_hit_rate > 0 else 0.0

    summary = {
        "total_cases_evaluated": total,
        "model_used": judge_model.get_model_name(),
        "baseline_retriever": {
            "collection": "welfare_schemes",
            "chunk_size": 500,
            "chunk_overlap": 200,
            "metadata_injected": False,
            "candidate_pool_size": 10,
            "metrics": {
                "contextual_recall": b_avg_recall,
                "contextual_precision": b_avg_precision,
                "hit_rate_at_5": b_hit_rate,
            }
        },
        "improved_retriever": {
            "collection": "welfare_schemes_v2",
            "chunk_size": 900,
            "chunk_overlap": 150,
            "metadata_injected": True,
            "candidate_pool_size": 20,
            "metrics": {
                "contextual_recall": i_avg_recall,
                "contextual_precision": i_avg_precision,
                "hit_rate_at_5": i_hit_rate,
            }
        },
        "improvements": {
            "contextual_recall": {
                "baseline": b_avg_recall,
                "improved": i_avg_recall,
                "absolute_gain": recall_abs_diff,
                "percentage_increase": f"{recall_pct_gain:+.2f}%",
            },
            "contextual_precision": {
                "baseline": b_avg_precision,
                "improved": i_avg_precision,
                "absolute_gain": prec_abs_diff,
                "percentage_increase": f"{prec_pct_gain:+.2f}%",
            },
            "hit_rate_at_5": {
                "baseline": b_hit_rate,
                "improved": i_hit_rate,
                "absolute_gain": hit_abs_diff,
                "percentage_increase": f"{hit_pct_gain:+.2f}%",
            }
        },
        "case_comparisons": case_comparisons,
    }

    return summary


def save_comparison_reports(summary: Dict[str, Any], output_dir: str = "evaluation_results"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_file = out_path / "retriever_comparison.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] JSON comparison report -> {json_file}")

    # Generate Markdown Report
    md_file = out_path / "retriever_comparison_report.md"
    imp = summary["improvements"]
    b_ret = summary["baseline_retriever"]
    i_ret = summary["improved_retriever"]

    md_content = f"""# RAG Retriever Optimization & Comparative Evaluation Report

## Executive Summary
This report documents the performance impact of transitioning from the baseline chunking & retrieval configuration to the **Context-Enriched Metadata-Injected Retriever** on government welfare schemes.

### Core Benchmark Metrics (Before vs. After)

| Metric | Baseline Retriever (`welfare_schemes`) | Improved Retriever (`welfare_schemes_v2`) | Absolute Gain | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Contextual Recall** | `{imp['contextual_recall']['baseline']:.4f}` | `{imp['contextual_recall']['improved']:.4f}` | `{imp['contextual_recall']['absolute_gain']:+.4f}` | **`{imp['contextual_recall']['percentage_increase']}`** |
| **Contextual Precision** | `{imp['contextual_precision']['baseline']:.4f}` | `{imp['contextual_precision']['improved']:.4f}` | `{imp['contextual_precision']['absolute_gain']:+.4f}` | **`{imp['contextual_precision']['percentage_increase']}`** |
| **Hit Rate @ 5** | `{imp['hit_rate_at_5']['baseline'] * 100:.1f}%` | `{imp['hit_rate_at_5']['improved'] * 100:.1f}%` | `{imp['hit_rate_at_5']['absolute_gain'] * 100:+.1f}%` | **`{imp['hit_rate_at_5']['percentage_increase']}`** |

---

## Architecture & Configuration Comparison

| Dimension | Baseline Retriever | Improved Retriever (Proposed) | Why It Matters |
| :--- | :--- | :--- | :--- |
| **Target Collection** | `welfare_schemes` (2,306 chunks) | `welfare_schemes_v2` (1,092 chunks) | Preserves existing production DB 100% intact. |
| **Chunk Size** | 500 characters | **900 characters** | Prevents splitting eligibility criteria & tabular records mid-sentence. |
| **Chunk Overlap** | 200 characters | **150 characters** | Ensures continuity while reducing token redundancy. |
| **Context Injection** | None (Raw chunk text only) | **`[Scheme: ... \| Document: ... \| Page: ...]`** | **Key Breakthrough**: Keyword (BM25) and Dense (BGE-M3) match the correct scheme even on generic queries. |
| **Candidate Retrieval Depth** | Top 10 BM25 + Top 10 Dense | **Top 20 BM25 + Top 20 Dense** | Expands the candidate pool to ensure high recall before reranking. |
| **Reciprocal Rank Fusion** | `top_n = 10` | `top_n = 20` | Delivers 20 top candidates to the cross-encoder reranker. |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-v2-m3` (Top 5) | `BAAI/bge-reranker-v2-m3` (Top 5) | High-precision cross-attention scoring. |

---

## Detailed Per-Case Analysis

| Case ID | Query | Baseline Recall | Improved Recall | Baseline Prec. | Improved Prec. | Baseline Hit@5 | Improved Hit@5 | Impact |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""

    for case in summary["case_comparisons"]:
        c_id = case["case_id"]
        q = case["input"][:45] + ("..." if len(case["input"]) > 45 else "")
        b_rec = case["baseline"]["recall"]
        i_rec = case["improved"]["recall"]
        b_prc = case["baseline"]["precision"]
        i_prc = case["improved"]["precision"]
        b_hit = "YES" if case["baseline"]["hit"] else "NO"
        i_hit = "YES" if case["improved"]["hit"] else "NO"
        
        if i_rec > b_rec or i_prc > b_prc or (case["improved"]["hit"] and not case["baseline"]["hit"]):
            impact = "Improved"
        elif i_rec == b_rec and i_prc == b_prc:
            impact = "Maintained"
        else:
            impact = "Regressed"

        md_content += f"| `{c_id}` | {q} | `{b_rec:.2f}` | `{i_rec:.2f}` | `{b_prc:.2f}` | `{i_prc:.2f}` | {b_hit} | {i_hit} | **{impact}** |\n"

    md_content += f"""
---

## DeepEval Evaluation Reasoning Breakdown

"""
    for case in summary["case_comparisons"]:
        md_content += f"### {case['case_id']}: \"{case['input']}\"\n\n"
        md_content += f"- **Expected Source**: `{case['source_doc']}` (Page {case['source_page']})\n"
        md_content += f"- **Baseline Recall ({case['baseline']['recall']})**: {case['baseline']['recall_reason']}\n"
        md_content += f"- **Improved Recall ({case['improved']['recall']})**: {case['improved']['recall_reason']}\n"
        md_content += f"- **Baseline Precision ({case['baseline']['precision']})**: {case['baseline']['precision_reason']}\n"
        md_content += f"- **Improved Precision ({case['improved']['precision']})**: {case['improved']['precision_reason']}\n\n"

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Saved] Markdown comparison report -> {md_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B Comparative Evaluation of Baseline vs. Improved Retriever")
    parser.add_argument("--limit", type=int, default=5, help="Number of test cases to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Starting index in dataset")
    args = parser.parse_args()

    loader = EvaluationDatasetLoader()
    all_examples = loader.get_validated_examples()
    test_cases = all_examples[args.offset : args.offset + args.limit]
    judge = GroqJudgeModel()

    summary = run_comparative_retriever_eval(test_cases, judge_model=judge)
    save_comparison_reports(summary)
