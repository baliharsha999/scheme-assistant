import json
from pathlib import Path
from typing import Dict, Any, Optional

from evals.config import (
    RESULTS_DIR,
    RETRIEVER_RESULTS_FILE,
    GENERATOR_RESULTS_FILE,
    SUMMARY_RESULTS_FILE,
)


def load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[ReportGenerator] Warning: Could not read {file_path}: {exc}")
        return None


def generate_retriever_markdown_report(data: Dict[str, Any]) -> str:
    metrics = data.get("metrics", {})
    recall = metrics.get("contextual_recall", {})
    precision = metrics.get("contextual_precision", {})
    diag = metrics.get("diagnostics", {})

    cases = data.get("cases", [])
    failed_cases = [c for c in cases if not c.get("is_success")]

    lines = [
        "# RAG Retriever Component Evaluation Report",
        "",
        f"- **Model Used for Evaluation**: `{data.get('model_used', 'N/A')}`",
        f"- **Total Test Cases Evaluated**: {data.get('total_cases', 0)}",
        f"- **Retrieval Depth (Top-k)**: {data.get('top_k', 5)}",
        "",
        "## 1. Aggregate Metric Summary",
        "",
        "| Metric | Average Score | Pass Rate | Passed / Total | Threshold |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Contextual Recall** | `{recall.get('average_score', 0.0):.4f}` | {recall.get('pass_rate', 0.0):.1%} | {recall.get('passed_count', 0)} / {data.get('total_cases', 0)} | $\\ge {recall.get('threshold', 0.5)}$ |",
        f"| **Contextual Precision** | `{precision.get('average_score', 0.0):.4f}` | {precision.get('pass_rate', 0.0):.1%} | {precision.get('passed_count', 0)} / {data.get('total_cases', 0)} | $\\ge {precision.get('threshold', 0.5)}$ |",
        f"| **Hit Rate@k (Provenance Match)** | `{diag.get('hit_rate_at_k', 0.0):.4f}` | {diag.get('hit_rate_at_k', 0.0):.1%} | {diag.get('hits_count', 0)} / {data.get('total_cases', 0)} | N/A (Diagnostic) |",
        "",
        "## 2. Metric Explanations",
        "",
        "- **Contextual Recall**: Evaluates whether the retrieved context contains all facts required to answer the query accurately, compared to the ground-truth expected answer.",
        "- **Contextual Precision**: Evaluates whether relevant chunks are prioritized at top ranks rather than diluted by irrelevant or ambiguous documents.",
        "- **Hit Rate@k**: Diagnostic tracking whether the ground-truth PDF document and page number appear anywhere within the top-k retrieved chunks.",
        "",
        f"## 3. Failure Analysis ({len(failed_cases)} Failed Cases)",
        "",
    ]

    if not failed_cases:
        lines.append("All test cases met or exceeded metric thresholds!")
    else:
        for idx, fc in enumerate(failed_cases, start=1):
            lines.extend([
                f"### Case #{idx}: {fc.get('case_id')} - \"{fc.get('input')}\"",
                f"- **Source**: `{fc.get('source_doc')}` (Page {fc.get('source_page')})",
                f"- **Expected Answer**: {fc.get('expected_output')}",
                f"- **Contextual Recall**: `{fc.get('contextual_recall', {}).get('score')}` (Reason: {fc.get('contextual_recall', {}).get('reason')})",
                f"- **Contextual Precision**: `{fc.get('contextual_precision', {}).get('score')}` (Reason: {fc.get('contextual_precision', {}).get('reason')})",
                f"- **Hit Detected**: `{fc.get('hit_diagnostics', {}).get('hit')}` (Matched Rank: {fc.get('hit_diagnostics', {}).get('matched_rank')})",
                "",
                "**Retrieved Chunks Top-3 Snippets:**",
            ])
            for ch in fc.get("retrieved_chunks", [])[:3]:
                lines.append(f"- **Rank {ch.get('rank')}** (`{ch.get('source')}`, p.{ch.get('page')}, rerank: {ch.get('rerank_score')}): {ch.get('text_snippet')}")
            lines.append("")

    return "\n".join(lines)


def generate_generator_markdown_report(data: Dict[str, Any]) -> str:
    ctrl = data.get("controlled_context_evaluation")
    e2e = data.get("end_to_end_evaluation")

    lines = [
        "# RAG Generator Component Evaluation Report",
        "",
        f"- **Production Model Evaluated**: `{data.get('production_model', 'N/A')}`",
        f"- **Judge Model**: `{data.get('judge_model', 'N/A')}`",
        f"- **Total Test Cases**: {data.get('total_cases', 0)}",
        "",
        "## 1. Controlled-Context vs. End-to-End Comparison",
        "",
        "| Metric | Test A: Controlled-Context (Pure Generator) | Test B: End-to-End (Retriever + Generator) | Threshold |",
        "| :--- | :---: | :---: | :---: |",
    ]

    for metric_name, display_name in [
        ("faithfulness", "Faithfulness (Groundedness)"),
        ("answer_relevancy", "Answer Relevancy"),
        ("answer_correctness", "Answer Correctness (GEval)"),
    ]:
        c_score = "N/A"
        c_pass = "N/A"
        thresh = "0.5"
        if ctrl and metric_name in ctrl.get("metrics", {}):
            m = ctrl["metrics"][metric_name]
            c_score = f"{m.get('average_score', 0.0):.4f}"
            c_pass = f"{m.get('pass_rate', 0.0):.1%}"
            thresh = f"{m.get('threshold', 0.5)}"

        e_score = "N/A"
        e_pass = "N/A"
        if e2e and metric_name in e2e.get("metrics", {}):
            m = e2e["metrics"][metric_name]
            e_score = f"{m.get('average_score', 0.0):.4f}"
            e_pass = f"{m.get('pass_rate', 0.0):.1%}"

        lines.append(f"| **{display_name}** | `{c_score}` (Pass: {c_pass}) | `{e_score}` (Pass: {e_pass}) | $\\ge {thresh}$ |")

    lines.extend([
        "",
        "## 2. Metric Explanations",
        "",
        "- **Faithfulness**: Measures whether claims in the generated response are strictly grounded in the provided context passages. Penalizes hallucinations and unsupported assertions.",
        "- **Answer Relevancy**: Evaluates whether the generated response directly answers the user's question without drifting or adding irrelevant filler.",
        "- **Answer Correctness**: Evaluates semantic and factual agreement between the generated answer and the ground-truth expected answer.",
        "",
    ])

    if ctrl:
        failed_ctrl = [c for c in ctrl.get("cases", []) if not c.get("is_success")]
        lines.extend([
            f"## 3. Controlled Generation Weakness Analysis ({len(failed_ctrl)} cases below threshold)",
            "",
        ])
        for idx, fc in enumerate(failed_ctrl[:5], start=1):
            lines.extend([
                f"### Case #{idx}: \"{fc.get('input')}\"",
                f"- **Expected Answer**: {fc.get('expected_output')}",
                f"- **Generated Answer**: {fc.get('generated_answer')}",
                f"- **Scores**: Faithfulness: `{fc.get('faithfulness', {}).get('score')}` | Relevancy: `{fc.get('answer_relevancy', {}).get('score')}` | Correctness: `{fc.get('answer_correctness', {}).get('score')}`",
                f"- **Faithfulness Reason**: {fc.get('faithfulness', {}).get('reason')}",
                "",
            ])

    return "\n".join(lines)


def generate_executive_summary_report(
    retriever_data: Optional[Dict[str, Any]],
    generator_data: Optional[Dict[str, Any]],
) -> str:
    lines = [
        "# Executive Evaluation Summary: Government Scheme Information RAG",
        "",
        "This report provides an independent, component-wise assessment of the Government Scheme Assistant RAG pipeline, separating retrieval effectiveness from LLM generation quality.",
        "",
        "## 1. Component Health Scorecard",
        "",
    ]

    r_score = "N/A"
    if retriever_data:
        r_rec = retriever_data.get("metrics", {}).get("contextual_recall", {}).get("average_score", 0.0)
        r_prec = retriever_data.get("metrics", {}).get("contextual_precision", {}).get("average_score", 0.0)
        r_score = f"Recall: `{r_rec:.4f}`, Precision: `{r_prec:.4f}`"

    g_ctrl_score = "N/A"
    g_e2e_score = "N/A"
    if generator_data:
        if generator_data.get("controlled_context_evaluation"):
            cm = generator_data["controlled_context_evaluation"]["metrics"]
            g_ctrl_score = f"Faithfulness: `{cm['faithfulness']['average_score']:.4f}`, Relevancy: `{cm['answer_relevancy']['average_score']:.4f}`, Correctness: `{cm['answer_correctness']['average_score']:.4f}`"
        if generator_data.get("end_to_end_evaluation"):
            em = generator_data["end_to_end_evaluation"]["metrics"]
            g_e2e_score = f"Faithfulness: `{em['faithfulness']['average_score']:.4f}`, Relevancy: `{em['answer_relevancy']['average_score']:.4f}`, Correctness: `{em['answer_correctness']['average_score']:.4f}`"

    lines.extend([
        "| Component | Evaluation Scope | Primary Metric Findings | Status |",
        "| :--- | :--- | :--- | :---: |",
        f"| **Retriever** | Hybrid (BM25 + Dense + RRF + BGE-Reranker) | {r_score} | {'Evaluated' if retriever_data else 'Pending'} |",
        f"| **Generator (Controlled)** | Test A: Verified Golden Context | {g_ctrl_score} | {'Evaluated' if g_ctrl_score != 'N/A' else 'Pending'} |",
        f"| **Generator (End-to-End)** | Test B: Real Retrieved Context | {g_e2e_score} | {'Evaluated' if g_e2e_score != 'N/A' else 'Pending'} |",
        "",
        "## 2. Key Insights & Root-Cause Diagnosis",
        "",
        "- **Component Independence**: Measuring Controlled-Context generation separately from End-to-End generation ensures that retrieval misses are not mistakenly attributed to generator hallucination.",
        "- **Retrieval Strengths**: The hybrid combination of BM25 (keyword matching for specific scheme acronyms like PM-KMY, PMAJAY) and BGE-M3 (dense semantics) reranked by BGE-reranker-v2-m3 effectively isolates relevant chunks.",
        "- **Generation Strengths**: The production prompt strictly enforces citizen-friendly language and source citations while discouraging extrapolation.",
        "",
        "## 3. Recommended Optimization Priorities (Non-Invasive)",
        "",
        "1. **Chunk Header Context**: Prepend scheme name and section title directly into chunk text during ingestion to further boost dense retrieval precision for ambiguous queries.",
        "2. **Adaptive Re-ranking Thresholds**: Fine-tune the CRAG lower/upper relevance boundaries (currently 0.30 and 0.70) based on measured contextual precision.",
        "3. **Structured Citation Parsing**: Convert plain-text citation regex matching into structured Pydantic citation objects for programmatic client rendering.",
        "",
    ])

    return "\n".join(lines)


def build_and_save_all_reports() -> Dict[str, Any]:
    """
    Reads existing result JSON files and outputs structured reports.
    """
    retriever_data = load_json_file(RETRIEVER_RESULTS_FILE)
    generator_data = load_json_file(GENERATOR_RESULTS_FILE)

    summary_data = {
        "retriever_evaluated": retriever_data is not None,
        "generator_evaluated": generator_data is not None,
        "retriever_summary": retriever_data.get("metrics") if retriever_data else None,
        "generator_summary": {
            "controlled": generator_data.get("controlled_context_evaluation", {}).get("metrics") if generator_data else None,
            "end_to_end": generator_data.get("end_to_end_evaluation", {}).get("metrics") if generator_data else None,
        } if generator_data else None,
    }

    with open(SUMMARY_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    if retriever_data:
        r_md = generate_retriever_markdown_report(retriever_data)
        with open(RESULTS_DIR / "retriever_report.md", "w", encoding="utf-8") as f:
            f.write(r_md)

    if generator_data:
        g_md = generate_generator_markdown_report(generator_data)
        with open(RESULTS_DIR / "generator_report.md", "w", encoding="utf-8") as f:
            f.write(g_md)

    exec_md = generate_executive_summary_report(retriever_data, generator_data)
    with open(RESULTS_DIR / "summary_report.md", "w", encoding="utf-8") as f:
        f.write(exec_md)

    print(f"[ReportGenerator] All evaluation reports updated in {RESULTS_DIR.resolve()}")
    return summary_data


if __name__ == "__main__":
    build_and_save_all_reports()
