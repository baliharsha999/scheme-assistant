import time
import json
from typing import List, Dict, Any, Optional

from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

from src.retrieval.hybrid_search import reciprocal_rank_fusion, reranker
from src.retrieval.bm25_retriever import bm25_search
from src.retrieval.dense_retriever import dense_search
from evals.config import (
    RETRIEVER_RESULTS_FILE,
    CONTEXTUAL_RECALL_THRESHOLD,
    CONTEXTUAL_PRECISION_THRESHOLD,
    EVAL_RATE_LIMIT_DELAY,
)
from evals.groq_model import GroqJudgeModel
from evals.dataset_loader import EvaluationDatasetLoader


def retrieve_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Executes the existing production hybrid retrieval pipeline:
    BM25 Search + Dense Search -> Reciprocal Rank Fusion -> BAAI/bge-reranker-v2-m3.
    """
    bm25_results = bm25_search(query, k=10)
    dense_results = dense_search(query, k=10)
    fused_results = reciprocal_rank_fusion([bm25_results, dense_results], top_n=10)
    reranked_results = reranker(query, fused_results, n=top_k)
    return reranked_results


def calculate_hit_rate(
    expected_source: Optional[str],
    expected_page: Optional[int],
    expected_chunk_id: Optional[str],
    retrieved_documents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Diagnostic metric: Checks whether the expected source document and page/chunk
    were retrieved within top-k.
    """
    doc_hit = False
    page_hit = False
    chunk_hit = False
    matched_rank = -1

    for rank, doc in enumerate(retrieved_documents, start=1):
        meta = doc.get("metadata") or {}
        doc_source = meta.get("source") or doc.get("source")
        doc_page = meta.get("page") or doc.get("page")
        doc_id = doc.get("id")

        if expected_source and doc_source and expected_source.strip().lower() in doc_source.strip().lower():
            doc_hit = True
            if expected_page is not None and doc_page is not None and int(expected_page) == int(doc_page):
                page_hit = True
                if matched_rank == -1:
                    matched_rank = rank
        if expected_chunk_id and doc_id and expected_chunk_id.strip() == doc_id.strip():
            chunk_hit = True
            if matched_rank == -1:
                matched_rank = rank

    return {
        "hit": page_hit or chunk_hit or (doc_hit and expected_page is None),
        "doc_hit": doc_hit,
        "page_hit": page_hit,
        "chunk_hit": chunk_hit,
        "matched_rank": matched_rank,
    }


class RetrieverEvaluator:
    """
    Component-wise evaluator for the RAG Retriever using DeepEval and Groq.
    """

    def __init__(
        self,
        judge_model: Optional[GroqJudgeModel] = None,
        top_k: int = 5,
        recall_threshold: float = CONTEXTUAL_RECALL_THRESHOLD,
        precision_threshold: float = CONTEXTUAL_PRECISION_THRESHOLD,
    ):
        self.judge_model = judge_model or GroqJudgeModel()
        self.top_k = top_k
        self.recall_threshold = recall_threshold
        self.precision_threshold = precision_threshold

        # Initialize DeepEval metrics with the Groq judge model
        self.recall_metric = ContextualRecallMetric(
            threshold=self.recall_threshold,
            model=self.judge_model,
            async_mode=False,
            include_reason=True,
        )
        self.precision_metric = ContextualPrecisionMetric(
            threshold=self.precision_threshold,
            model=self.judge_model,
            async_mode=False,
            include_reason=True,
        )

    def evaluate_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single test case through the retriever and metrics.
        """
        query = case["input"]
        expected_output = case["expected_output"]
        expected_context = case["expected_retrieval_context"]
        source_doc = case.get("source")
        source_page = case.get("page")
        source_chunk_id = case.get("chunk_id")

        # 1. Execute actual retrieval
        retrieved_docs = retrieve_chunks(query, top_k=self.top_k)
        retrieval_context = [doc["text"] for doc in retrieved_docs if doc.get("text")]

        # If retrieval returns nothing, handle gracefully
        if not retrieval_context:
            retrieval_context = ["No documents retrieved."]

        # 2. Check diagnostic hit rate
        hit_diag = calculate_hit_rate(
            expected_source=source_doc,
            expected_page=source_page,
            expected_chunk_id=source_chunk_id,
            retrieved_documents=retrieved_docs,
        )

        # 3. Build DeepEval LLMTestCase
        test_case = LLMTestCase(
            input=query,
            actual_output=expected_output,
            expected_output=expected_output,
            context=expected_context,
            retrieval_context=retrieval_context,
            metadata={
                "source": source_doc,
                "page": source_page,
                "chunk_id": source_chunk_id,
            },
            name=case.get("case_id"),
        )

        # 4. Measure Contextual Recall
        recall_score = 0.0
        recall_reason = ""
        try:
            recall_score = self.recall_metric.measure(test_case)
            recall_reason = getattr(self.recall_metric, "reason", "")
        except Exception as exc:
            recall_reason = f"Error computing Contextual Recall: {exc}"

        time.sleep(EVAL_RATE_LIMIT_DELAY)

        # 5. Measure Contextual Precision
        precision_score = 0.0
        precision_reason = ""
        try:
            precision_score = self.precision_metric.measure(test_case)
            precision_reason = getattr(self.precision_metric, "reason", "")
        except Exception as exc:
            precision_reason = f"Error computing Contextual Precision: {exc}"

        time.sleep(EVAL_RATE_LIMIT_DELAY)

        is_success = (
            recall_score >= self.recall_threshold
            and precision_score >= self.precision_threshold
        )

        # Prepare retrieved chunks summary
        retrieved_summary = []
        for rank, doc in enumerate(retrieved_docs, start=1):
            meta = doc.get("metadata") or {}
            retrieved_summary.append({
                "rank": rank,
                "chunk_id": doc.get("id"),
                "source": meta.get("source") or doc.get("source"),
                "page": meta.get("page") or doc.get("page"),
                "rerank_score": doc.get("rerank_score"),
                "text_snippet": (doc.get("text") or "")[:200] + "...",
            })

        return {
            "case_id": case.get("case_id"),
            "input": query,
            "expected_output": expected_output,
            "expected_retrieval_context": expected_context,
            "source_doc": source_doc,
            "source_page": source_page,
            "source_chunk_id": source_chunk_id,
            "retrieved_chunks": retrieved_summary,
            "hit_diagnostics": hit_diag,
            "contextual_recall": {
                "score": float(recall_score),
                "threshold": self.recall_threshold,
                "passed": bool(recall_score >= self.recall_threshold),
                "reason": recall_reason,
            },
            "contextual_precision": {
                "score": float(precision_score),
                "threshold": self.precision_threshold,
                "passed": bool(precision_score >= self.precision_threshold),
                "reason": precision_reason,
            },
            "is_success": is_success,
        }

    def evaluate(
        self,
        cases: Optional[List[Dict[str, Any]]] = None,
        max_cases: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run retriever evaluation across the provided or golden test cases.
        """
        if cases is None:
            loader = EvaluationDatasetLoader()
            cases = loader.get_validated_examples()

        if max_cases is not None and max_cases > 0:
            cases = cases[:max_cases]

        print(f"\n=======================================================")
        print(f"STARTING RETRIEVER EVALUATION: {len(cases)} test cases")
        print(f"Judge Model: {self.judge_model.get_model_name()}")
        print(f"Top-K Chunks: {self.top_k}")
        print(f"=======================================================\n")

        case_results = []
        total_recall = 0.0
        total_precision = 0.0
        recall_passed_count = 0
        precision_passed_count = 0
        hits_count = 0

        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] Evaluating Retriever for: '{case['input'][:65]}...'")
            result = self.evaluate_case(case)
            case_results.append(result)

            rec_score = result["contextual_recall"]["score"]
            prec_score = result["contextual_precision"]["score"]
            total_recall += rec_score
            total_precision += prec_score

            if result["contextual_recall"]["passed"]:
                recall_passed_count += 1
            if result["contextual_precision"]["passed"]:
                precision_passed_count += 1
            if result["hit_diagnostics"]["hit"]:
                hits_count += 1

            print(
                f"   -> Recall: {rec_score:.2f} | Precision: {prec_score:.2f} | "
                f"Hit: {result['hit_diagnostics']['hit']}"
            )

        n = len(cases)
        avg_recall = total_recall / n if n > 0 else 0.0
        avg_precision = total_precision / n if n > 0 else 0.0
        recall_pass_rate = recall_passed_count / n if n > 0 else 0.0
        precision_pass_rate = precision_passed_count / n if n > 0 else 0.0
        hit_rate = hits_count / n if n > 0 else 0.0

        summary = {
            "component": "Retriever",
            "model_used": self.judge_model.get_model_name(),
            "total_cases": n,
            "top_k": self.top_k,
            "metrics": {
                "contextual_recall": {
                    "average_score": round(avg_recall, 4),
                    "pass_rate": round(recall_pass_rate, 4),
                    "passed_count": recall_passed_count,
                    "threshold": self.recall_threshold,
                },
                "contextual_precision": {
                    "average_score": round(avg_precision, 4),
                    "pass_rate": round(precision_pass_rate, 4),
                    "passed_count": precision_passed_count,
                    "threshold": self.precision_threshold,
                },
                "diagnostics": {
                    "hit_rate_at_k": round(hit_rate, 4),
                    "hits_count": hits_count,
                }
            },
            "cases": case_results,
        }

        # Save results
        with open(RETRIEVER_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n=======================================================")
        print(f"RETRIEVER EVALUATION COMPLETED")
        print(f"Average Contextual Recall:    {avg_recall:.4f} (Pass rate: {recall_pass_rate:.1%})")
        print(f"Average Contextual Precision: {avg_precision:.4f} (Pass rate: {precision_pass_rate:.1%})")
        print(f"Diagnostic Hit Rate@{self.top_k}:         {hit_rate:.1%}")
        print(f"Results saved to: {RETRIEVER_RESULTS_FILE.resolve()}")
        print(f"=======================================================\n")

        return summary


def run_retriever_evaluation(max_cases: Optional[int] = None) -> Dict[str, Any]:
    evaluator = RetrieverEvaluator()
    return evaluator.evaluate(max_cases=max_cases)


if __name__ == "__main__":
    run_retriever_evaluation(max_cases=5)
