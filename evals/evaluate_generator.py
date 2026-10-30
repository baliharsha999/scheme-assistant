import time
import json
import asyncio
from typing import List, Dict, Any, Optional

from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from src.llm.model import llm as production_llm
from evals.config import (
    GENERATOR_RESULTS_FILE,
    FAITHFULNESS_THRESHOLD,
    ANSWER_RELEVANCY_THRESHOLD,
    ANSWER_CORRECTNESS_THRESHOLD,
    EVAL_RATE_LIMIT_DELAY,
)
from evals.groq_model import GroqJudgeModel
from evals.dataset_loader import EvaluationDatasetLoader
from evals.evaluate_retriever import retrieve_chunks


def format_context_string(passages: List[str], metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Formats reference passages into the exact context block structure used
    by the application's generate_answer_node.
    """
    if not passages:
        return "No evidence available."

    meta = metadata or {}
    source = meta.get("source", "Official Scheme Document")
    page = meta.get("page", "General")

    parts = []
    for passage in passages:
        if passage and passage.strip():
            parts.append(
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content: {passage.strip()}"
            )
    return "\n\n".join(parts)


def format_retrieved_context_string(retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Formats chunks returned by the retriever into the application's context block structure.
    """
    if not retrieved_docs:
        return "No evidence available."

    parts = []
    for doc in retrieved_docs:
        text = doc.get("text") or doc.get("content") or ""
        meta = doc.get("metadata") or {}
        source = meta.get("source") or doc.get("source", "Unknown source")
        page = meta.get("page") or doc.get("page", "Unknown")

        if text.strip():
            parts.append(
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content: {text.strip()}"
            )
    return "\n\n".join(parts)


def generate_production_answer(query: str, formatted_context: str) -> str:
    """
    Invokes the production generator model using the exact production prompt structure
    found in src/CRag/graph_nodes.py.
    """
    prompt = f"""
    You are a helpful and accurate government welfare scheme assistant.

    Your task is to answer the user's question using only the evidence provided.

    Instructions:

    1. Answer the user's question directly in clear, simple language.

    2. Synthesize information from the relevant passages into a coherent answer.
       Do not simply copy entire passages from the documents.

    3. Include only information relevant to the user's question.

    4. Use bullet points for eligibility criteria, benefits, required documents,
       and application procedures when appropriate.

    5. Do not invent facts, eligibility conditions, benefits, dates, or procedures.

    6. If a requested detail is missing from the evidence, explicitly say that
       the available documents do not specify it.

    7. Cite sources using the document filename and page number provided
       in the context, for example: [Source: scheme_guidelines.pdf, Page 12].

    8. Do not invent or guess source names or page numbers.

    9. Do not mention internal technical terms such as chunks, embeddings,
       retrievers, or RAG.

    User question:

    {query}

    Evidence:

    {formatted_context}

    Provide a clear, concise answer for an ordinary citizen.
    """
    response = production_llm.invoke(prompt)
    return response.content.strip()


class GeneratorEvaluator:
    """
    Component-wise evaluator for the RAG Generator using DeepEval and Groq.
    Supports both Controlled-Context and End-to-End evaluations.
    """

    def __init__(
        self,
        judge_model: Optional[GroqJudgeModel] = None,
        faithfulness_threshold: float = FAITHFULNESS_THRESHOLD,
        relevancy_threshold: float = ANSWER_RELEVANCY_THRESHOLD,
        correctness_threshold: float = ANSWER_CORRECTNESS_THRESHOLD,
    ):
        self.judge_model = judge_model or GroqJudgeModel()
        self.faithfulness_threshold = faithfulness_threshold
        self.relevancy_threshold = relevancy_threshold
        self.correctness_threshold = correctness_threshold

        # Initialize DeepEval metrics with Groq judge model
        self.faithfulness_metric = FaithfulnessMetric(
            threshold=self.faithfulness_threshold,
            model=self.judge_model,
            async_mode=False,
            include_reason=True,
        )
        self.relevancy_metric = AnswerRelevancyMetric(
            threshold=self.relevancy_threshold,
            model=self.judge_model,
            async_mode=False,
            include_reason=True,
        )
        self.correctness_metric = GEval(
            name="Answer Correctness",
            criteria="Determine whether the actual output is factually accurate, consistent with, and adequately answers the user question based on the expected reference output.",
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            threshold=self.correctness_threshold,
            model=self.judge_model,
            async_mode=False,
        )

    def evaluate_controlled_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test A: Controlled-Context Generation
        Provides the generator with verified golden context passages.
        """
        query = case["input"]
        expected_output = case["expected_output"]
        reference_passages = case["expected_retrieval_context"]

        # Format verified reference context
        formatted_context = format_context_string(
            reference_passages,
            metadata=case.get("metadata")
        )

        # Generate answer using production generator
        generated_answer = generate_production_answer(query, formatted_context)

        # Build DeepEval LLMTestCase
        test_case = LLMTestCase(
            input=query,
            actual_output=generated_answer,
            expected_output=expected_output,
            retrieval_context=reference_passages if reference_passages else [formatted_context],
            context=reference_passages,
            name=f"{case.get('case_id')}_controlled",
        )

        # 1. Faithfulness
        faith_score = 0.0
        faith_reason = ""
        try:
            faith_score = self.faithfulness_metric.measure(test_case)
            faith_reason = getattr(self.faithfulness_metric, "reason", "")
        except Exception as exc:
            faith_reason = f"Error computing Faithfulness: {exc}"

        time.sleep(EVAL_RATE_LIMIT_DELAY)

        # 2. Answer Relevancy
        rel_score = 0.0
        rel_reason = ""
        try:
            rel_score = self.relevancy_metric.measure(test_case)
            rel_reason = getattr(self.relevancy_metric, "reason", "")
        except Exception as exc:
            rel_reason = f"Error computing Answer Relevancy: {exc}"

        time.sleep(EVAL_RATE_LIMIT_DELAY)

        # 3. Answer Correctness
        corr_score = 0.0
        corr_reason = ""
        try:
            corr_score = self.correctness_metric.measure(test_case)
            corr_reason = getattr(self.correctness_metric, "reason", "")
        except Exception as exc:
            corr_reason = f"Error computing Answer Correctness: {exc}"

        time.sleep(EVAL_RATE_LIMIT_DELAY)

        return {
            "case_id": case.get("case_id"),
            "input": query,
            "expected_output": expected_output,
            "generated_answer": generated_answer,
            "provided_context": reference_passages,
            "faithfulness": {
                "score": float(faith_score),
                "threshold": self.faithfulness_threshold,
                "passed": bool(faith_score >= self.faithfulness_threshold),
                "reason": faith_reason,
            },
            "answer_relevancy": {
                "score": float(rel_score),
                "threshold": self.relevancy_threshold,
                "passed": bool(rel_score >= self.relevancy_threshold),
                "reason": rel_reason,
            },
            "answer_correctness": {
                "score": float(corr_score),
                "threshold": self.correctness_threshold,
                "passed": bool(corr_score >= self.correctness_threshold),
                "reason": corr_reason,
            },
            "is_success": (
                faith_score >= self.faithfulness_threshold
                and rel_score >= self.relevancy_threshold
                and corr_score >= self.correctness_threshold
            ),
        }

    def evaluate_e2e_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test B: End-to-End Generation
        Provides the generator with actual retrieved chunks from the hybrid retriever.
        """
        query = case["input"]
        expected_output = case["expected_output"]

        # Run actual retrieval
        retrieved_docs = retrieve_chunks(query, top_k=5)
        retrieved_passages = [doc["text"] for doc in retrieved_docs if doc.get("text")]
        formatted_context = format_retrieved_context_string(retrieved_docs)

        # Generate answer using production generator
        generated_answer = generate_production_answer(query, formatted_context)

        # Build DeepEval LLMTestCase
        test_case = LLMTestCase(
            input=query,
            actual_output=generated_answer,
            expected_output=expected_output,
            retrieval_context=retrieved_passages if retrieved_passages else ["No retrieved context"],
            context=case.get("expected_retrieval_context"),
            name=f"{case.get('case_id')}_e2e",
        )

        # 1. Faithfulness
        faith_score = 0.0
        faith_reason = ""
        try:
            faith_score = self.faithfulness_metric.measure(test_case)
            faith_reason = getattr(self.faithfulness_metric, "reason", "")
        except Exception as exc:
            faith_reason = f"Error computing Faithfulness: {exc}"

        time.sleep(EVAL_RATE_LIMIT_DELAY)

        # 2. Answer Relevancy
        rel_score = 0.0
        rel_reason = ""
        try:
            rel_score = self.relevancy_metric.measure(test_case)
            rel_reason = getattr(self.relevancy_metric, "reason", "")
        except Exception as exc:
            rel_reason = f"Error computing Answer Relevancy: {exc}"

        time.sleep(EVAL_RATE_LIMIT_DELAY)

        # 3. Answer Correctness
        corr_score = 0.0
        corr_reason = ""
        try:
            corr_score = self.correctness_metric.measure(test_case)
            corr_reason = getattr(self.correctness_metric, "reason", "")
        except Exception as exc:
            corr_reason = f"Error computing Answer Correctness: {exc}"

        time.sleep(EVAL_RATE_LIMIT_DELAY)

        return {
            "case_id": case.get("case_id"),
            "input": query,
            "expected_output": expected_output,
            "generated_answer": generated_answer,
            "retrieved_context": retrieved_passages,
            "faithfulness": {
                "score": float(faith_score),
                "threshold": self.faithfulness_threshold,
                "passed": bool(faith_score >= self.faithfulness_threshold),
                "reason": faith_reason,
            },
            "answer_relevancy": {
                "score": float(rel_score),
                "threshold": self.relevancy_threshold,
                "passed": bool(rel_score >= self.relevancy_threshold),
                "reason": rel_reason,
            },
            "answer_correctness": {
                "score": float(corr_score),
                "threshold": self.correctness_threshold,
                "passed": bool(corr_score >= self.correctness_threshold),
                "reason": corr_reason,
            },
            "is_success": (
                faith_score >= self.faithfulness_threshold
                and rel_score >= self.relevancy_threshold
                and corr_score >= self.correctness_threshold
            ),
        }

    def evaluate(
        self,
        cases: Optional[List[Dict[str, Any]]] = None,
        mode: str = "both",
        max_cases: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run generator evaluation.
        mode can be 'controlled', 'e2e', or 'both'.
        """
        if cases is None:
            loader = EvaluationDatasetLoader()
            cases = loader.get_validated_examples()

        if max_cases is not None and max_cases > 0:
            cases = cases[:max_cases]

        print(f"\n=======================================================")
        print(f"STARTING GENERATOR EVALUATION: {len(cases)} test cases (Mode: {mode})")
        print(f"Production Generator Model: {production_llm.model_name if hasattr(production_llm, 'model_name') else 'Groq'}")
        print(f"Judge Model: {self.judge_model.get_model_name()}")
        print(f"=======================================================\n")

        results = {
            "component": "Generator",
            "production_model": getattr(production_llm, "model_name", "openai/gpt-oss-120b"),
            "judge_model": self.judge_model.get_model_name(),
            "total_cases": len(cases),
            "controlled_context_evaluation": None,
            "end_to_end_evaluation": None,
        }

        # Test A: Controlled-Context
        if mode in ("controlled", "both"):
            print("--- Executing Test A: Controlled-Context Generation ---")
            controlled_cases = []
            f_sum, r_sum, c_sum = 0.0, 0.0, 0.0
            f_pass, r_pass, c_pass = 0, 0, 0

            for index, case in enumerate(cases, start=1):
                print(f"[Controlled {index}/{len(cases)}] Evaluating: '{case['input'][:65]}...'")
                res = self.evaluate_controlled_case(case)
                controlled_cases.append(res)

                fs = res["faithfulness"]["score"]
                rs = res["answer_relevancy"]["score"]
                cs = res["answer_correctness"]["score"]

                f_sum += fs
                r_sum += rs
                c_sum += cs
                if res["faithfulness"]["passed"]: f_pass += 1
                if res["answer_relevancy"]["passed"]: r_pass += 1
                if res["answer_correctness"]["passed"]: c_pass += 1

                print(f"   -> Faithfulness: {fs:.2f} | Relevancy: {rs:.2f} | Correctness: {cs:.2f}")

            n = len(cases)
            results["controlled_context_evaluation"] = {
                "total_cases": n,
                "metrics": {
                    "faithfulness": {
                        "average_score": round(f_sum / n, 4) if n > 0 else 0.0,
                        "pass_rate": round(f_pass / n, 4) if n > 0 else 0.0,
                        "passed_count": f_pass,
                        "threshold": self.faithfulness_threshold,
                    },
                    "answer_relevancy": {
                        "average_score": round(r_sum / n, 4) if n > 0 else 0.0,
                        "pass_rate": round(r_pass / n, 4) if n > 0 else 0.0,
                        "passed_count": r_pass,
                        "threshold": self.relevancy_threshold,
                    },
                    "answer_correctness": {
                        "average_score": round(c_sum / n, 4) if n > 0 else 0.0,
                        "pass_rate": round(c_pass / n, 4) if n > 0 else 0.0,
                        "passed_count": c_pass,
                        "threshold": self.correctness_threshold,
                    },
                },
                "cases": controlled_cases,
            }

        # Test B: End-to-End
        if mode in ("e2e", "both"):
            print("\n--- Executing Test B: End-to-End Generation ---")
            e2e_cases = []
            f_sum, r_sum, c_sum = 0.0, 0.0, 0.0
            f_pass, r_pass, c_pass = 0, 0, 0

            for index, case in enumerate(cases, start=1):
                print(f"[E2E {index}/{len(cases)}] Evaluating: '{case['input'][:65]}...'")
                res = self.evaluate_e2e_case(case)
                e2e_cases.append(res)

                fs = res["faithfulness"]["score"]
                rs = res["answer_relevancy"]["score"]
                cs = res["answer_correctness"]["score"]

                f_sum += fs
                r_sum += rs
                c_sum += cs
                if res["faithfulness"]["passed"]: f_pass += 1
                if res["answer_relevancy"]["passed"]: r_pass += 1
                if res["answer_correctness"]["passed"]: c_pass += 1

                print(f"   -> Faithfulness: {fs:.2f} | Relevancy: {rs:.2f} | Correctness: {cs:.2f}")

            n = len(cases)
            results["end_to_end_evaluation"] = {
                "total_cases": n,
                "metrics": {
                    "faithfulness": {
                        "average_score": round(f_sum / n, 4) if n > 0 else 0.0,
                        "pass_rate": round(f_pass / n, 4) if n > 0 else 0.0,
                        "passed_count": f_pass,
                        "threshold": self.faithfulness_threshold,
                    },
                    "answer_relevancy": {
                        "average_score": round(r_sum / n, 4) if n > 0 else 0.0,
                        "pass_rate": round(r_pass / n, 4) if n > 0 else 0.0,
                        "passed_count": r_pass,
                        "threshold": self.relevancy_threshold,
                    },
                    "answer_correctness": {
                        "average_score": round(c_sum / n, 4) if n > 0 else 0.0,
                        "pass_rate": round(c_pass / n, 4) if n > 0 else 0.0,
                        "passed_count": c_pass,
                        "threshold": self.correctness_threshold,
                    },
                },
                "cases": e2e_cases,
            }

        # Save results
        with open(GENERATOR_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n=======================================================")
        print(f"GENERATOR EVALUATION COMPLETED")
        if results.get("controlled_context_evaluation"):
            ctrl = results["controlled_context_evaluation"]["metrics"]
            print(f"Controlled Faithfulness:   {ctrl['faithfulness']['average_score']:.4f}")
            print(f"Controlled Relevancy:      {ctrl['answer_relevancy']['average_score']:.4f}")
            print(f"Controlled Correctness:    {ctrl['answer_correctness']['average_score']:.4f}")
        if results.get("end_to_end_evaluation"):
            e2e = results["end_to_end_evaluation"]["metrics"]
            print(f"End-to-End Faithfulness:   {e2e['faithfulness']['average_score']:.4f}")
            print(f"End-to-End Relevancy:      {e2e['answer_relevancy']['average_score']:.4f}")
            print(f"End-to-End Correctness:    {e2e['answer_correctness']['average_score']:.4f}")
        print(f"Results saved to: {GENERATOR_RESULTS_FILE.resolve()}")
        print(f"=======================================================\n")

        return results


def run_generator_evaluation(mode: str = "both", max_cases: Optional[int] = None) -> Dict[str, Any]:
    evaluator = GeneratorEvaluator()
    return evaluator.evaluate(mode=mode, max_cases=max_cases)


if __name__ == "__main__":
    run_generator_evaluation(mode="both", max_cases=3)
