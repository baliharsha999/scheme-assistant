import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from deepeval.test_case import LLMTestCase
from evals.config import GOLDEN_DATASET_FILE


class EvaluationDatasetLoader:
    """
    Loads, validates, and prepares golden test cases for retriever
    and generator evaluation, preserving document provenance.
    """

    def __init__(self, dataset_path: Optional[Path] = None):
        self.dataset_path = dataset_path or GOLDEN_DATASET_FILE

    def load_raw_dataset(self) -> Dict[str, Any]:
        """Load raw JSON file from disk."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found at {self.dataset_path.resolve()}"
            )
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def get_validated_examples(self) -> List[Dict[str, Any]]:
        """
        Validate and normalize golden examples from the dataset file.
        Each example must contain a valid question and expected answer.
        """
        raw_data = self.load_raw_dataset()
        raw_examples = raw_data.get("examples", [])
        if not raw_examples and isinstance(raw_data, list):
            raw_examples = raw_data

        validated = []
        for idx, ex in enumerate(raw_examples):
            question = str(ex.get("input") or ex.get("question") or "").strip()
            expected_output = str(ex.get("expected_output") or ex.get("answer") or "").strip()
            
            # Reference passage context
            expected_context = (
                ex.get("expected_retrieval_context")
                or ex.get("retrieval_context")
                or []
            )
            if isinstance(expected_context, str):
                expected_context = [expected_context]

            if not question:
                print(f"[DatasetLoader] Skipping example #{idx + 1}: missing input/question.")
                continue

            if not expected_output:
                print(f"[DatasetLoader] Skipping example #{idx + 1}: missing expected_output.")
                continue

            source = ex.get("source", "Unknown")
            page = ex.get("page")
            chunk_id = ex.get("chunk_id", f"case_{idx + 1}")

            validated.append({
                "case_id": f"eval_case_{idx + 1}",
                "input": question,
                "expected_output": expected_output,
                "expected_retrieval_context": expected_context,
                "source": source,
                "page": page,
                "chunk_id": chunk_id,
                "metadata": {
                    "source": source,
                    "page": page,
                    "chunk_id": chunk_id,
                }
            })

        print(f"[DatasetLoader] Successfully loaded {len(validated)} validated test cases from {self.dataset_path.name}")
        return validated

    def create_deepeval_test_cases(
        self,
        examples: Optional[List[Dict[str, Any]]] = None,
        retrieval_contexts_map: Optional[Dict[str, List[str]]] = None,
        actual_outputs_map: Optional[Dict[str, str]] = None,
    ) -> List[LLMTestCase]:
        """
        Convert raw evaluation cases into DeepEval LLMTestCase objects.
        """
        if examples is None:
            examples = self.get_validated_examples()

        test_cases = []
        for ex in examples:
            cid = ex["case_id"]
            retrieval_context = (
                retrieval_contexts_map.get(cid)
                if retrieval_contexts_map else None
            )
            actual_output = (
                actual_outputs_map.get(cid)
                if actual_outputs_map else None
            )

            tc = LLMTestCase(
                input=ex["input"],
                actual_output=actual_output or "N/A",
                expected_output=ex["expected_output"],
                context=ex["expected_retrieval_context"],
                retrieval_context=retrieval_context,
                metadata=ex["metadata"],
                name=cid,
            )
            test_cases.append(tc)

        return test_cases
