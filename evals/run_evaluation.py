import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.dataset_loader import EvaluationDatasetLoader
from evals.evaluate_retriever import RetrieverEvaluator
from evals.evaluate_generator import GeneratorEvaluator
from evals.report_generator import build_and_save_all_reports


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Comprehensive Evaluation Suite for Government Scheme RAG Pipeline using DeepEval & Groq"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the complete evaluation suite (retriever + generator controlled & e2e + reports)",
    )
    parser.add_argument(
        "--retriever",
        action="store_true",
        help="Run Retriever evaluation (Contextual Recall, Contextual Precision, Hit Rate@k)",
    )
    parser.add_argument(
        "--generator",
        action="store_true",
        help="Run Generator evaluation (Faithfulness, Relevancy, Correctness)",
    )
    parser.add_argument(
        "--mode",
        choices=["controlled", "e2e", "both"],
        default="both",
        help="Generator evaluation mode: 'controlled' (verified context), 'e2e' (retrieved context), or 'both'",
    )
    parser.add_argument(
        "--sample-size",
        "-n",
        type=int,
        default=None,
        help="Limit evaluation to the first N test cases (useful for quick verification)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks for retriever evaluation (default: 5)",
    )
    parser.add_argument(
        "--generate-dataset",
        action="store_true",
        help="Trigger synthetic dataset generation pipeline",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild Markdown reports from existing result JSON files without re-running models",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.generate_dataset:
        print("\n--- Running Dataset Generation ---")
        from evals.generate_dataset import main as run_dataset_gen
        run_dataset_gen()
        return

    if args.report_only:
        print("\n--- Generating Reports from Existing Results ---")
        build_and_save_all_reports()
        return

    # If neither --all, --retriever, nor --generator specified, default to --all with sample if not specified
    run_all = args.all or (not args.retriever and not args.generator)
    run_retriever = run_all or args.retriever
    run_generator = run_all or args.generator

    # Load validated dataset
    loader = EvaluationDatasetLoader()
    cases = loader.get_validated_examples()

    if args.sample_size:
        cases = cases[:args.sample_size]
        print(f"[Runner] Evaluating sample of {len(cases)} test cases.")
    else:
        print(f"[Runner] Evaluating full dataset of {len(cases)} test cases.")

    # 1. Retriever Evaluation
    if run_retriever:
        print("\n>>> Phase 1: Evaluating Retriever Component <<<")
        retriever_evaluator = RetrieverEvaluator(top_k=args.top_k)
        retriever_evaluator.evaluate(cases=cases)

    # 2. Generator Evaluation
    if run_generator:
        print(f"\n>>> Phase 2: Evaluating Generator Component (Mode: {args.mode}) <<<")
        generator_evaluator = GeneratorEvaluator()
        generator_evaluator.evaluate(cases=cases, mode=args.mode)

    # 3. Report Generation
    print("\n>>> Phase 3: Building Evaluation Reports <<<")
    build_and_save_all_reports()

    print("\nEvaluation pipeline finished successfully.")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print("\n\n[FATAL ERROR IN RUNNER]:", e)
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
