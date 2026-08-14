import json
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.utility.bbq_sample import load_jsonl, build_items
from core.config.config import (
    MODEL,
    RESULTS_JSON,
    RESULTS_CSV,
    test_effort_conversion,
    test_ollama_conversion,
    MAX_EXAMPLES,
    TASK_WORKERS,
    PROBE_CUTS,
    CHECKPOINT_INTERVAL,
    PAIRS_DATASET_PATH,
    CLEAN_DATASET_PATH,
)
from core.evaluation.metrics import (
    get_condition_name,
    print_summary,
    save_json,
    save_csv,
)
from core.evaluation.evaluation import build_conditions, process_example_condition


def save_checkpoint(results):
    try:
        RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        save_json(results, RESULTS_JSON)
    except Exception as e:
        print(f"Checkpoint save failed: {e}")


def load_twin_pair_dataset():
    """Load the frozen, opposite-alignment-filtered twin-pair subset (with
    matched ambiguous siblings) - the same canonical dataset models/eval.py
    uses. build_items() is a kept-in-sync local copy
    (core/utility/bbq_sample.py), not a cross-package import."""
    twins = load_jsonl(PAIRS_DATASET_PATH)
    clean = load_jsonl(CLEAN_DATASET_PATH)
    clean_by_uid = {row["uid"]: row for row in clean}
    return build_items(twins, clean_by_uid)


def normalize_dataset(dataset):
    if not isinstance(dataset, list) or len(dataset) == 0:
        return dataset

    first = dataset[0]

    if isinstance(first, dict):
        return dataset

    if isinstance(first, list):
        flattened = []
        for item in dataset:
            if isinstance(item, list):
                flattened.extend(item)
            elif isinstance(item, dict):
                flattened.append(item)
        return flattened

    return dataset


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run model evaluation on dataset.")
    parser.add_argument(
        "--jsonl", 
        type=str, 
        default=None, 
        help="Optional: Path to a single JSONL file to load instead of the default twin-pair dataset."
    )
    args = parser.parse_args()

    test_effort_conversion()
    test_ollama_conversion()

    print("good conversion\n")
    
    # Conditionally load based on arguments
    if args.jsonl:
        print(f"Loading custom JSONL dataset from {args.jsonl}...")
        dataset = load_jsonl(args.jsonl)
    else:
        print("Loading BBQ twin-pair dataset...")
        dataset = load_twin_pair_dataset()
        
    dataset = normalize_dataset(dataset)

    original_count = len(dataset)

    if MAX_EXAMPLES is not None:
        dataset = dataset[:MAX_EXAMPLES]

    print(f"Loaded {original_count} examples.")
    print(f"Using {len(dataset)} examples.")

    if dataset and not isinstance(dataset[0], dict):
        raise TypeError(
            "Dataset did not return dictionaries "
            f"after normalization. First item type: {type(dataset[0])}"
        )

    ambiguous_count = sum(
        1 for example in dataset if example.get("context_condition") == "ambig"
    )
    disambiguated_count = sum(
        1 for example in dataset if example.get("context_condition") == "disambig"
    )

    print(f"Ambiguous examples: {ambiguous_count}")
    print(f"Disambiguated examples: {disambiguated_count}")

    conditions = build_conditions()

    total_tasks = len(dataset) * len(conditions)
    calls_per_task = 1 + PROBE_CUTS
    estimated_calls = total_tasks * calls_per_task

    print(f"\nConditions: {len(conditions)}")
    print(f"Estimated model calls: {estimated_calls}\n")

    tasks = [
        (example, condition)
        for example in dataset
        for condition in conditions
    ]

    all_results = []
    completed_keys = set()

    if RESULTS_JSON.exists() and RESULTS_JSON.stat().st_size > 0:
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []
            print(f"Warning: existing results file {RESULTS_JSON} was unreadable; starting fresh.")

        if isinstance(existing_results, list):
            for result in existing_results:
                if "error" in result:
                    continue
                all_results.append(result)
                
                # Removed 'result["model"]' check here. Since Azure often returns the underlying 
                # deployment name (e.g. gpt-4o-2024) rather than the constant dict key, checking it 
                # strictly prevents proper test resuming.
                completed_keys.add((
                    result["uid"],
                    get_condition_name(result),
                ))

            tasks = [
                (example, condition)
                for example, condition in tasks
                if (
                    example["uid"],
                    get_condition_name(condition),
                ) not in completed_keys
            ]

            print(
                f"Resuming: {len(completed_keys)} task(s) already completed, "
                f"{len(tasks)} remaining."
            )

            # total_tasks above still reflects the full, pre-resume dataset -
            # rebase it to what's actually left to run so the progress
            # percentage and ETA below (which only count newly-completed
            # tasks this run) stay meaningful instead of asymptoting below
            # 100%.
            total_tasks = len(tasks)

    start_time = time.perf_counter()
    tasks_completed = 0

    with ThreadPoolExecutor(max_workers=TASK_WORKERS) as executor:
        futures = {
            executor.submit(
                process_example_condition,
                example,
                condition,
                MODEL,
            ): (example, condition)
            for example, condition in tasks
        }

        for future in as_completed(futures):
            tasks_completed += 1
            example, condition = futures[future]

            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"Worker error for UID={example.get('uid', 'unknown')}: {e}")

            elapsed = time.perf_counter() - start_time
            average_time = elapsed / tasks_completed
            remaining = total_tasks - tasks_completed
            eta = average_time * remaining
            
            # Avoid division by zero if total_tasks is somehow 0
            progress = (tasks_completed / total_tasks * 100) if total_tasks > 0 else 100

            print(
                f"Progress: {tasks_completed}/{total_tasks} "
                f"({progress:.1f}%) | elapsed {elapsed / 60:.1f}m | ETA {eta / 60:.1f}m"
            )

            if tasks_completed % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(all_results)

    total_elapsed = time.perf_counter() - start_time

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    save_json(all_results, RESULTS_JSON)
    save_csv(all_results, RESULTS_CSV)

    print(f"\nFinished processing {len(all_results)} results.")
    print(f"Total runtime: {total_elapsed / 60:.2f} minutes")
    print(f"JSON saved to: {RESULTS_JSON}")
    print(f"CSV saved to: {RESULTS_CSV}\n")

    print_summary(all_results)


if __name__ == "__main__":
    main()