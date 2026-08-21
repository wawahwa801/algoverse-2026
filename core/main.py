import json
import os
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
from core.evaluation.evaluation import (
    build_conditions,
    process_example_condition,
    ENABLE_FLIP_RATE_EVAL,
    FLIP_RATE_K,
)


def _result_key(result_or_example, condition=None):

    if condition is None:
        return (
            str(result_or_example.get("uid")),
            get_condition_name(result_or_example),
        )
    return (
        str(result_or_example["uid"]),
        get_condition_name(condition),
    )


def save_checkpoint(results):

    try:
        RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        temp_path = RESULTS_JSON.with_suffix(RESULTS_JSON.suffix + ".tmp")
        save_json(results, temp_path)
        os.replace(temp_path, RESULTS_JSON)
    except Exception as e:
        print(f"Checkpoint save failed: {type(e).__name__}: {e}", flush=True)


def load_twin_pair_dataset():

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


def _same_model(a: str | None, b: str | None) -> bool:
    return bool(a and b and a.strip().lower() == b.strip().lower())


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation on dataset.")
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Optional: Path to a single JSONL file to load instead of the default twin-pair dataset.",
    )
    args = parser.parse_args()

    test_effort_conversion()
    test_ollama_conversion()

    print("good conversion\n")

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

    conditions = build_conditions(model_name=MODEL)

    total_tasks = len(dataset) * len(conditions)
    calls_per_task = 1 + PROBE_CUTS + (FLIP_RATE_K if ENABLE_FLIP_RATE_EVAL else 0)
    estimated_calls = total_tasks * calls_per_task

    print(f"\nConditions: {len(conditions)}")
    print(f"Estimated model calls: {estimated_calls}\n")

    tasks = [
        (example, condition)
        for example in dataset
        for condition in conditions
    ]


    result_by_key = {}

    if RESULTS_JSON.exists() and RESULTS_JSON.stat().st_size > 0:
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []
            print(
                f"Warning: existing results file {RESULTS_JSON} was unreadable; "
                "starting fresh."
            )

        if isinstance(existing_results, list):
            for result in existing_results:
                if result.get("error") or result.get("status") == "error":
                    continue

                result_model = result.get("experiment_model", result.get("model"))
                if not _same_model(result_model, MODEL):
                    continue

                key = _result_key(result)
                result_by_key[key] = result

            completed_keys = set(result_by_key)

            tasks = [
                (example, condition)
                for example, condition in tasks
                if _result_key(example, condition) not in completed_keys
            ]

            print(
                f"Resuming: {len(completed_keys)} unique task(s) already completed, "
                f"{len(tasks)} remaining."
            )
        else:
            print("Warning: existing JSON was not a list; starting fresh.")

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
                if result.get("error") or result.get("status") == "error":
                    print(
                        f"Task failed after retries: UID={example.get('uid')} "
                        f"condition={get_condition_name(condition)} "
                        f"error={result.get('error')}",
                        flush=True,
                    )
                else:
                    result_by_key[_result_key(result)] = result
            except Exception as e:
                print(
                    f"Worker error for UID={example.get('uid', 'unknown')} "
                    f"condition={get_condition_name(condition)} "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

            elapsed = time.perf_counter() - start_time
            average_time = elapsed / tasks_completed
            remaining = total_tasks - tasks_completed
            eta = average_time * remaining
            progress = (tasks_completed / total_tasks * 100) if total_tasks > 0 else 100

            print(
                f"Progress: {tasks_completed}/{total_tasks} "
                f"({progress:.1f}%) | elapsed {elapsed / 60:.1f}m | ETA {eta / 60:.1f}m",
                flush=True,
            )

            if tasks_completed % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(list(result_by_key.values()))

    total_elapsed = time.perf_counter() - start_time
    all_results = list(result_by_key.values())

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    save_checkpoint(all_results)
    save_csv(all_results, RESULTS_CSV)

    print(f"\nFinished processing {len(all_results)} unique successful results.")
    print(f"Total runtime: {total_elapsed / 60:.2f} minutes")
    print(f"JSON saved to: {RESULTS_JSON}")
    print(f"CSV saved to: {RESULTS_CSV}\n")

    print_summary(all_results)


if __name__ == "__main__":
    main()