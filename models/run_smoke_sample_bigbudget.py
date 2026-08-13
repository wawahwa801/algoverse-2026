"""Exploratory test: does raising the budget-forced token cap (2048/4096/8192,
well beyond config.py's official 128/512/1024) eliminate the invalid-answer
problem found in the first smoke test? This does NOT change config.BUDGETS -
it's a one-off comparison, kept in its own output files."""

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import eval
from config import MODEL
from eval import load_twin_pair_dataset, normalize_dataset, process_example_condition
from metrics import save_json, save_csv

# This test measures whether the model finishes naturally at each budget -
# the forced-answer fallback (models/eval.py::ENABLE_FORCED_ANSWER) would
# mask exactly that signal by silently recovering answers the model didn't
# actually state in time, so it's off for this run specifically.
eval.ENABLE_FORCED_ANSWER = False

SAMPLE_PAIRS = 5
ITEMS_PER_PAIR = 3  # a, b, ambig
SAMPLE_ITEMS = SAMPLE_PAIRS * ITEMS_PER_PAIR
TASK_WORKERS = 4

TEST_BUDGETS = [2048, 4096, 8192]

RESULTS_JSON = Path("results/bbq_results_qwen3.5-9b_bigbudget.json")
RESULTS_CSV = Path("results/bbq_results_qwen3.5-9b_bigbudget.csv")


def build_test_conditions():
    return [
        {
            "control_type": "budget",
            "effort": "medium",
            "max_tokens": max_tokens,
            "prompt_control": None,
            "think": True,
        }
        for max_tokens in TEST_BUDGETS
    ]


def main():
    dataset = normalize_dataset(load_twin_pair_dataset())[:SAMPLE_ITEMS]
    conditions = build_test_conditions()

    tasks = [(example, condition) for example in dataset for condition in conditions]
    print(f"Sample: {SAMPLE_PAIRS} pairs ({len(dataset)} items) x {len(conditions)} conditions = {len(tasks)} tasks")
    print(f"Budgets under test: {TEST_BUDGETS}")

    all_results = []
    start_time = time.perf_counter()
    completed = 0

    with ThreadPoolExecutor(max_workers=TASK_WORKERS) as executor:
        futures = {
            executor.submit(process_example_condition, example, condition, MODEL): (example, condition)
            for example, condition in tasks
        }
        for future in as_completed(futures):
            completed += 1
            example, condition = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                err = result.get("error")
                status = f"ERROR: {err}" if err else f"budget={condition['max_tokens']} answer={result.get('model_answer')} thinking_chars={result.get('thinking_chars')}"
                print(f"[{completed}/{len(tasks)}] {example['uid']} ({example.get('twin_side')}) {status}")
            except Exception as e:
                print(f"[{completed}/{len(tasks)}] WORKER ERROR for {example['uid']}: {e}")

    elapsed = time.perf_counter() - start_time
    print(f"\nDone: {len(all_results)} results in {elapsed/60:.1f} minutes")

    save_json(all_results, RESULTS_JSON)
    save_csv(all_results, RESULTS_CSV)
    print(f"Wrote: {RESULTS_JSON}")
    print(f"Wrote: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
