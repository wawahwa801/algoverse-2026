"""Small real sample run through the actual fixed pipeline (post context-window
fix, probe-indexing fix, dataset rebuild, prompt-conditions removal), saving
real output to disk - unlike this session's ad-hoc verification calls, which
only printed to console. Kept separate from bbq_results_qwen3.5-9b.json (that
file predates today's fixes and its probe data is known-unreliable)."""

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import MODEL
from eval import load_twin_pair_dataset, normalize_dataset, build_conditions, process_example_condition
from metrics import save_json, save_csv

SAMPLE_PAIRS = 5
ITEMS_PER_PAIR = 3  # a, b, ambig
SAMPLE_ITEMS = SAMPLE_PAIRS * ITEMS_PER_PAIR
TASK_WORKERS = 4  # bumped from eval.py's default of 2 for this time-boxed run

RESULTS_JSON = Path("results/bbq_results_qwen3.5-9b_smoketest.json")
RESULTS_CSV = Path("results/bbq_results_qwen3.5-9b_smoketest.csv")


def main():
    dataset = normalize_dataset(load_twin_pair_dataset())[:SAMPLE_ITEMS]
    # Budget-forced conditions only for this time-boxed run - bounded to
    # 128/512/1024 tokens, so no risk of a slow uncapped native-effort
    # outlier blowing the time budget.
    conditions = [c for c in build_conditions() if c["control_type"] == "budget"]

    tasks = [(example, condition) for example in dataset for condition in conditions]
    print(f"Sample: {SAMPLE_PAIRS} pairs ({len(dataset)} items) x {len(conditions)} conditions = {len(tasks)} tasks")

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
                status = f"ERROR: {err}" if err else f"answer={result.get('model_answer')} probe={result.get('probe_final_answer')}"
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
