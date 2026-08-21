"""Small real sample run through the fixed core pipeline, saving output to disk.

Kept separate from the main bbq_results file so smoke-test runs don't
overwrite full sweep results.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config.config import MODEL
from core.main import load_twin_pair_dataset, normalize_dataset
from core.evaluation.evaluation import build_conditions, process_example_condition
from core.evaluation.metrics import save_json, save_csv

SAMPLE_PAIRS = 5
ITEMS_PER_PAIR = 3  
SAMPLE_ITEMS = SAMPLE_PAIRS * ITEMS_PER_PAIR
TASK_WORKERS = 4

RESULTS_JSON = Path(__file__).resolve().parents[1] / "results" / "bbq_results_qwen3.5-9b_smoketest.json"
RESULTS_CSV = Path(__file__).resolve().parents[1] / "results" / "bbq_results_qwen3.5-9b_smoketest.csv"


def main():
    dataset = normalize_dataset(load_twin_pair_dataset())[:SAMPLE_ITEMS]
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
                status = (
                    f"ERROR: {err}"
                    if err
                    else f"answer={result.get('model_answer')} probe={result.get('probe_final_answer')}"
                )
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
