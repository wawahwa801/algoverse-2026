
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.config.config as config
from core.config.config import MODEL
from core.main import load_twin_pair_dataset, normalize_dataset
from core.evaluation.evaluation import process_example_condition
from core.evaluation.metrics import save_json, save_csv


config.ENABLE_FORCED_ANSWER = False

SAMPLE_PAIRS = 5
ITEMS_PER_PAIR = 3
SAMPLE_ITEMS = SAMPLE_PAIRS * ITEMS_PER_PAIR
TASK_WORKERS = 4

TEST_BUDGETS = [2048, 4096, 8192]

RESULTS_JSON = Path(__file__).resolve().parents[1] / "results" / "bbq_results_qwen3.5-9b_bigbudget.json"
RESULTS_CSV = Path(__file__).resolve().parents[1] / "results" / "bbq_results_qwen3.5-9b_bigbudget.csv"


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
                status = (
                    f"ERROR: {err}"
                    if err
                    else (
                        f"budget={condition['max_tokens']} "
                        f"answer={result.get('model_answer')} "
                        f"thinking_chars={result.get('thinking_chars')}"
                    )
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
