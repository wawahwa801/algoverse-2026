"""Measure real wall-clock cost of eval.py's actual per-(item, condition) task
(evaluate_example + full-chain regen + PROBE_CUTS probe calls) at a couple of
TASK_WORKERS settings, on whatever machine/Ollama instance this is run from
(e.g. run this directly on the Lightning AI T4, not pointed at it remotely -
eval.py's get_client() always targets localhost).

This replaces guessing at pairs/day with a real measured number: run it, read
the printed tasks/hour and pairs/day at each concurrency level, and use that
to decide TASK_WORKERS for the real sweep.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config.config import MODEL
from core.evaluation.evaluation import build_conditions, process_example_condition
from core.main import normalize_dataset, load_twin_pair_dataset

TASKS_PER_LEVEL = 12
CONCURRENCY_LEVELS = [2, 4]
CALLS_PER_TASK = 5  # 1 evaluate_example (reused as full chain) + PROBE_CUTS(4) probes


def run_level(tasks, concurrency):
    start = time.perf_counter()
    completed = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(process_example_condition, example, condition, MODEL) for example, condition in tasks]
        for future in as_completed(futures):
            result = future.result()
            if "error" in result:
                errors += 1
            completed += 1

    elapsed = time.perf_counter() - start
    return elapsed, completed, errors


def main():
    dataset = normalize_dataset(load_twin_pair_dataset())
    conditions = build_conditions()

    print(f"Model: {MODEL}")
    print(f"Calls per task (assumed): {CALLS_PER_TASK}\n")

    for concurrency in CONCURRENCY_LEVELS:
        tasks = [(dataset[i % len(dataset)], conditions[i % len(conditions)]) for i in range(TASKS_PER_LEVEL)]

        print(f"--- concurrency={concurrency}: running {TASKS_PER_LEVEL} real tasks ---")
        elapsed, completed, errors = run_level(tasks, concurrency)

        seconds_per_task = elapsed / completed
        tasks_per_hour = 3600 / seconds_per_task
        tasks_per_day = tasks_per_hour * 24
        items_per_day = tasks_per_day / len(conditions)
        pairs_per_day = items_per_day / 2
        calls_per_hour = tasks_per_hour * CALLS_PER_TASK

        print(f"  completed={completed}  errors={errors}  elapsed={elapsed:.1f}s")
        print(f"  seconds/task={seconds_per_task:.1f}")
        print(f"  => tasks/day={tasks_per_day:.0f}  calls/hour={calls_per_hour:.0f}")
        print(f"  => items/day={items_per_day:.0f} (at {len(conditions)} conditions/item)")
        print(f"  => pairs/day={pairs_per_day:.0f} (2 items/pair)\n")

    print("These extrapolate a small sample over 24h - treat as an order-of-magnitude")
    print("estimate, not a guarantee. Re-run with more TASKS_PER_LEVEL for a tighter number.")


if __name__ == "__main__":
    main()
