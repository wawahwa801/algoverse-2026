

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.utility.bbq_sample import load_sample, SWEEP_PAIRS_PER_CATEGORY
from core.clients.olllama_client import Qwen3Client
from core.utility.util import format_prompt, get_answer_metadata, parse_answer
from core.evaluation.evaluation import build_conditions
from core.evaluation.metrics import get_condition_name

MODEL = "qwen3.5:9b"
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "qwen_local_sweep.jsonl"

CONCURRENCY = int(os.environ.get("QWEN_SWEEP_CONCURRENCY", "2"))


def load_seen_keys():
    seen = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    seen.add((row["uid"], row["condition_name"]))
    return seen


def build_result(example, condition, response, model_answer, metadata, elapsed):
    correct_answer = int(example["correct_answer"])
    return {
        "uid": example["uid"],
        "twin_side": example.get("twin_side"),
        "twin_partner_uid": example.get("twin_partner_uid"),
        "category": example["category"],
        "context_condition": example["context_condition"],
        "model_key": "qwen3.5-9b",
        "model_id": MODEL,
        "condition_name": get_condition_name(condition),
        "effort": condition["effort"],
        "max_tokens": condition["max_tokens"],
        "prompt_control": condition["prompt_control"],
        "model_answer": model_answer,
        "correct_answer": correct_answer,
        "is_correct": model_answer == correct_answer,
        "selected_unknown": model_answer == metadata["unknown_index"],
        "selected_stereotype": model_answer == metadata["stereotype_index"],
        "selected_anti_stereotype": model_answer == metadata["anti_stereotype_index"],
        "thinking_chars": response.thinking_chars,
        "content": response.content,
        "latency_seconds": elapsed,
    }


def run_all(client, items, conditions):
    seen_keys = load_seen_keys()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(RESULTS_PATH, "a", encoding="utf-8")
    write_lock = Lock()

    jobs = [
        (example, condition)
        for example in items
        for condition in conditions
        if (example["uid"], get_condition_name(condition)) not in seen_keys
    ]
    total = len(items) * len(conditions)
    print(f"Total calls: {total} ({len(seen_keys)} already recorded, {len(jobs)} remaining)")

    done_count = 0
    error_count = 0

    def run_one(example, condition):
        prompt = format_prompt(example, prompt_control=condition["prompt_control"])
        metadata = get_answer_metadata(example)
        valid_indices = [int(i) for i in example["answers"].keys()]

        start = time.perf_counter()
        response = client.ask(
            prompt,
            effort=condition["effort"],
            max_tokens=condition["max_tokens"],
        )
        elapsed = time.perf_counter() - start

        model_answer = parse_answer(response.content, valid_indices)
        return build_result(example, condition, response, model_answer, metadata, elapsed)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {
            pool.submit(run_one, example, condition): (example, condition)
            for example, condition in jobs
        }

        for i, future in enumerate(as_completed(futures), start=1):
            example, condition = futures[future]
            try:
                result = future.result()
            except Exception as e:
                error_count += 1
                print(f"ERROR {example['uid']} {get_condition_name(condition)}: {e}")
                continue

            with write_lock:
                out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_file.flush()
            done_count += 1

            if i % 100 == 0:
                print(f"[{i}/{len(jobs)}] processed  (written={done_count}  errors={error_count})")

    out_file.close()
    print(f"\nDone. Written this run: {done_count}  Errors this run: {error_count}")
    print(f"Results: {RESULTS_PATH}")


def main():
    sampled_twins, _sampled_singles, items = load_sample(
        pairs_per_category=SWEEP_PAIRS_PER_CATEGORY,
    )
    categories = sorted({row["category"] for row in sampled_twins})
    print(f"Sampled {len(sampled_twins)} twin pairs across {len(categories)} categories")
    print(f"Built {len(items)} items")

    conditions = build_conditions()
    total_calls = len(conditions) * len(items)
    print(f"{len(conditions)} conditions x {len(items)} items = {total_calls} calls, concurrency={CONCURRENCY}")

    host = os.environ.get("OLLAMA_HOST")
    client = Qwen3Client(model=MODEL, host=host)
    if host:
        print(f"Using remote Ollama host: {host}")

    run_all(client, items, conditions)


if __name__ == "__main__":
    main()
