"""Run the same reasoning-budget sweep as run_openrouter_sweep.py, but
against qwen3.5:9b via Ollama - either locally or on a remote GPU (e.g. a
Lightning AI T4 instance) reachable over HTTP. Same sample
(bbq_sample.load_sample), same 8 conditions, same result schema (minus
model_id/twin fields staying uid-based) so runs can be concatenated for
analysis.

Concurrency is bounded by CONCURRENCY, which should match (or stay under)
the server's OLLAMA_NUM_PARALLEL setting - requesting more concurrent
generations than the server can actually run in parallel just queues them
there instead of here, with no speed benefit.

Point at a remote GPU by setting OLLAMA_HOST, e.g.:
    OLLAMA_HOST=http://<lightning-ai-instance>:11434 py run_qwen_local_sweep.py
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from bbq_sample import load_sample
from client import Qwen3Client
from eval import format_prompt, get_answer_metadata, parse_answer, build_conditions, get_condition_name

MODEL = "qwen3.5:9b"
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "qwen_local_sweep.jsonl"

# Match this to the server's OLLAMA_NUM_PARALLEL. Requesting more than the
# server can actually run concurrently just queues extra requests there.
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
        futures = {pool.submit(run_one, example, condition): (example, condition) for example, condition in jobs}

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
    sampled_twins, items = load_sample()
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
