"""Run the reasoning-budget sweep against hosted open-weight models via OpenRouter.

Requires OPENROUTER_API_KEY to be set in the environment. Writes results
incrementally to core/results/openrouter_sweep.jsonl and skips any
(uid, model, condition) triple already present there, so an interrupted run
can resume without re-paying for completed calls.
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.utility.bbq_sample import load_sample, SWEEP_PAIRS_PER_CATEGORY
from core.clients.openrouter_client import OpenRouterClient
from core.utility.util import format_prompt, get_answer_metadata, parse_answer
from core.evaluation.evaluation import build_conditions
from core.evaluation.metrics import get_condition_name

RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "openrouter_sweep.jsonl"

MODELS = {
    "glm-5.2": "z-ai/glm-5.2",
    "kimi-k3": "moonshotai/kimi-k3",
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
}

CONCURRENCY_PER_MODEL = 10


def load_seen_keys():
    seen = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    seen.add((row["uid"], row["model_key"], row["condition_name"]))
    return seen


def build_result(example, model_key, model_id, condition, response, model_answer, metadata):
    correct_answer = int(example["correct_answer"])
    return {
        "uid": example["uid"],
        "twin_side": example.get("twin_side"),
        "twin_partner_uid": example.get("twin_partner_uid"),
        "category": example["category"],
        "context_condition": example["context_condition"],
        "model_key": model_key,
        "model_id": model_id,
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
        "reasoning_chars": response.reasoning_chars,
        "content": response.content,
    }


async def run_all(items, conditions):
    seen_keys = load_seen_keys()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(RESULTS_PATH, "a", encoding="utf-8")

    clients = {key: OpenRouterClient() for key in MODELS}
    semaphores = {key: asyncio.Semaphore(CONCURRENCY_PER_MODEL) for key in MODELS}
    done_count = 0
    error_count = 0

    async def run_one(model_key, model_id, example, condition):
        nonlocal done_count, error_count
        condition_name = get_condition_name(condition)
        key = (example["uid"], model_key, condition_name)
        if key in seen_keys:
            return

        prompt = format_prompt(example, prompt_control=condition["prompt_control"])
        metadata = get_answer_metadata(example)
        valid_indices = [int(i) for i in example["answers"].keys()]

        async with semaphores[model_key]:
            try:
                response = await clients[model_key].ask(
                    model_id,
                    prompt,
                    effort=condition["effort"],
                    max_tokens=condition["max_tokens"],
                )
            except Exception as e:
                error_count += 1
                print(f"ERROR {model_key} {example['uid']} {condition_name}: {e}")
                return

        model_answer = parse_answer(response.content, valid_indices)
        result = build_result(example, model_key, model_id, condition, response, model_answer, metadata)
        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_file.flush()
        done_count += 1

    tasks = [
        run_one(model_key, model_id, example, condition)
        for model_key, model_id in MODELS.items()
        for example in items
        for condition in conditions
    ]

    total = len(tasks)
    already_done = len(seen_keys)
    print(f"Total calls: {total} ({already_done} already recorded, resuming)")

    for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
        await coro
        if i % 200 == 0:
            print(f"[{i}/{total}] processed  (written={done_count}  errors={error_count})")

    out_file.close()
    for client in clients.values():
        await client.aclose()

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
    total_calls = len(conditions) * len(items) * len(MODELS)
    print(f"{len(conditions)} conditions x {len(items)} items x {len(MODELS)} models = {total_calls} calls")

    asyncio.run(run_all(items, conditions))


if __name__ == "__main__":
    main()
