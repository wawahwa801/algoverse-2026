import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.clients.olllama_client import Qwen3Client
from core.config.config import DATASET_PATH, PROJECT_ROOT
from core.utility.util import load_bbq, format_prompt, get_answer_metadata, parse_answer


MODEL = "gpt-oss:20b"

RESULTS_PATH = PROJECT_ROOT / "core" / "results" / "gptoss_effort_results.json"

EFFORTS = ["low", "medium", "high"]


def run_effort_sweep(client, dataset):
    results = []

    for example in dataset:
        prompt = format_prompt(example)
        metadata = get_answer_metadata(example)
        valid_indices = [int(index) for index in example["answers"].keys()]

        for effort in EFFORTS:
            start = time.perf_counter()
            response = client.ask(prompt, effort=effort)
            elapsed = time.perf_counter() - start

            model_answer = parse_answer(response.content, valid_indices)

            result = {
                "uid": example["uid"],
                "effort": effort,
                "model_answer": model_answer,
                "correct_answer": int(example["correct_answer"]),
                "unknown_index": metadata["unknown_index"],
                "stereotype_index": metadata["stereotype_index"],
                "content": response.content,
                "thinking_chars": response.thinking_chars,
                "latency_seconds": elapsed,
            }
            results.append(result)

            print(
                f"{example['uid']} | effort={effort:6s} | "
                f"thinking_chars={response.thinking_chars:5d} | "
                f"latency={elapsed:.2f}s"
            )

    return results


def summarize(results):
    print()
    print("=== Average thinking length by effort ===")
    for effort in EFFORTS:
        subset = [r for r in results if r["effort"] == effort]
        if not subset:
            print(f"{effort:6s}: no results")
            continue
        avg_chars = sum(r["thinking_chars"] for r in subset) / len(subset)
        avg_latency = sum(r["latency_seconds"] for r in subset) / len(subset)
        print(
            f"{effort:6s}: avg thinking_chars={avg_chars:.1f}  "
            f"avg latency={avg_latency:.2f}s  n={len(subset)}"
        )


def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_bbq(DATASET_PATH)
    print(f"Loaded {len(dataset)} examples.")

    client = Qwen3Client(model=MODEL)

    results = run_effort_sweep(client, dataset)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summarize(results)
    print(f"\nWrote: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
