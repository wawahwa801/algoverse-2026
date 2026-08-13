import json
import random
from pathlib import Path

from core.utility.bbq_sample import load_jsonl, build_items, CLEAN_PATH
from core.config.config import PAIRS_DATASET_PATH
from core.clients.olllama_client import Qwen3Client
from core.utility.util import format_prompt, get_answer_metadata, parse_answer
from core.evaluation.metrics import keyword_terms, find_mentions

MODEL = "qwen3.5:9b"
EFFORTS = ["low", "medium", "high"]
SAMPLE_PAIRS = 5
SEED = 0
# Draw from the same frozen, opposite-alignment subset the main pipeline
# uses (core/config/config.py::PAIRS_DATASET_PATH) rather than the raw twin
# pool - keeps this a small, separate case study (per
# feedback_points_pratham.md, Mentor 1) while guaranteeing every example
# pair actually tests commitment asymmetry, and picks up each pair's
# ambiguous sibling for free via bbq_sample.build_items.
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "cot_bias_case_study.jsonl"


def sample_pairs(twins, n, seed):
    rng = random.Random(seed)
    twins = twins[:]
    rng.shuffle(twins)
    return twins[:n]


def run_item(client, example, effort):
    prompt = format_prompt(example)
    metadata = get_answer_metadata(example)
    valid_indices = [int(i) for i in example["answers"].keys()]

    response = client.ask(prompt, effort=effort)
    model_answer = parse_answer(response.content, valid_indices)
    thinking = response.thinking or ""

    stereotype_terms = keyword_terms(example, metadata["stereotype_index"])
    anti_stereotype_terms = keyword_terms(example, metadata["anti_stereotype_index"])
    stereotype_mentions = find_mentions(thinking, stereotype_terms)
    anti_stereotype_mentions = find_mentions(thinking, anti_stereotype_terms)

    return {
        "uid": example["uid"],
        "twin_side": example.get("twin_side"),
        "twin_partner_uid": example.get("twin_partner_uid"),
        "category": example["category"],
        "effort": effort,
        "model_answer": model_answer,
        "correct_answer": int(example["correct_answer"]),
        "stereotype_index": metadata["stereotype_index"],
        "anti_stereotype_index": metadata["anti_stereotype_index"],
        "selected_stereotype": model_answer == metadata["stereotype_index"],
        "selected_anti_stereotype": model_answer == metadata["anti_stereotype_index"],
        "stereotype_terms": stereotype_terms,
        "anti_stereotype_terms": anti_stereotype_terms,
        "stereotype_mentions": stereotype_mentions,
        "anti_stereotype_mentions": anti_stereotype_mentions,
        "first_stereotype_mention_pct": stereotype_mentions[0]["pct_through_reasoning"] if stereotype_mentions else None,
        "first_anti_stereotype_mention_pct": anti_stereotype_mentions[0]["pct_through_reasoning"] if anti_stereotype_mentions else None,
        "thinking_chars": len(thinking),
        "thinking": thinking,
        "content": response.content,
    }


def print_summary(results):
    print()
    print("=" * 80)
    print("COT BIAS CASE STUDY SUMMARY")
    print("=" * 80)
    for effort in EFFORTS:
        subset = [r for r in results if r["effort"] == effort]
        if not subset:
            continue
        n = len(subset)
        n_mentions_stereotype = sum(1 for r in subset if r["stereotype_mentions"])
        n_selected_stereotype = sum(1 for r in subset if r["selected_stereotype"])
        avg_first_mention = [r["first_stereotype_mention_pct"] for r in subset if r["first_stereotype_mention_pct"] is not None]
        avg_first_mention_str = f"{sum(avg_first_mention)/len(avg_first_mention):.1f}%" if avg_first_mention else "N/A"
        print(f"\neffort={effort}  (n={n})")
        print(f"  reasoning explicitly names the stereotype: {n_mentions_stereotype}/{n}")
        print(f"  final answer matches the stereotype:       {n_selected_stereotype}/{n}")
        print(f"  avg position of first stereotype mention:  {avg_first_mention_str} through the reasoning")


def main():
    clean = load_jsonl(CLEAN_PATH)
    clean_by_uid = {row["uid"]: row for row in clean}
    twins = load_jsonl(PAIRS_DATASET_PATH)

    pairs = sample_pairs(twins, SAMPLE_PAIRS, SEED)
    items = build_items(pairs, clean_by_uid)
    print(f"Sampled {len(pairs)} twin pairs ({len(items)} items) across categories: "
          f"{sorted({p['category'] for p in pairs})}")

    client = Qwen3Client(model=MODEL)

    results = []
    total = len(items) * len(EFFORTS)
    call_index = 0
    for example in items:
        for effort in EFFORTS:
            call_index += 1
            print(f"[{call_index}/{total}] {example['uid']} ({example['twin_side']}) effort={effort}")
            result = run_item(client, example, effort)
            results.append(result)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print_summary(results)
    print(f"\nWrote: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
