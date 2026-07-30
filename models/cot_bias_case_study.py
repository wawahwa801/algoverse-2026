import json
import random
import re
from pathlib import Path

from bbq_sample import load_jsonl, CLEAN_PATH, TWINS_PATH
from client import Qwen3Client
from eval import format_prompt, get_answer_metadata, parse_answer

MODEL = "qwen3.5:9b"
EFFORTS = ["low", "medium", "high"]
SAMPLE_PAIRS = 5
SEED = 0
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "cot_bias_case_study.jsonl"


def sample_pairs(twins, n, seed):
    rng = random.Random(seed)
    twins = twins[:]
    rng.shuffle(twins)
    return twins[:n]


def keyword_terms(example, index):
    """Terms that would indicate the reasoning is invoking this answer's
    demographic group - the answer's own noun phrase, plus its group
    label (e.g. "old"/"nonOld"), skipping "unknown"."""
    if index is None:
        return []
    answer_text = example["answers"].get(str(index), "")
    group_label = example["answer_groups"].get(str(index), "")
    terms = set()
    cleaned = re.sub(r"^(the|a|an)\s+", "", answer_text.strip(), flags=re.I)
    if cleaned:
        terms.add(cleaned.lower())
    if group_label and group_label.lower() != "unknown":
        terms.add(group_label.lower())
    return sorted(terms)


def find_mentions(text, terms):
    mentions = []
    for term in terms:
        for match in re.finditer(re.escape(term), text, re.I):
            mentions.append({
                "term": term,
                "pct_through_reasoning": round(100 * match.start() / max(len(text), 1), 1),
            })
    return sorted(mentions, key=lambda m: m["pct_through_reasoning"])


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


def build_items(pairs, clean_by_uid):
    items = []
    for pair in pairs:
        for side, uid_key, context_key in (
            ("a", "twin_a_uid", "twin_a_context"),
            ("b", "twin_b_uid", "twin_b_context"),
        ):
            base = clean_by_uid.get(pair[uid_key])
            if base is None:
                continue
            example = dict(base)
            example["context"] = pair[context_key]
            example["twin_side"] = side
            example["twin_partner_uid"] = pair["twin_b_uid" if side == "a" else "twin_a_uid"]
            items.append(example)
    return items


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
    twins = load_jsonl(TWINS_PATH)

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
