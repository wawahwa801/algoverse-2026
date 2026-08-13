"""Shared stratified sampling of BBQ counterfactual twin pairs.

Used by both the OpenRouter sweep (GLM/Kimi/DeepSeek) and the local Qwen
GPU sweep, so the two runs pull from the same sample instead of two
independently-sampled subsets that can't be compared category-by-category.
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "bbq_clean.jsonl"
TWINS_PATH = ROOT / "data" / "bbq_twins.jsonl"

DEFAULT_PAIRS_PER_CATEGORY = 200
DEFAULT_SEED = 0


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_twins(twins, pairs_per_category=DEFAULT_PAIRS_PER_CATEGORY, seed=DEFAULT_SEED):
    by_category = defaultdict(list)
    for row in twins:
        by_category[row["category"]].append(row)

    rng = random.Random(seed)
    sampled = []
    for rows in by_category.values():
        rows = rows[:]
        rng.shuffle(rows)
        sampled.extend(rows[:pairs_per_category])
    return sampled


def build_items(twins, clean_by_uid):
    """Turn each twin pair into two scoreable items: the original question's
    answers/metadata, but with context swapped to the twin's (recombined)
    version. Pairs tagged with an "ambig_uid" (see
    data/build_clean_pairs_subset.py) also contribute their matching
    ambiguous sibling as a third item, so s_AMB is computable alongside
    s_DIS."""
    items = []
    for pair in twins:
        for side, uid_key, context_key in (
            ("a", "twin_a_uid", "twin_a_context"),
            ("b", "twin_b_uid", "twin_b_context"),
        ):
            uid = pair[uid_key]
            base = clean_by_uid.get(uid)
            if base is None:
                continue
            example = dict(base)
            example["context"] = pair[context_key]
            example["twin_side"] = side
            example["twin_partner_uid"] = (
                pair["twin_b_uid"] if side == "a" else pair["twin_a_uid"]
            )
            items.append(example)

        ambig_uid = pair.get("ambig_uid")
        if ambig_uid is not None:
            ambig_base = clean_by_uid.get(ambig_uid)
            if ambig_base is not None:
                ambig_item = dict(ambig_base)
                ambig_item["twin_side"] = "ambig"
                ambig_item["twin_partner_uid"] = None
                items.append(ambig_item)
    return items


def load_sample(pairs_per_category=DEFAULT_PAIRS_PER_CATEGORY, seed=DEFAULT_SEED):
    clean = load_jsonl(CLEAN_PATH)
    clean_by_uid = {row["uid"]: row for row in clean}
    twins = load_jsonl(TWINS_PATH)

    sampled_twins = sample_twins(twins, pairs_per_category, seed)
    items = build_items(sampled_twins, clean_by_uid)
    return sampled_twins, items


if __name__ == "__main__":
    sampled_twins, items = load_sample()
    categories = sorted({row["category"] for row in sampled_twins})
    print(f"Sampled {len(sampled_twins)} twin pairs across {len(categories)} categories")
    print(f"Built {len(items)} items")
    for category in categories:
        count = sum(1 for row in sampled_twins if row["category"] == category)
        print(f"  {category}: {count} pairs")
