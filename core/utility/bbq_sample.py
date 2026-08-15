"""Shared stratified sampling of BBQ counterfactual twin pairs and ambiguous singles.

Used by both the OpenRouter sweep (GLM/Kimi/DeepSeek) and the local Qwen
GPU sweep, so the two runs pull from the same sample instead of two
independently-sampled subsets that can't be compared category-by-category.
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = ROOT / "data" / "bbq_clean.jsonl"
TWINS_PATH = ROOT / "data" / "bbq_twins_fixed.jsonl"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "bbq_subset.jsonl"

DEFAULT_PAIRS_PER_CATEGORY = 1
DEFAULT_SINGLES_PER_CATEGORY = 2
# Stratified sweep size used by models/bbq_sample.py and the OpenRouter/Qwen
# local sweep scripts (core/scripts/run_*_sweep.py).
SWEEP_PAIRS_PER_CATEGORY = 200
DEFAULT_SEED = 0


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return [json.loads(line) for line in content.splitlines() if line.strip()]

    if isinstance(parsed, list):
        return parsed

    raise TypeError(f"Unsupported JSON payload in {path}: expected a list or JSONL")


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


def sample_ambig_singles(clean, twins, singles_per_category=DEFAULT_SINGLES_PER_CATEGORY, seed=DEFAULT_SEED):

    paired_uids = set()
    for pair in twins:
        paired_uids.add(pair.get("twin_a_uid"))
        paired_uids.add(pair.get("twin_b_uid"))

    ambig_singles = [
        row for row in clean
        if row.get("context_condition") == "ambig" and row["uid"] not in paired_uids
    ]


    by_category = defaultdict(list)
    for row in ambig_singles:
        by_category[row["category"]].append(row)

    rng = random.Random(seed)
    sampled = []
    for rows in by_category.values():
        rows = rows[:]
        rng.shuffle(rows)
        

        for item in rows[:singles_per_category]:
            item_copy = dict(item)
            item_copy["is_twin"] = False
            sampled.append(item_copy)
            
    return sampled


def build_items(twins, clean_by_uid):
    """Turn each twin pair into two scoreable items: the original question's
    answers/metadata, but with context swapped to the twin's (recombined)
    version. Pairs tagged with an "ambig_uid" (see
    data/build_clean_pairs_subset.py) also contribute their matching
    ambiguous sibling as a third item, so s_AMB is computable alongside
    s_DIS. Mirrors models/bbq_sample.py::build_items - kept as a local copy
    rather than a cross-package import so core/ stays self-contained."""
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
            example["is_twin"] = True
            items.append(example)

        ambig_uid = pair.get("ambig_uid")
        if ambig_uid is not None:
            ambig_base = clean_by_uid.get(ambig_uid)
            if ambig_base is not None:
                ambig_item = dict(ambig_base)
                ambig_item["twin_side"] = "ambig"
                ambig_item["twin_partner_uid"] = None
                ambig_item["is_twin"] = False
                items.append(ambig_item)
    return items


def load_sample(pairs_per_category=DEFAULT_PAIRS_PER_CATEGORY, singles_per_category=DEFAULT_SINGLES_PER_CATEGORY, seed=DEFAULT_SEED):
    clean = load_jsonl(CLEAN_PATH)
    clean_by_uid = {row["uid"]: row for row in clean}
    twins = load_jsonl(TWINS_PATH)

    sampled_twins = sample_twins(twins, pairs_per_category, seed)
    items = build_items(sampled_twins, clean_by_uid)
    

    sampled_singles = sample_ambig_singles(clean, twins, singles_per_category, seed)
    items.extend(sampled_singles)
    
    return sampled_twins, sampled_singles, items


if __name__ == "__main__":
    sampled_twins, sampled_singles, items = load_sample()
    
    twin_categories = sorted({row["category"] for row in sampled_twins})
    single_categories = sorted({row["category"] for row in sampled_singles})
    all_categories = sorted(set(twin_categories + single_categories))
    
    print(f"Sampled {len(sampled_twins)} twin pairs across {len(twin_categories)} categories")
    print(f"Sampled {len(sampled_singles)} ambig singles across {len(single_categories)} categories")
    print(f"Built {len(items)} total items")
    print("-" * 40)
    
    for category in all_categories:
        twin_count = sum(1 for row in sampled_twins if row["category"] == category)
        single_count = sum(1 for row in sampled_singles if row["category"] == category)
        print(f"  {category}: {twin_count} pairs | {single_count} singles")
    sampled_twins, sampled_singles, items = load_sample()

    output_path = "bbq_subset.jsonl"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved {len(items)} items to {output_path}")