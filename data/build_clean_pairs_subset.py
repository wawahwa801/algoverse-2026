"""Build the frozen, filtered twin-pair subset per the team's feedback
decisions (feedback_points_pratham.md): opposite-alignment twins only
(drop stereo<->stereo / counter<->counter pairs), each tagged with its
matching ambiguous sibling's uid, sampled category-stratified with a
fixed seed so the sample doesn't drift on re-runs.

Pair count is 700, not the doc's 1000 - kept at 700 to match the team's
own throughput-driven sizing decision (see conversation), while adopting
the doc's quality requirements (alignment filter + ambig sibling).
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_bbq_paper_twins import locate_opening  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parent
TWINS_PATH = DATA_ROOT / "bbq_twins.jsonl"
CLEAN_PATH = DATA_ROOT / "bbq_clean.jsonl"

TOTAL_PAIRS = 700
SEED = 0
OUTPUT_PATH = DATA_ROOT / "bbq_pairs_subset_700_clean.jsonl"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_opposite_alignment(twin):
    return {
        twin["twin_a_evidence_allignment"],
        twin["twin_b_evidence_allignment"],
    } == {"stereotype_aligned", "counter_stereotype"}


def build_ambiguous_index(clean_rows):
    index = defaultdict(list)
    for row in clean_rows:
        if row["context_condition"] == "ambig":
            key = (row["category"], row["question_index"], row["question_polarity"])
            index[key].append(row)
    return index


def main():
    twins = load_jsonl(TWINS_PATH)
    clean = load_jsonl(CLEAN_PATH)
    ambiguous_index = build_ambiguous_index(clean)

    filtered = []
    dropped_same_alignment = 0
    dropped_no_ambig_match = 0

    for twin in twins:
        if not is_opposite_alignment(twin):
            dropped_same_alignment += 1
            continue

        key = (twin["category"], twin["question_index"], twin["question_polarity"])
        candidates = ambiguous_index.get(key, [])
        # Same prefix-match used to build the twins in the first place
        # (build_bbq_paper_twins.build_twins) - both twin_a_context and
        # twin_b_context share the same ambiguous opening as a verbatim
        # prefix, so this reliably re-identifies which ambiguous row it was.
        opening = locate_opening({"context": twin["twin_a_context"]}, candidates)
        if opening is None:
            dropped_no_ambig_match += 1
            continue

        twin = dict(twin)
        twin["ambig_uid"] = opening["uid"]
        filtered.append(twin)

    print(f"Twins loaded: {len(twins)}")
    print(f"Dropped (same-alignment, not opposite): {dropped_same_alignment}")
    print(f"Dropped (no matching ambiguous sibling found): {dropped_no_ambig_match}")
    print(f"Eligible opposite-alignment twins with ambig sibling: {len(filtered)}")

    by_category = defaultdict(list)
    for row in filtered:
        by_category[row["category"]].append(row)

    rng = random.Random(SEED)
    num_categories = len(by_category)
    base_count = TOTAL_PAIRS // num_categories
    remainder = TOTAL_PAIRS % num_categories

    subset = []
    for category in sorted(by_category):
        rows = by_category[category][:]
        rng.shuffle(rows)
        sample_size = base_count + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        sample_size = min(sample_size, len(rows))
        subset.extend(rows[:sample_size])

    rng.shuffle(subset)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in subset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(subset)} pairs to {OUTPUT_PATH}")
    for category in sorted(by_category):
        count = sum(1 for row in subset if row["category"] == category)
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
