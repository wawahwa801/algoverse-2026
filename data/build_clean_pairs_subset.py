import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # Adjust path as needed
CLEAN_PATH = ROOT / "data" / "bbq_clean.jsonl"
TWINS_PATH = ROOT / "data" / "bbq_twins.jsonl"
OUTPUT_TWINS_PATH = ROOT / "data" / "bbq_twins_fixed.jsonl"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fix_twins():
    clean_rows = load_jsonl(CLEAN_PATH)
    # Map each UID to its clean context string from bbq_clean
    context_by_uid = {row["uid"]: row["context"] for row in clean_rows}

    twins_rows = load_jsonl(TWINS_PATH)
    fixed_twins = []
    missing_uids = 0

    for twin in twins_rows:
        twin_a_uid = twin.get("twin_a_uid")
        twin_b_uid = twin.get("twin_b_uid")

        # Update twin_a_context if present in clean dataset
        if twin_a_uid in context_by_uid:
            twin["twin_a_context"] = context_by_uid[twin_a_uid]
        else:
            print(f"Warning: twin_a_uid '{twin_a_uid}' not found in bbq_clean.jsonl")
            missing_uids += 1

        # Update twin_b_context if present in clean dataset
        if twin_b_uid in context_by_uid:
            twin["twin_b_context"] = context_by_uid[twin_b_uid]
        else:
            print(f"Warning: twin_b_uid '{twin_b_uid}' not found in bbq_clean.jsonl")
            missing_uids += 1

        fixed_twins.append(twin)

    # Save fixed twins back out as JSONLines
    with open(OUTPUT_TWINS_PATH, "w", encoding="utf-8") as f:
        for twin in fixed_twins:
            f.write(json.dumps(twin, ensure_ascii=False) + "\n")

    print(f"Successfully processed {len(fixed_twins)} twin pairs.")
    print(f"Missing UID matches: {missing_uids}")
    print(f"Saved fixed twins to {OUTPUT_TWINS_PATH}")


if __name__ == "__main__":
    fix_twins()