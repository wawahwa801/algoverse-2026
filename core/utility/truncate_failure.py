"""Keep only successful, unique results before the first failed task.

Usage:
    python truncate_failed_run.py /path/to/bbq_results_qwen3.5-9b.json /path/to/bbq_results_qwen3.5-9b.csv
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

FIELDNAMES = [
    "uid", "category", "subcategory", "question_index", "question_polarity",
    "context_condition", "model", "control_type", "effort", "max_tokens",
    "prompt_control", "think", "model_answer", "effective_answer",
    "answer_is_forced", "correct_answer", "is_correct", "selected_unknown",
    "selected_stereotype", "selected_anti_stereotype", "unknown_index",
    "stereotype_index", "anti_stereotype_index", "content", "thinking_chars",
    "latency_seconds", "probe_final_answer", "commitment_point_frac",
    "first_stereotype_mention_pct", "first_anti_stereotype_mention_pct",
    "flip_rate", "evidence_alignment", "is_twin", "twin_partner_uid",
    "twin_side", "status", "error",
]


def condition_key(row: dict) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("uid")),
        str(row.get("control_type")),
        str(row.get("effort")),
        str(row.get("max_tokens")),
        str(row.get("prompt_control")),
        str(row.get("think")),
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python truncate_failed_run.py RESULTS.json RESULTS.csv")

    json_path = Path(sys.argv[1])
    csv_path = Path(sys.argv[2])

    for path in (json_path, csv_path):
        backup = path.with_suffix(path.suffix + ".before_truncate.bak")
        if not backup.exists():
            shutil.copy2(path, backup)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    first_error = next((i for i, row in enumerate(data) if "error" in row), len(data))

    cleaned = []
    seen = set()
    for row in data[:first_error]:
        if "error" in row:
            continue
        key = condition_key(row)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(row)

    json_path.write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in cleaned:
            writer.writerow({field: row.get(field) for field in FIELDNAMES})

    print(f"First error index: {first_error}")
    print(f"Successful prefix records: {first_error}")
    print(f"Unique resumable records written: {len(cleaned)}")
    print(f"Backups: {json_path.with_suffix(json_path.suffix + '.before_truncate.bak')}")
    print(f"         {csv_path.with_suffix(csv_path.suffix + '.before_truncate.bak')}")


if __name__ == "__main__":
    main()