#!/usr/bin/env python3
"""
scripts/validate_dataset.py
Validates the conversational JSONL dataset for the DBMS Tutor fine-tuning pipeline.

Checks:
- Valid JSON per line
- Top-level 'messages' field (must be list)
- Correct message structure ({role, content})
- Permitted roles: 'system', 'user', 'assistant'
- Non-empty content
- Duplicate question detection
- Character & token length distribution
- Exit code 0 if valid, 1 if invalid
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Set, Tuple


ALLOWED_ROLES = {"system", "user", "assistant"}
EXPECTED_ROLE_SEQUENCE = ["system", "user", "assistant"]


def validate_record(record: dict, line_num: int) -> Tuple[bool, str, str, str]:
    """
    Validates a single record dictionary.
    Returns (is_valid, error_message, user_query, assistant_response).
    """
    if not isinstance(record, dict):
        return False, f"Line {line_num}: Record is not a JSON object.", "", ""

    if "messages" not in record:
        return False, f"Line {line_num}: Missing 'messages' field.", "", ""

    messages = record["messages"]
    if not isinstance(messages, list):
        return False, f"Line {line_num}: 'messages' must be a list.", "", ""

    if len(messages) < 2:
        return False, f"Line {line_num}: 'messages' must have at least user and assistant turns.", "", ""

    user_query = ""
    assistant_response = ""

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return False, f"Line {line_num}, Message {idx}: Message is not a JSON object.", "", ""

        if "role" not in msg or "content" not in msg:
            return False, f"Line {line_num}, Message {idx}: Missing 'role' or 'content'.", "", ""

        role = msg["role"]
        content = msg["content"]

        if role not in ALLOWED_ROLES:
            return False, f"Line {line_num}, Message {idx}: Invalid role '{role}'. Allowed: {ALLOWED_ROLES}", "", ""

        if not isinstance(content, str) or not content.strip():
            return False, f"Line {line_num}, Message {idx}: Empty content for role '{role}'.", "", ""

        if role == "user" and not user_query:
            user_query = content.strip()
        elif role == "assistant" and not assistant_response:
            assistant_response = content.strip()

    if not user_query:
        return False, f"Line {line_num}: Missing 'user' query in conversation.", "", ""
    if not assistant_response:
        return False, f"Line {line_num}: Missing 'assistant' response in conversation.", "", ""

    return True, "", user_query, assistant_response


def validate_file(file_path: str) -> bool:
    """
    Validates entire dataset file and prints structured report.
    """
    if not os.path.exists(file_path):
        print(f"ERROR: Dataset file not found at: {file_path}")
        return False

    total_examples = 0
    valid_examples = 0
    invalid_examples = 0
    errors: List[str] = []
    warnings: List[str] = []

    seen_questions: Dict[str, int] = {}
    duplicate_count = 0

    user_lengths: List[int] = []
    assistant_lengths: List[int] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            total_examples += 1
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as e:
                invalid_examples += 1
                errors.append(f"Line {line_num}: Invalid JSON syntax - {e}")
                continue

            is_valid, err_msg, u_query, a_resp = validate_record(data, line_num)
            if not is_valid:
                invalid_examples += 1
                errors.append(err_msg)
                continue

            # Duplicate check (case-insensitive normalized)
            norm_q = " ".join(u_query.lower().split())
            if norm_q in seen_questions:
                duplicate_count += 1
                warnings.append(
                    f"Duplicate question detected: '{u_query}' (first seen line {seen_questions[norm_q]}, again at line {line_num})"
                )
            else:
                seen_questions[norm_q] = line_num

            # Quality heuristic warnings
            if len(a_resp) < 30:
                warnings.append(f"Line {line_num}: Very short assistant answer ({len(a_resp)} chars).")
            elif len(a_resp) > 1500:
                warnings.append(f"Line {line_num}: Unusually long assistant answer ({len(a_resp)} chars).")

            valid_examples += 1
            user_lengths.append(len(u_query))
            assistant_lengths.append(len(a_resp))

    # Calculate statistics
    avg_user_len = sum(user_lengths) / len(user_lengths) if user_lengths else 0
    min_user_len = min(user_lengths) if user_lengths else 0
    max_user_len = max(user_lengths) if user_lengths else 0

    avg_asst_len = sum(assistant_lengths) / len(assistant_lengths) if assistant_lengths else 0
    min_asst_len = min(assistant_lengths) if assistant_lengths else 0
    max_asst_len = max(assistant_lengths) if assistant_lengths else 0

    # Output Validation Report
    print("\n================ DATASET VALIDATION ================")
    print(f"File: {file_path}")
    print(f"Total examples: {total_examples}")
    print(f"Valid examples: {valid_examples}")
    print(f"Invalid examples: {invalid_examples}")
    print(f"Duplicates: {duplicate_count}")
    print("----------------------------------------------------")
    print(f"User Query Lengths (chars):      min={min_user_len}, max={max_user_len}, avg={avg_user_len:.1f}")
    print(f"Assistant Answer Lengths (chars): min={min_asst_len}, max={max_asst_len}, avg={avg_asst_len:.1f}")
    print("----------------------------------------------------")

    if errors:
        print("ERRORS FOUND:")
        for err in errors[:10]:
            print(f"  [X] {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors.")

    if warnings:
        print("QUALITY WARNINGS:")
        for w in warnings[:10]:
            print(f"  [!] {w}")

    # Small dataset warning
    if total_examples < 100:
        print("\nWARNING:")
        print(f"The dataset contains only {total_examples} examples.")
        print("This is sufficient for testing the pipeline but insufficient for robust specialization.")
        print("Recommended future dataset size: 500-2000+ high-quality examples.")

    passed = (invalid_examples == 0 and duplicate_count == 0 and total_examples > 0)
    print(f"\nStatus: {'PASS' if passed else 'FAIL'}")
    print("=====================================================\n")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Validate conversational JSONL dataset.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/dbms_tutor.jsonl",
        help="Path to JSONL file to validate.",
    )
    args = parser.parse_args()

    success = validate_file(args.data_path)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
