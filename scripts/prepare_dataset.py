#!/usr/bin/env python3
"""
scripts/prepare_dataset.py
Splits data/dbms_tutor.jsonl into data/train.jsonl (80%) and data/validation.jsonl (20%)
using a reproducible random seed (42) and verifies that no data leakage exists.
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Set


def prepare_splits(
    input_path: str = "data/dbms_tutor.jsonl",
    train_path: str = "data/train.jsonl",
    val_path: str = "data/validation.jsonl",
    train_ratio: float = 0.8,
    seed: int = 42,
):
    if not os.path.exists(input_path):
        print(f"ERROR: Input dataset not found: {input_path}")
        sys.exit(1)

    records: List[dict] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"ERROR parsing line {line_num}: {e}")
                sys.exit(1)

    total_records = len(records)
    if total_records == 0:
        print("ERROR: Dataset is empty.")
        sys.exit(1)

    # Shuffle with fixed seed for exact reproducibility
    random.seed(seed)
    shuffled_indices = list(range(total_records))
    random.shuffle(shuffled_indices)

    # Compute split point
    train_count = int(total_records * train_ratio)
    # Ensure at least 1 record in validation if total_records >= 2
    if train_count == total_records and total_records >= 2:
        train_count = total_records - 1

    train_indices = set(shuffled_indices[:train_count])
    val_indices = set(shuffled_indices[train_count:])

    train_records = [records[i] for i in sorted(train_indices)]
    val_records = [records[i] for i in sorted(val_indices)]

    # --- Data Leakage Verification ---
    def extract_user_query(rec: dict) -> str:
        for msg in rec.get("messages", []):
            if msg.get("role") == "user":
                return " ".join(msg.get("content", "").lower().split())
        return ""

    train_queries: Set[str] = {extract_user_query(r) for r in train_records}
    val_queries: Set[str] = {extract_user_query(r) for r in val_records}

    intersection = train_queries.intersection(val_queries)
    if intersection:
        print("CRITICAL ERROR: Data leakage detected between train and validation sets!")
        print(f"Overlapping questions: {intersection}")
        sys.exit(1)

    # Save to disk
    os.makedirs(os.path.dirname(train_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(val_path) or ".", exist_ok=True)

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n================ DATASET SPLIT SUMMARY ================")
    print(f"Random Seed:             {seed}")
    print(f"Input Dataset:           {input_path} ({total_records} examples)")
    print(f"Training Split (80%):    {train_path} ({len(train_records)} examples)")
    print(f"Validation Split (20%):  {val_path} ({len(val_records)} examples)")
    print(f"Data Leakage Check:      PASSED (0 overlapping questions)")
    print("-------------------------------------------------------")

    if len(val_records) < 50:
        print("WARNING:")
        print(f"The validation set contains only {len(val_records)} examples.")
        print("Validation metrics will exhibit high statistical variance and CANNOT prove model generalization.")
        print("Do not rely on tiny validation loss numbers as proof of model capability.")
        print("Run scripts/evaluate_model.py on unseen benchmark questions for true evaluation.")
    print("=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train and validation sets.")
    parser.add_argument("--input_path", type=str, default="data/dbms_tutor.jsonl")
    parser.add_argument("--train_path", type=str, default="data/train.jsonl")
    parser.add_argument("--val_path", type=str, default="data/validation.jsonl")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_splits(
        input_path=args.input_path,
        train_path=args.train_path,
        val_path=args.val_path,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
