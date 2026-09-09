import os
import json
import random
from typing import List, Dict, Any

def create_splits(input_json: str, train_out: str, eval_out: str, seed: int = 42, eval_ratio: float = 0.2):
    """
    Splits mined dataset into train (80%) and held-out eval (20%) JSONL splits.
    """
    with open(input_json, "r") as f:
        data = json.load(f)

    random.seed(seed)
    random.shuffle(data)

    eval_count = int(len(data) * eval_ratio)
    eval_set = data[:eval_count]
    train_set = data[eval_count:]

    os.makedirs(os.path.dirname(train_out), exist_ok=True)

    with open(train_out, "w") as f:
        for item in train_set:
            f.write(json.dumps(item) + "\n")

    with open(eval_out, "w") as f:
        for item in eval_set:
            f.write(json.dumps(item) + "\n")

    print(f"[+] Dataset Split Summary:")
    print(f"    Total items: {len(data)}")
    print(f"    Train set:   {len(train_set)} -> {train_out}")
    print(f"    Eval set:    {len(eval_set)}  -> {eval_out}")

if __name__ == "__main__":
    create_splits(
        input_json="/Users/devojha/DoctorBoom/dataset/mined_candidates.json",
        train_out="/Users/devojha/DoctorBoom/dataset/train.jsonl",
        eval_out="/Users/devojha/DoctorBoom/dataset/eval.jsonl"
    )
