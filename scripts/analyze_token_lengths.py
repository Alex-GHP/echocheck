"""
Tokenize the training corpus and report length distribution.

Usage:
    uv run python scripts/analyze_token_length.py          # full train split

Outputs:
    token_length_analysis.json - stats to analyze and interpret (project root)
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from echocheck.config import PROJECT_ROOT, settings
from tqdm import tqdm
from transformers import AutoTokenizer


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def load_texts(jsonl_path: Path, sample_size: int | None, seed: int) -> list[str]:
    total = count_lines(jsonl_path)

    if sample_size is None or sample_size == total:
        print(f"Loading full corpus: {total:,} articles")
        texts = []
        with jsonl_path.open(encoding="utf-8") as f:
            for line in tqdm(f, total=total, desc="Reading"):
                texts.append(json.loads(line)["text"])
        return texts

    print(f"Subsampling {sample_size:,} of {total:,} articles. seed={seed}")
    rng = random.Random(seed)
    indices = set(rng.sample(range(total), sample_size))
    texts = []
    with jsonl_path.open(encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=total, desc="Reading")):
            if i in indices:
                texts.append(json.loads(line)["text"])

    return texts


def tokenize_lengths(texts: list[str], tokenizer, batch_size: int = 512) -> np.ndarray:
    lengths = np.zeros(len(texts), dtype=np.int32)
    for start in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
        batch = texts[start : start + batch_size]
        encodings = tokenizer(batch, truncation=False, padding=False, add_special_tokens=True)
        for i, ids in enumerate(encodings["input_ids"]):
            lengths[start + i] = len(ids)
    return lengths


def compute_stats(lengths: np.ndarray, thresholds: tuple[int, ...] = (64, 128, 256, 512)) -> dict:
    return {
        "nr_samples": int(lengths.size),
        "mean": float(lengths.mean()),
        "std": float(lengths.std()),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "percentiles": {
            "p50": int(np.percentile(lengths, 50)),
            "p75": int(np.percentile(lengths, 75)),
            "p90": int(np.percentile(lengths, 90)),
            "p95": int(np.percentile(lengths, 95)),
            "p99": int(np.percentile(lengths, 99)),
        },
        "fraction_under": {str(t): float((lengths < t).mean()) for t in thresholds},
        "fraction_truncated_at_512": float((lengths > 512).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Analyze a random subsample of this size (default: full corpus)",
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="train", help="Which split to analyze"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="token_length_analysis.json",
        help="Where to write the JSON summary",
    )
    args = parser.parse_args()

    jsonl_path = PROJECT_ROOT / settings.processed_data_jsonl_dir / f"{args.split}.jsonl"
    if not jsonl_path.exists():
        raise Exception(
            f"File not found: {jsonl_path}\n"
            "Run scripts/preprocess.py then scripts/convert_to_jsonl.py first"
        )

    texts = load_texts(jsonl_path, sample_size=args.sample, seed=settings.random_seed)

    tokenizer = AutoTokenizer.from_pretrained(settings.base_model)

    lengths = tokenize_lengths(texts, tokenizer)

    stats = compute_stats(lengths)
    stats["split"] = args.split
    stats["tokenizer"] = settings.base_model
    stats["timestamp"] = datetime.now().isoformat()

    output_path = Path(args.output)
    with output_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()
