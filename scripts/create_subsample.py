"""Stratified subsample of the JSONL splits"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from echocheck.config import PROJECT_ROOT, settings

TARGETS: dict[str, int] = {"train": 100_000, "val": 12_500, "test": 12_500}


def stratified_subsample(
    input_path: Path,
    output_path: Path,
    target_total: int,
    labels: tuple[str, ...],
    seed: int,
) -> None:
    rng = random.Random(seed)

    by_label: dict[str, list[str]] = defaultdict(list)
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            article = json.loads(line)
            by_label[article["label"]].append(line)

    per_class = target_total // len(labels)
    print(
        f"{input_path.name}: {sum(len(v) for v in by_label.values()):,} → {target_total:,} ({per_class:,}/class)"
    )

    sampled: list[str] = []
    for label in labels:
        pool = by_label[label]
        if per_class > len(pool):
            raise ValueError(
                f"Not enough '{label}' articles in {input_path.name}: have {len(pool)}, need {per_class}"
            )
        sampled.extend(rng.sample(pool, per_class))

    rng.shuffle(sampled)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for line in sampled:
            f.write(line + "\n")


def main() -> None:
    input_dir = PROJECT_ROOT / settings.processed_data_jsonl_dir

    from echocheck.config import SubsampleSettings

    output_dir = PROJECT_ROOT / SubsampleSettings().processed_data_jsonl_dir
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()

    for split, target in TARGETS.items():
        stratified_subsample(
            input_path=input_dir / f"{split}.jsonl",
            output_path=output_dir / f"{split}.jsonl",
            target_total=target,
            labels=settings.labels,
            seed=settings.random_seed,
        )

    print("Subsample creation complete.")


if __name__ == "__main__":
    main()
