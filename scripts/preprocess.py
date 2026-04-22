from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from echocheck.config import PROJECT_ROOT, settings


def load_json_files(file_paths):
    """Load JSON files and extract articles with labels."""
    all_articles = []
    for file_path in file_paths:
        path = Path(file_path)
        label = path.name.replace("data_", "").replace(".json", "")  # center, left, right
        with path.open(encoding="utf-8") as f:
            articles = json.load(f)

        for article in articles:
            processed_article = {
                "text": join_text_paragraphs(article["text"]),
                "label": label,
                "title": article.get("title", ""),
                "date": article.get("date", ""),
                "source": article.get("source", ""),
            }
            all_articles.append(processed_article)
    return all_articles


def join_text_paragraphs(text_list):
    if not text_list:
        return ""
    if isinstance(text_list, str):
        return text_list
    if isinstance(text_list, list) and len(text_list) == 1:
        return text_list[0]

    full_text = " ".join(text_list)
    return " ".join(full_text.split())  # Clean up multiple spaces


def filter_articles(articles, min_length: int):
    return [article for article in articles if len(article["text"]) >= min_length]


def group_by_label(articles):
    articles_by_label = defaultdict(list)
    for article in articles:
        articles_by_label[article["label"]].append(article)
    return articles_by_label


def create_splits(articles, train_ratio: float, val_ratio: float, seed: int, labels: tuple):
    """Create stratified train/val/test splits."""
    random.seed(seed)

    shuffled = articles.copy()
    random.shuffle(shuffled)

    articles_by_label = group_by_label(shuffled)

    train, val, test = [], [], []

    for label in labels:
        label_articles = articles_by_label[label]
        n = len(label_articles)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train.extend(label_articles[:train_end])
        val.extend(label_articles[train_end:val_end])
        test.extend(label_articles[val_end:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def print_statistics(train, val, test, labels: tuple):
    """Print split-size statistics with per-label percentages."""

    def get_label_counts(articles):
        return Counter(article["label"] for article in articles)

    splits = [("Training", train), ("Validation", val), ("Test", test)]
    print("\n" + "=" * 60)
    print("SPLIT STATISTICS")
    print("=" * 60)

    for name, split in splits:
        counts = get_label_counts(split)
        print(f"\n{name} Set: {len(split):,} articles")
        for label in labels:
            count = counts[label]
            percentage = (count / len(split)) * 100 if split else 0
            print(f"  - {label.capitalize()}: {count:,} ({percentage:.1f}%)")

    print("\n" + "=" * 60)


def save_splits(train, val, test, output_dir: Path, labels: tuple):
    """Save splits to JSON files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving splits to '{output_dir}' directory...")

    for name, data in [("train", train), ("val", val), ("test", test)]:
        file_path = out / f"{name}.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {file_path.name} ({len(data):,} articles)")

    print_statistics(train, val, test, labels)


def main():
    print("=" * 60)
    print("ECHOCHECK DATA PREPROCESSING")
    print("=" * 60)

    data_dir = PROJECT_ROOT / settings.raw_data_dir
    processed_dir = PROJECT_ROOT / settings.processed_data_dir

    file_paths = [data_dir / f"data_{label}.json" for label in settings.labels]

    if not all(fp.exists() for fp in file_paths):
        print("\nError: Could not find JSON files!")
        print(f"Looking in: {data_dir.absolute()}")
        print("\nPlease ensure files are named:")
        for fp in file_paths:
            print(f"- {fp.relative_to(PROJECT_ROOT)}")
        return

    print("\n" + "=" * 60)
    print("STEP 1: Loading JSON files")
    print("=" * 60)
    all_articles = load_json_files(file_paths)
    print(f"\nTotal articles loaded: {len(all_articles):,}")

    print("\n" + "=" * 60)
    print(f"STEP 2: Filtering articles (min_length={settings.min_article_length})")
    print("=" * 60)
    filtered_articles = filter_articles(all_articles, min_length=settings.min_article_length)
    print(f"Articles after filtering: {len(filtered_articles):,}")
    print(f"Removed: {len(all_articles) - len(filtered_articles):,} articles")

    print("\n" + "=" * 60)
    print("STEP 3: Creating train/validation/test splits")
    print("=" * 60)
    train, val, test = create_splits(
        filtered_articles,
        train_ratio=settings.train_ratio,
        val_ratio=settings.val_ratio,
        seed=settings.random_seed,
        labels=settings.labels,
    )
    print("Splits created successfully!")

    print("\n" + "=" * 60)
    print("STEP 4: Saving splits to files")
    print("=" * 60)
    save_splits(train, val, test, output_dir=processed_dir, labels=settings.labels)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
