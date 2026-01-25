"""Preprocessing script to make sure the data is prepared and splitted accordingly"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path


def load_json_files(file_paths):
    """Load JSON files and extract articles with labels."""
    all_articles = []
    for file_path in file_paths:
        label = (
            os.path.basename(file_path).replace("data_", "").replace(".json", "")
        )  # center, left, right
        with open(file_path, "r", encoding="utf-8") as f:
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


def filter_articles(articles, min_length=50):
    return list(filter(lambda article: len(article["text"]) >= min_length, articles))


def group_by_label(articles):
    """Group articles by their label."""
    articles_by_label = defaultdict(list)
    for article in articles:
        articles_by_label[article["label"]].append(article)
    return articles_by_label


def create_splits(articles, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Create train/validation/test splits with stratification."""
    random.seed(42)

    shuffled = articles.copy()
    random.shuffle(shuffled)

    articles_by_label = group_by_label(shuffled)

    train = []
    val = []
    test = []

    for label in ["center", "left", "right"]:
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


def print_statistics(train, val, test):
    """Print statistics about the splits."""
    from collections import Counter

    def get_label_counts(articles):
        labels = [article["label"] for article in articles]
        return Counter(labels)

    train_counts = get_label_counts(train)
    val_counts = get_label_counts(val)
    test_counts = get_label_counts(test)

    print("\n" + "=" * 60)
    print("SPLIT STATISTICS")
    print("=" * 60)

    print(f"\nTraining Set: {len(train):,} articles")
    for label in ["center", "left", "right"]:
        count = train_counts[label]
        percentage = (count / len(train)) * 100 if train else 0
        print(f"  - {label.capitalize()}: {count:,} ({percentage:.1f}%)")

    print(f"\nValidation Set: {len(val):,} articles")
    for label in ["center", "left", "right"]:
        count = val_counts[label]
        percentage = (count / len(val)) * 100 if val else 0
        print(f"  - {label.capitalize()}: {count:,} ({percentage:.1f}%)")

    print(f"\nTest Set: {len(test):,} articles")
    for label in ["center", "left", "right"]:
        count = test_counts[label]
        percentage = (count / len(test)) * 100 if test else 0
        print(f"  - {label.capitalize()}: {count:,} ({percentage:.1f}%)")

    print("\n" + "=" * 60)


def save_splits(train, val, test, output_dir="processed_data"):
    """Save splits to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving splits to '{output_dir}' directory...")

    with open(f"{output_dir}/train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    print(f"Saved train.json ({len(train):,} articles)")

    with open(f"{output_dir}/val.json", "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)
    print(f"Saved val.json ({len(val):,} articles)")

    with open(f"{output_dir}/test.json", "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)
    print(f"Saved test.json ({len(test):,} articles)")

    print_statistics(train, val, test)


def main():
    """Main function to run the preprocessing pipeline."""
    print("=" * 60)
    print("ECHCHECKER DATA PREPROCESSING")
    print("=" * 60)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    data_dir = project_root / "data"
    processed_dir = project_root / "processed_data"

    file_paths = [
        data_dir / "data_center.json",
        data_dir / "data_left.json",
        data_dir / "data_right.json",
    ]

    file_paths_str = [str(fp) for fp in file_paths]

    if not all(os.path.exists(fp) for fp in file_paths_str):
        print("\nError: Could not find JSON files!")
        print(f"Looking in: {data_dir.absolute()}")
        print("\nPlease ensure files are named:")
        print("- data/data_center.json")
        print("- data/data_left.json")
        print("- data/data_right.json")
        return

    print("\n" + "=" * 60)
    print("STEP 1: Loading JSON files")
    print("=" * 60)
    all_articles = load_json_files(file_paths_str)
    print(f"\nTotal articles loaded: {len(all_articles):,}")

    print("\n" + "=" * 60)
    print("STEP 2: Filtering articles (min_length=50)")
    print("=" * 60)
    filtered_articles = filter_articles(all_articles, min_length=50)
    print(f"Articles after filtering: {len(filtered_articles):,}")
    print(f"Removed: {len(all_articles) - len(filtered_articles):,} articles")

    print("\n" + "=" * 60)
    print("STEP 3: Creating train/validation/test splits")
    print("=" * 60)
    train, val, test = create_splits(filtered_articles)
    print("Splits created successfully!")

    print("\n" + "=" * 60)
    print("STEP 4: Saving splits to files")
    print("=" * 60)
    save_splits(train, val, test, output_dir=str(processed_dir))

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("1. Review the statistics above")
    print("2. Check the 'processed_data' directory")
    print("3. Start creating your data loader for training!")


if __name__ == "__main__":
    main()
