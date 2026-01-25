"""
Data Exploration Script for EchoChecker
This script helps understand the dataset structure and statistics.
"""

import json
import sys
from collections import Counter
from pathlib import Path


def load_json_file(filepath):
    """Load a JSON file and return the data."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        sys.exit(f"File not found: {filepath}")
    except json.JSONDecodeError as e:
        sys.exit(f"JSON decode error: {e}")
    except Exception as e:
        sys.exit(f"Error while loading JSON files: {e}")


def analyze_article_structure(articles, sample_size=3):
    """Analyze the structure of articles and show examples."""
    if not articles:
        sys.exit("No articles to analyze")

    print("First article")
    first_article = articles[0]
    for key, value in first_article.items():
        if isinstance(value, list):
            print(f"{key}: list with {len(value)} items")
            if len(value) > 0:
                print(f"First item type: {type(value[0]).__name__}")
                if isinstance(value[0], str) and len(value[0]) > 50:
                    print(f"First item preview: {value[0][:50]}...")
                else:
                    print(f"First item: {value[0]}")
        else:
            print(f"{key}: {type(value).__name__} = {value}")

    print(f"\nSample articles (showing {min(sample_size, len(articles))}):")
    for i, article in enumerate(articles[:sample_size]):
        print(f"\n--- Article {i + 1} ---")
        print(f"Title: {article.get('title', 'N/A')}")
        print(f"Date: {article.get('date', 'N/A')}")
        print(f"Source: {article.get('source', 'N/A')}")

        text = article.get("text", [])
        if isinstance(text, list):
            full_text = " ".join(text)
            print(f"Text: {len(text)} paragraphs, {len(full_text)} total characters")
            print(f"Preview: {full_text[:200]}...")
        elif isinstance(text, str):
            print(f"Text: {len(text)} characters")
            print(f"Preview: {text[:200]}...")


def calculate_statistics(articles, label):
    """Calculate statistics for a set of articles."""
    print("\n" + "=" * 60)
    print(f"STATISTICS FOR {label.upper()} ARTICLES")
    print("=" * 60)

    if not articles:
        sys.exit("No articles to analyze")

    total_articles = len(articles)

    text_lengths = []
    word_counts = []
    sources = []
    dates = []

    for article in articles:
        text = article.get("text", [])
        if isinstance(text, list):
            full_text = " ".join(text)
        elif isinstance(text, str):
            full_text = text
        else:
            full_text = ""

        text_lengths.append(len(full_text))
        word_counts.append(len(full_text.split()))
        sources.append(article.get("source", "unknown"))
        dates.append(article.get("date", "unknown"))

    print("\nBasic Stats:")
    print(f"Total articles: {total_articles:,}")
    print(f"Articles with text: {sum(1 for length in text_lengths if length > 0):,}")

    print("\nText Length Statistics:")
    print(f"Average characters: {sum(text_lengths) / len(text_lengths):.0f}")
    print(f"Min characters: {min(text_lengths):,}")
    print(f"Max characters: {max(text_lengths):,}")
    print(f"Median characters: {sorted(text_lengths)[len(text_lengths) // 2]:,}")

    print("\nWord Count Statistics:")
    print(f"Average words: {sum(word_counts) / len(word_counts):.0f}")
    print(f"Min words: {min(word_counts):,}")
    print(f"Max words: {max(word_counts):,}")
    print(f"Median words: {sorted(word_counts)[len(word_counts) // 2]:,}")

    source_counts = Counter(sources)
    print("\nSource Distribution:")
    for source, count in source_counts.most_common(10):
        print(f"{source}: {count:,} articles ({count / total_articles * 100:.1f}%)")

    valid_dates = [d for d in dates if d != "unknown" and d]
    if valid_dates:
        print("\nDate Range:")
        print(f"Earliest: {min(valid_dates)}")
        print(f"Latest: {max(valid_dates)}")
        print(f"Unique dates: {len(set(valid_dates))}")


def check_data_quality(articles):
    """Check for data quality issues."""
    print("\n" + "=" * 60)
    print("DATA QUALITY CHECK")
    print("=" * 60)

    if not articles:
        sys.exit("No articles to check")

    issues = {
        "missing_title": 0,
        "missing_text": 0,
        "empty_text": 0,
        "missing_date": 0,
        "missing_source": 0,
    }

    for article in articles:
        if not article.get("title"):
            issues["missing_title"] += 1
        if not article.get("text"):
            issues["missing_text"] += 1
        elif (
            isinstance(article.get("text"), list) and len(article.get("text", [])) == 0
        ):
            issues["empty_text"] += 1
        elif isinstance(article.get("text"), str) and len(article.get("text", "")) == 0:
            issues["empty_text"] += 1
        if not article.get("date"):
            issues["missing_date"] += 1
        if not article.get("source"):
            issues["missing_source"] += 1

    print("\nQuality Issues Found:")
    total = len(articles)
    for issue, count in issues.items():
        if count > 0:
            print(f"{issue}: {count:,} articles ({count / total * 100:.1f}%)")
        else:
            print(f"{issue}: None")

    return issues


def main():
    """Main exploration function."""
    print("=" * 60)
    print("ECHCHECKER DATA EXPLORATION")
    print("=" * 60)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    data_dir = project_root / "data"

    files = {
        "center": data_dir / "data_center.json",
        "left": data_dir / "data_left.json",
        "right": data_dir / "data_right.json",
    }

    if not all(f.exists() for f in files.values()):
        print("\nFiles not found in 'data/' directory.")
        print(f"Looking in: {data_dir.absolute()}")
        print("\nTip: Make sure your JSON files are in the 'data/' directory:")
        print("- data/data_center.json")
        print("- data/data_left.json")
        print("- data/data_right.json")

    datasets = {}
    for label, filepath in files.items():
        data = load_json_file(filepath)
        if data:
            datasets[label] = data

    if not datasets:
        print("\nNo data files found. Please check file paths.")
        print("\nTip: Make sure your JSON files are named:")
        print("- data_center.json")
        print("- data_left.json")
        print("- data_right.json")
        return

    if "center" in datasets:
        analyze_article_structure(datasets["center"], sample_size=2)

    for label, articles in datasets.items():
        calculate_statistics(articles, label)
        check_data_quality(articles)

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    total_articles = sum(len(articles) for articles in datasets.values())
    print(f"\nTotal articles across all labels: {total_articles:,}")
    for label, articles in datasets.items():
        print(
            f"{label}: {len(articles):,} articles ({len(articles) / total_articles * 100:.1f}%)"
        )

    print("\nExploration complete!")
    print("\nNext steps:")
    print("1. Review the statistics above")
    print("2. Check for any data quality issues")
    print("3. Create a data preprocessing script")


if __name__ == "__main__":
    main()
