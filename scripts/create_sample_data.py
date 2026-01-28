import json
import random
import sys
import gc
from pathlib import Path


def create_sample(input_file, output_file, sample_size=1000, seed=42):
    random.seed(seed)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"({total:,} articles)")

    actual_sample_size = min(sample_size, total)
    sampled = random.sample(data, actual_sample_size)

    del data
    gc.collect()

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    print(f"  Saved {actual_sample_size:,} samples to {output_file.name}")

    del sampled
    gc.collect()

    return actual_sample_size


def create_tiny_test_data(output_dir):
    """Create tiny synthetic test data without loading the big files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    center_texts = [
        "The bipartisan committee reached a compromise on the infrastructure bill after weeks of negotiations. Both parties made concessions to reach the agreement.",
        "Economic analysts present mixed views on the proposed policy changes. Some experts see potential benefits while others express concerns about implementation.",
        "The mayor announced a new initiative to address urban development, citing input from various community stakeholders across the political spectrum.",
    ]

    left_texts = [
        "Progressive lawmakers are pushing for expanded healthcare coverage and workers' rights protections. The proposed legislation would increase funding for social programs.",
        "Environmental activists rallied at the state capitol demanding immediate action on climate change. Speakers called for a transition to renewable energy sources.",
        "The labor union announced plans to strike if demands for better wages and working conditions are not met. Workers have been organizing for months.",
    ]

    right_texts = [
        "Conservative leaders emphasized the importance of fiscal responsibility and reducing government spending. The proposed budget cuts aim to lower the national debt.",
        "Business groups praised the new deregulation measures, saying they will promote economic growth and job creation in the private sector.",
        "The senator called for stronger border security and immigration enforcement. The speech emphasized law and order and national sovereignty.",
    ]

    def make_articles(texts, label):
        return [
            {
                "text": text * 3,
                "label": label,
                "title": f"Article {i}",
                "date": "2024-01-01",
                "source": "test",
            }
            for i, text in enumerate(texts)
        ]

    train_data = (
        make_articles(center_texts, "center") * 50
        + make_articles(left_texts, "left") * 50
        + make_articles(right_texts, "right") * 50
    )
    random.shuffle(train_data)

    val_data = (
        make_articles(center_texts, "center") * 10
        + make_articles(left_texts, "left") * 10
        + make_articles(right_texts, "right") * 10
    )
    random.shuffle(val_data)

    test_data = (
        make_articles(center_texts, "center") * 10
        + make_articles(left_texts, "left") * 10
        + make_articles(right_texts, "right") * 10
    )
    random.shuffle(test_data)

    for name, data in [
        ("train.json", train_data),
        ("val.json", val_data),
        ("test.json", test_data),
    ]:
        with open(output_dir / name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Created {name} with {len(data)} articles")


def main():
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "processed_data"
    output_dir = project_root / "processed_data_sample"

    print("=" * 60)
    print("Creating Sample Dataset")
    print("=" * 60)

    # Check for --tiny flag
    if "--tiny" in sys.argv:
        create_tiny_test_data(output_dir)
        print("\n" + "=" * 60)
        print("Tiny test dataset created!")
        print(f"Location: {output_dir}")
        print("=" * 60)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    files_config = [
        ("val.json", 500),
        ("test.json", 500),
        ("train.json", 5000),
    ]

    for filename, sample_size in files_config:
        input_file = input_dir / filename
        output_file = output_dir / filename

        if not input_file.exists():
            print(f"  Skipping {filename} - not found")
            continue

        try:
            create_sample(input_file, output_file, sample_size)
        except MemoryError:
            print(f"\n  ERROR: Out of memory loading {filename}")
            return

        gc.collect()

    print("\n" + "=" * 60)
    print("Sample dataset created!")
    print(f"Location: {output_dir}")
    print("=" * 60)
    print("\nTo train on this sample, USE_SAMPLE_DATA=True in train.py (default)")


if __name__ == "__main__":
    main()
