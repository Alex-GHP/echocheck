import sys
import json
from pathlib import Path
import importlib.util

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

spec = importlib.util.spec_from_file_location(
    "create_dataloaders_module", project_root / "src" / "data" / "create_dataloaders.py"
)
create_dataloaders_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(create_dataloaders_module)
create_dataloaders = create_dataloaders_module.create_dataloaders


def create_minimal_test_data(output_dir):
    """Create minimal test JSON files for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_data = {
        "train": [
            {
                "text": "This is a test article about political issues and government policies. "
                * 10,
                "label": "center",
                "title": "Test Article Center",
                "date": "2020-01-01",
                "source": "test",
            },
            {
                "text": "This article discusses progressive policies and social justice issues. "
                * 10,
                "label": "left",
                "title": "Test Article Left",
                "date": "2020-01-02",
                "source": "test",
            },
            {
                "text": "This article covers conservative values and economic freedom. "
                * 10,
                "label": "right",
                "title": "Test Article Right",
                "date": "2020-01-03",
                "source": "test",
            },
        ],
        "val": [
            {
                "text": "Validation article about center politics. " * 15,
                "label": "center",
                "title": "Val Article Center",
                "date": "2020-01-01",
                "source": "test",
            },
            {
                "text": "Validation article about left politics. " * 15,
                "label": "left",
                "title": "Val Article Left",
                "date": "2020-01-02",
                "source": "test",
            },
        ],
        "test": [
            {
                "text": "Test article about right politics. " * 15,
                "label": "right",
                "title": "Test Article Right",
                "date": "2020-01-01",
                "source": "test",
            },
            {
                "text": "Another test article about center politics. " * 15,
                "label": "center",
                "title": "Test Article Center 2",
                "date": "2020-01-02",
                "source": "test",
            },
        ],
    }

    for split, data in test_data.items():
        file_path = output_dir / f"{split}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return output_dir


def test_dataloaders():
    """Test the create_dataloaders function."""
    print("=" * 60)
    print("Testing DataLoaders (Minimal Version)")
    print("=" * 60)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    test_data_dir = project_root / "test_data_minimal"

    print("\n1. Creating minimal test data...")
    create_minimal_test_data(test_data_dir)

    print("\n2. Testing create_dataloaders function...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=str(test_data_dir),
            batch_size=2,  # Small batch size for testing
            num_workers=0,  # Use 0 workers to avoid multiprocessing issues in test
            shuffle_train=True,
            shuffle_val=False,
            shuffle_test=False,
        )
        print("DataLoaders created successfully!")
    except Exception as e:
        print(f"Error creating DataLoaders: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n3. Testing training DataLoader...")
    try:
        batch = next(iter(train_loader))

        print(f"Batch keys: {list(batch.keys())}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        print(f"Labels: {batch['labels']}")

        assert batch["input_ids"].shape[0] <= 2, (
            f"Expected batch size <= 2, got {batch['input_ids'].shape[0]}"
        )
        assert batch["input_ids"].shape[1] == 512, (
            f"Expected sequence length 512, got {batch['input_ids'].shape[1]}"
        )
        assert batch["attention_mask"].shape == batch["input_ids"].shape, (
            "Attention mask shape mismatch"
        )
        assert batch["labels"].shape[0] == batch["input_ids"].shape[0], (
            "Labels batch size mismatch"
        )

        print("Shape validation passed!")

    except Exception as e:
        print(f"Error testing training DataLoader: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n4. Testing validation DataLoader...")
    try:
        batch = next(iter(val_loader))
        print(f"Validation batch shape: {batch['input_ids'].shape}")
        print(f"Validation labels: {batch['labels']}")
    except Exception as e:
        print(f"Error testing validation DataLoader: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n5. Testing test DataLoader...")
    try:
        batch = next(iter(test_loader))
        print(f"Test batch shape: {batch['input_ids'].shape}")
        print(f"Test labels: {batch['labels']}")
    except Exception as e:
        print(f"Error testing test DataLoader: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n6. Testing iteration over batches...")
    try:
        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            if batch_count >= 2:  # Only test first 2 batches
                break
        print(f"Successfully iterated over {batch_count} batches")
    except Exception as e:
        print(f"Error iterating over batches: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n7. Testing DataLoader lengths...")
    try:
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")

        assert len(train_loader) > 0, "Training DataLoader should have batches"
        assert len(val_loader) > 0, "Validation DataLoader should have batches"
        assert len(test_loader) > 0, "Test DataLoader should have batches"

        print("DataLoader lengths are valid!")
    except Exception as e:
        print(f"Error checking DataLoader lengths: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("All DataLoader tests passed!")
    print("=" * 60)
    print("\nYour DataLoaders are working correctly!")
    print("They're ready to use in your training script.")

    return 0


if __name__ == "__main__":
    exit(test_dataloaders())
