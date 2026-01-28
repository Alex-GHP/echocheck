from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.data.dataset import PoliticalDataset, PoliticalDatasetJSONL
from pathlib import Path


def create_dataloaders(
    data_dir,
    batch_size=8,
    num_workers=0,
    tokenizer_name="roberta-base",
    max_length=512,
    shuffle_train=True,
    shuffle_val=False,
    shuffle_test=False,
    use_jsonl=None,
):
    """
    Create DataLoaders for train, validation, and test sets.

    Parameters:
    - data_dir: Directory containing data files
    - batch_size: Number of articles per batch (default: 8)
    - num_workers: Number of parallel data loading workers (default: 0)
    - tokenizer_name: Name of the tokenizer to use (default: roberta-base)
    - max_length: Maximum sequence length (default: 512)
    - shuffle_train: Whether to shuffle training data (default: True)
    - shuffle_val: Whether to shuffle validation data (default: False)
    - shuffle_test: Whether to shuffle test data (default: False)
    - use_jsonl: Use JSONL format (memory-efficient). Auto-detects if None.

    Returns:
    - Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    if use_jsonl is None:
        use_jsonl = (data_dir / "train.jsonl").exists()

    if use_jsonl:
        ext = ".jsonl"
        DatasetClass = PoliticalDatasetJSONL
    else:
        ext = ".json"
        DatasetClass = PoliticalDataset

    train_file = data_dir / f"train{ext}"
    val_file = data_dir / f"val{ext}"
    test_file = data_dir / f"test{ext}"

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_dataset = DatasetClass(
        str(train_file), tokenizer=tokenizer, max_length=max_length
    )
    val_dataset = DatasetClass(
        str(val_file), tokenizer=tokenizer, max_length=max_length
    )
    test_dataset = DatasetClass(
        str(test_file), tokenizer=tokenizer, max_length=max_length
    )

    if use_jsonl and num_workers > 0:
        print("Warning: Using num_workers=0 for JSONL datasets (file handle safety)")
        num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_val,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
