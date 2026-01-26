from torch.utils.data import DataLoader
from src.data.dataset import PoliticalDataset
from pathlib import Path


def create_dataloaders(
    data_dir,
    batch_size=8,
    num_workers=2,
    shuffle_train=True,
    shuffle_val=False,
    shuffle_test=False,
):
    """
    Create DataLoaders for train, validation, and test sets.

    Parameters:
    - data_dir: Directory containing train.json, val.json, test.json
    - batch_size: Number of articles per batch (default: 8)
    - num_workers: Number of parallel data loading workers (default: 2)
    - shuffle_train: Whether to shuffle training data (default: True)
    - shuffle_val: Whether to shuffle validation data (default: False)
    - shuffle_test: Whether to shuffle test data (default: False)

    Returns:
    - Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    train_file = data_dir / "train.json"
    val_file = data_dir / "val.json"
    test_file = data_dir / "test.json"

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    train_dataset = PoliticalDataset(str(train_file))
    val_dataset = PoliticalDataset(str(val_file))
    test_dataset = PoliticalDataset(str(test_file))

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
