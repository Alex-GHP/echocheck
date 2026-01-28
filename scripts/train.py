import importlib.util
import json
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


def _load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_project_root = Path(__file__).parent.parent
_dataloaders_module = _load_module_from_path(
    "create_dataloaders", _project_root / "src" / "data" / "create_dataloaders.py"
)
_model_module = _load_module_from_path(
    "model", _project_root / "src" / "models" / "model.py"
)

create_dataloaders = _dataloaders_module.create_dataloaders
load_political_classifier = _model_module.load_political_classifier


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    loss_function,
    device,
    epoch,
    save_dir=None,
    checkpoint_every=5000,
    global_step=0,
):
    """
    Train the model for one epoch.

    Parameters:
    - model: The model to train
    - train_loader: DataLoader for training data
    - optimizer: Optimizer for updating weights
    - scheduler: Learning rate scheduler
    - loss_function: Loss function to use
    - device: Device to train on (CPU/GPU)
    - epoch: Current epoch number
    - save_dir: Directory to save mid-epoch checkpoints
    - checkpoint_every: Save checkpoint every N batches (default: 5000)
    - global_step: Starting global step (for resuming)

    Returns:
    - Average loss for the epoch
    - Average accuracy for the epoch
    - Final global step
    """

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = loss_function(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        scheduler.step()

        predictions = torch.argmax(logits, dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item()
        global_step += 1

        current_loss = total_loss / (batch_idx + 1)
        current_acc = total_correct / total_samples
        progress_bar.set_postfix(
            {"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"}
        )

        if save_dir and checkpoint_every and (global_step % checkpoint_every == 0):
            checkpoint_path = Path(save_dir) / "checkpoint_latest.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "batch_idx": batch_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": current_loss,
                    "train_acc": current_acc,
                },
                checkpoint_path,
            )
            tqdm.write(f"[Checkpoint saved at step {global_step:,}]")

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_correct / total_samples

    return avg_loss, avg_accuracy, global_step


def validate(model, val_loader, loss_function, device, epoch):
    """
    Validate the model on validation set.

    Parameters:
    - model: The model to validate
    - val_loader: DataLoader for validation data
    - loss_function: Loss function to use
    - device: Device to validate on
    - epoch: Current epoch number

    Returns:
    - Average loss
    - Average accuracy
    - All predictions and labels (for detailed metrics)
    """

    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_function(logits, labels)

            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions)

    return avg_loss, accuracy, all_predictions, all_labels


def main():
    """
    Main training function that orchestrates everything.
    """
    USE_SAMPLE_DATA = False

    config = {
        "data_dir": "processed_data_sample"
        if USE_SAMPLE_DATA
        else "processed_data_jsonl",
        "model_name": "roberta-base",
        "batch_size": 24,
        "num_epochs": 3,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "num_workers": 0,
        "save_dir": "models/checkpoints",
        "best_model_dir": "models/best_model",
        "max_grad_norm": 1.0,
        "checkpoint_every": 5000,
        "resume_from": None,
    }

    latest_checkpoint = Path(config["save_dir"]) / "checkpoint_latest.pt"
    if latest_checkpoint.exists() and config["resume_from"] is None:
        config["resume_from"] = str(latest_checkpoint)

    print("=" * 60)
    print("ECHOCHECKER MODEL TRAINING")
    print("=" * 60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle_train=True,
        shuffle_val=False,
        shuffle_test=False,
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print("\n" + "=" * 60)
    print("Loading Model")
    print("=" * 60)

    model, tokenizer = load_political_classifier(
        model_name=config["model_name"], num_labels=3, device=device
    )

    loss_function = nn.CrossEntropyLoss()
    print("\nLoss function: CrossEntropyLoss")

    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    print("\nOptimizer: AdamW")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Weight decay: {config['weight_decay']}")

    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print("\nScheduler: Linear with warmup")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Warmup steps: {warmup_steps:,}")

    Path(config["save_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["best_model_dir"]).mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    global_step = 0
    best_val_accuracy = 0.0
    training_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    if config["resume_from"] and Path(config["resume_from"]).exists():
        print("\n" + "=" * 60)
        print("Resuming from Checkpoint")
        print("=" * 60)
        checkpoint = torch.load(config["resume_from"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        print(f"  Resumed from epoch {start_epoch}, step {global_step:,}")
        print(
            f"  Train loss: {checkpoint['train_loss']:.4f}, Train acc: {checkpoint['train_acc']:.4f}"
        )

        history_path = Path(config["save_dir"]) / "training_history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                training_history = json.load(f)

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Checkpoints will be saved every {config['checkpoint_every']:,} batches")

    for epoch in range(start_epoch, config["num_epochs"] + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'=' * 60}")

        train_loss, train_acc, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_function,
            device,
            epoch,
            save_dir=config["save_dir"],
            checkpoint_every=config["checkpoint_every"],
            global_step=global_step,
        )

        val_loss, val_acc, val_predictions, val_labels = validate(
            model, val_loader, loss_function, device, epoch
        )

        training_history["train_loss"].append(train_loss)
        training_history["train_acc"].append(train_acc)
        training_history["val_loss"].append(val_loss)
        training_history["val_acc"].append(val_acc)

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save epoch checkpoint
        checkpoint_path = Path(config["save_dir"]) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_path = Path(config["best_model_dir"]) / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                },
                best_model_path,
            )
            print(f"New best model! Val Acc: {val_acc:.4f}")
            print(f"Saved best model: {best_model_path}")

        # Save training history after each epoch
        history_path = Path(config["save_dir"]) / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2)

        # Print classification report
        label_names = ["center", "left", "right"]
        print("\nValidation Classification Report:")
        print(
            classification_report(val_labels, val_predictions, target_names=label_names)
        )

        # Remove mid-epoch checkpoint after successful epoch completion
        latest_checkpoint = Path(config["save_dir"]) / "checkpoint_latest.pt"
        if latest_checkpoint.exists():
            latest_checkpoint.unlink()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nBest Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"Best model saved to: {config['best_model_dir']}/best_model.pt")
    print(f"\nTraining history saved to: {history_path}")

    return model


if __name__ == "__main__":
    main()
