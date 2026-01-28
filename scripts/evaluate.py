import importlib.util
import json
from datetime import datetime
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm


def _load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_project_root = Path(__file__).parent.parent
_dataset = _load_module_from_path(
    "dataset", _project_root / "src" / "data" / "dataset.py"
)
_model_module = _load_module_from_path(
    "model", _project_root / "src" / "models" / "model.py"
)

PoliticalDatasetJSONL = _dataset.PoliticalDatasetJSONL
load_political_classifier = _model_module.load_political_classifier

CONFIG = {
    "model_path": "models/best_model/best_model.pt",
    "test_data": "processed_data_jsonl/test.jsonl",
    "output_file": "evaluation_results.json",
    "batch_size": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def load_trained_model(checkpoint_path, device):
    """Load the trained model from checkpoint."""
    model, tokenizer = load_political_classifier(
        model_name="roberta-base", num_labels=3, device=device
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"Validation accuracy at save time: {checkpoint['val_accuracy']:.4f}")

    return model, tokenizer


def load_test_data(test_path, tokenizer, batch_size):
    """Load test dataset and create DataLoader."""
    test_dataset = PoliticalDatasetJSONL(test_path, tokenizer=tokenizer)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Test set loaded: {len(test_dataset):,} samples")
    print(f"Test batches: {len(test_loader):,}")

    return test_loader


def evaluate(model, test_loader, device):
    """Run evaluation on test set."""
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return all_predictions, all_labels, all_probabilities


def calculate_metrics(predictions, labels):
    """Calculate evaluation metrics."""
    label_names = ["center", "left", "right"]

    accuracy = accuracy_score(labels, predictions)

    report = classification_report(
        labels, predictions, target_names=label_names, output_dict=True
    )

    report_str = classification_report(labels, predictions, target_names=label_names)

    conf_matrix = confusion_matrix(labels, predictions)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "classification_report_str": report_str,
        "confusion_matrix": conf_matrix.tolist(),
        "confusion_matrix_raw": conf_matrix,
    }


def display_results(metrics):
    """Display evaluation results."""
    label_names = ["center", "left", "right"]

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"\nOverall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)"
    )

    print("\nClassification Report:")
    print(metrics["classification_report_str"])

    print("Confusion Matrix:")
    print(f"{'':>12} {'Predicted':^36}")
    print(f"{'':>12} {'center':>12} {'left':>12} {'right':>12}")
    conf_matrix = metrics["confusion_matrix_raw"]
    for i, label in enumerate(label_names):
        row = conf_matrix[i]
        print(f"{'Actual ' + label:>12} {row[0]:>12,} {row[1]:>12,} {row[2]:>12,}")


def save_results(metrics, output_path):
    """Save evaluation results to JSON file."""
    results = {
        "test_accuracy": metrics["accuracy"],
        "classification_report": metrics["classification_report"],
        "confusion_matrix": metrics["confusion_matrix"],
        "timestamp": datetime.now().isoformat(),
        "model_path": CONFIG["model_path"],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    print("=" * 60)
    print("ECHOCHECKER - TEST SET EVALUATION")
    print("=" * 60)

    device = CONFIG["device"]
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n[1/4] Loading trained model...")
    model, tokenizer = load_trained_model(CONFIG["model_path"], device)

    print("\n[2/4] Loading test dataset...")
    test_loader = load_test_data(CONFIG["test_data"], tokenizer, CONFIG["batch_size"])

    print("\n[3/4] Running evaluation on test set...")
    predictions, labels, probabilities = evaluate(model, test_loader, device)

    print("\n[4/4] Calculating metrics...")
    metrics = calculate_metrics(predictions, labels)
    display_results(metrics)
    save_results(metrics, CONFIG["output_file"])

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
