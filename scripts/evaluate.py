from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from echocheck.config import settings
from echocheck.data.dataset import PoliticalDatasetJSONL
from echocheck.metrics import compute_metrics
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def main():
    print("ECHOCHECK — TEST SET EVALUATION")

    os.environ["WANDB_PROJECT"] = settings.wandb_project
    if settings.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", settings.wandb_entity)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    model_dir = str(settings.eval_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    print(f"Loaded model from {model_dir}")

    test_dataset = PoliticalDatasetJSONL(
        str(settings.eval_test_data),
        tokenizer=tokenizer,
        max_length=settings.max_length,
    )

    args = TrainingArguments(
        output_dir="/tmp/echocheck_eval",
        per_device_eval_batch_size=settings.eval_batch_size,
        report_to=["wandb"],
        run_name=settings.eval_run_name,
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=settings.dataloader_num_workers,
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\nRunning evaluation on test set...")
    metrics = trainer.evaluate()

    print("Computing confusion matrix...")
    predictions_output = trainer.predict(test_dataset)
    preds = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids
    cm = confusion_matrix(labels, preds).tolist()
    report = classification_report(
        labels, preds, target_names=list(settings.labels), output_dict=True
    )

    results = {
        "test_metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        "classification_report": report,
        "confusion_matrix": cm,
        "timestamp": datetime.now().isoformat(),
        "model_dir": model_dir,
    }
    with Path(settings.eval_output_file).open("w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Test accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"Test macro-F1: {metrics['eval_macro_f1']:.4f}")
    print(f"Results saved to {settings.eval_output_file}")
    print()


if __name__ == "__main__":
    main()
