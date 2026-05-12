"""Per-article evaluation for windowed models, using majority voting"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from echocheck.config import WindowedSettings
from echocheck.data.windowed_dataset import WindowedPoliticalDataset
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import wandb


def main():
    cfg = WindowedSettings()
    print("ECHOCHECK — WINDOWED TEST EVALUATION (majority vote per article)")

    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    if cfg.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", cfg.wandb_entity)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_dir = str(cfg.eval_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)

    test_dataset = WindowedPoliticalDataset(
        str(cfg.eval_test_data),
        tokenizer=tokenizer,
        window_size=cfg.window_size,
        stride=cfg.stride,
    )

    args = TrainingArguments(
        output_dir="/tmp/echocheck_eval_windowed",
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        report_to=["wandb"],
        run_name=cfg.eval_run_name,
        bf16=torch.cuda.is_available(),
        dataloader_num_workers=cfg.dataloader_num_workers,
    )

    trainer = Trainer(model=model, args=args, eval_dataset=test_dataset, processing_class=tokenizer)

    print(f"Running predict on {len(test_dataset):,} windows")
    pred_output = trainer.predict(test_dataset)
    logits_per_window = pred_output.predictions
    preds_per_window = np.argmax(logits_per_window, axis=-1)

    print("Aggregating per-article (majority vote)")
    votes_per_article: dict[int, list[int]] = defaultdict(list)
    for i, article_id in enumerate(test_dataset.article_ids):
        votes_per_article[article_id].append(int(preds_per_window[i]))

    n_labels = len(cfg.labels)
    article_ids_sorted = sorted(votes_per_article.keys())
    article_preds = []
    article_labels = []
    tie_count = 0

    for aid in article_ids_sorted:
        votes = votes_per_article[aid]
        counts = np.bincount(votes, minlength=n_labels)

        article_preds.append(int(np.argmax(counts)))
        article_labels.append(test_dataset.article_labels[aid])

        if (counts == counts.max()).sum() > 1:
            tie_count += 1

    article_preds = np.array(article_preds)
    article_labels = np.array(article_labels)

    cm = confusion_matrix(article_labels, article_preds).tolist()
    report = classification_report(
        article_labels,
        article_preds,
        target_names=list(cfg.labels),
        output_dict=True,
    )

    accuracy = (article_preds == article_labels).mean()
    macro_f1 = report["macro avg"]["f1-score"]

    results = {
        "test_metrics": {
            "n_articles": len(article_ids_sorted),
            "n_windows_total": len(test_dataset),
            "avg_windows_per_article": len(test_dataset) / len(article_ids_sorted),
            "n_articles_with_tied_vote": tie_count,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        },
        "aggregation": "majority_vote",
        "classification_report": report,
        "confusion_matrix": cm,
        "timestamp": datetime.now().isoformat(),
        "model_dir": model_dir,
        "window_size": cfg.window_size,
        "stride": cfg.stride,
    }

    with Path(cfg.eval_output_file).open("w") as f:
        json.dump(results, f, indent=2)

    # Log per-article aggregated metrics to wandb. The Trainer only logs
    # per-window metrics during predict(); the headline numbers (article-level
    # majority vote) are computed outside the Trainer, so they need an explicit
    # push. Using `eval_*` keys to match the baseline run's column names so
    # the runs are directly comparable in the wandb dashboard.
    wandb.log(
        {
            "eval_accuracy": float(accuracy),
            "eval_macro_f1": float(macro_f1),
            "eval_weighted_f1": float(report["weighted avg"]["f1-score"]),
            "eval_precision_center": report["center"]["precision"],
            "eval_precision_left": report["left"]["precision"],
            "eval_precision_right": report["right"]["precision"],
            "eval_recall_center": report["center"]["recall"],
            "eval_recall_left": report["left"]["recall"],
            "eval_recall_right": report["right"]["recall"],
            "eval_f1_center": report["center"]["f1-score"],
            "eval_f1_left": report["left"]["f1-score"],
            "eval_f1_right": report["right"]["f1-score"],
            "n_articles_with_tied_vote": tie_count,
            "avg_windows_per_article": len(test_dataset) / len(article_ids_sorted),
        }
    )
    wandb.finish()

    print(f"Per-article test accuracy: {accuracy:.4f}")
    print(f"Per-article macro-F1:      {macro_f1:.4f}")
    print(f"({len(test_dataset):,} windows → {len(article_ids_sorted):,} articles)")
    print(
        f"Articles with tied votes:   {tie_count:,} ({tie_count / len(article_ids_sorted) * 100:.2f}%)"
    )
    print(f"Results saved to {cfg.eval_output_file}")


if __name__ == "__main__":
    main()
