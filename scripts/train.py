from __future__ import annotations

import os
import sys
from pathlib import Path

from echocheck.config import Settings, SmokeSettings
from echocheck.data.dataset import PoliticalDatasetJSONL
from echocheck.metrics import compute_metrics
from echocheck.models.model import load_political_classifier
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def main():
    import torch

    smoke = "--smoke" in sys.argv
    cfg = SmokeSettings() if smoke else Settings()

    print(f"ECHOCHECK — TRAINER + WANDB {'(SMOKE)' if smoke else ''}".strip())

    # wandb env vars the Trainer picks up automatically.
    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    if cfg.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", cfg.wandb_entity)
    os.environ.setdefault("WANDB_LOG_MODEL", cfg.wandb_log_model)
    os.environ.setdefault("WANDB_WATCH", cfg.wandb_watch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    print("\nLoading model...")
    model, tokenizer = load_political_classifier(
        model_name=cfg.base_model,
        num_labels=cfg.num_labels,
        device=device,
    )

    print("\nLoading datasets...")
    data_dir = Path(cfg.processed_data_jsonl_dir)
    train_dataset = PoliticalDatasetJSONL(
        str(data_dir / "train.jsonl"),
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )
    val_dataset = PoliticalDatasetJSONL(
        str(data_dir / "val.jsonl"),
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )

    training_args = TrainingArguments(
        output_dir=str(cfg.model_output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=True,
        logging_steps=cfg.logging_steps,
        report_to=["wandb"],
        run_name=cfg.run_name,
        bf16=cfg.bf16,
        dataloader_num_workers=cfg.dataloader_num_workers,
        overwrite_output_dir=False,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    resume = any(Path(cfg.model_output_dir).glob("checkpoint-*"))
    print(f"\nStarting training (resume_from_checkpoint={resume})...\n")
    trainer.train(resume_from_checkpoint=resume)

    final_dir = cfg.final_model_dir
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nFinal model saved to {final_dir}")

    metrics = trainer.evaluate()
    print("\nFinal validation metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
