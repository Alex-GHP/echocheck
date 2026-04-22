from __future__ import annotations

import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification

from echocheck.config import settings


def load_political_classifier(
    model_name: str | None = None,
    num_labels: int | None = None,
    device: torch.device | None = None,
):
    """Load RoBERTa model with classification head for political stance.

    All defaults come from `src.config.settings`. Override any argument
    explicitly to experiment without touching global config.

    Returns (model, tokenizer).
    """
    if model_name is None:
        model_name = settings.base_model
    if num_labels is None:
        num_labels = settings.num_labels
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification",
        id2label=settings.id2label,
        label2id=settings.label2id,
    )

    model = model.to(device)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    print("\nModel Information:")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Number of classes: {num_labels}")

    return model, tokenizer
