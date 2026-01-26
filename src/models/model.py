import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer
from typing import Optional


def load_political_classifier(
    model_name: str = "roberta-base",
    num_labels: int = 3,
    device: Optional[torch.device] = None,
):
    """
    Load RoBERTa model with classification head for political stance.

    Params:
    - model_name: The pre-trained model (default: "roberta-base")
    - num_labels: Number of classes (default: 3 for center/left/right)
    - device: Device to load model on (CPU or GPU)

    Returns:
    - model: Loaded and configured model
    - tokenizer: Associated tokenizer
    """

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = RobertaForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, problem_type="single_label_classification"
    )

    model = model.to(device)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

    print("\nModel Information:")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Number of classes: {num_labels}")

    return model, tokenizer


def print_model_info(model):
    """
    Print detailed information about the model architecture
    """

    print("\n" + "=" * 60)
    print("Model Architecture")
    print("=" * 60)

    embedding_params = sum(
        param.numel() for param in model.roberta.embeddings.parameters()
    )
    encoder_params = sum(param.numel() for param in model.roberta.encoder.parameters())
    classifier_params = sum(
        param.numel() for param in model.roberta.classifier.parameters()
    )

    print("\nParams Breakdown:")
    print(f"Embeddings: {embedding_params:,}")
    print(f"Encoder (Transformer): {encoder_params:,}")
    print(f"Classifier Head: {classifier_params:,}")
    print(f"Total: {embedding_params + encoder_params + classifier_params:,}")


def configure_model_training(model, freeze_base: bool = False):
    """
    Configure which parts of the model to train.

    Parameters:
    - model: The model to configure
    - freeze_base: If True, freeze RoBERTa base (only train classifier)
                   If False, train entire model (recommended)
    """
    if freeze_base:
        for param in model.roberta.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    trainable = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    print(f"Trainable Params: {trainable:,}")

    return model
