import sys
from pathlib import Path
import importlib.util
import torch


project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

spec = importlib.util.spec_from_file_location(
    "model_module", project_root / "src" / "models" / "model.py"
)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
load_political_classifier = model_module.load_political_classifier


def main():
    """Test the model loading and inference."""
    print("=" * 60)
    print("Testing Political Classifier Model")
    print("=" * 60)

    try:
        model, tokenizer = load_political_classifier()
        print("Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return 1

    device = next(model.parameters()).device
    print(f"Model is on device: {device}")

    print("\n2. Testing tokenization...")
    sample_text = "This is a test article about politics and government policies."
    try:
        inputs = tokenizer(
            sample_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        print("Tokenization successful")
        print(f"Input shape: {inputs['input_ids'].shape}")
    except Exception as e:
        print(f"Error during tokenization: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n3. Testing forward pass...")
    model.eval()

    try:
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)

        print("Forward pass successful")
        print(f"Logits shape: {logits.shape}")
        print(f"Logits: {logits}")
        print(f"Probabilities: {probabilities}")
        print(f"Predicted class: {predicted_class.item()} (0=center, 1=left, 2=right)")

        assert logits.shape == (1, 3), f"Expected shape (1, 3), got {logits.shape}"
        assert probabilities.shape == (1, 3), (
            f"Expected shape (1, 3), got {probabilities.shape}"
        )
        print("Output shape validation passed")

    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n4. Testing with multiple samples...")
    sample_texts = [
        "This article discusses healthcare reform and social programs.",
        "The government announced new economic policies.",
        "Conservative values and free market principles are important.",
    ]

    try:
        inputs = tokenizer(
            sample_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_classes = torch.argmax(probabilities, dim=-1)

        print("Batch processing successful")
        print(f"Batch size: {len(sample_texts)}")
        print(f"Logits shape: {logits.shape}")
        print("Predictions:")
        label_map = {0: "center", 1: "left", 2: "right"}
        for i, (text, pred) in enumerate(zip(sample_texts, predicted_classes)):
            prob = probabilities[i][pred].item()
            print(
                f"     Sample {i + 1}: {pred.item()} ({label_map[pred.item()]}) - Confidence: {prob:.2%}"
            )

    except Exception as e:
        print(f"Error during batch processing: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("All model tests passed!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
