import argparse
import importlib.util
from pathlib import Path

import torch


# Load local modules
def _load_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_project_root = Path(__file__).parent.parent
_model_module = _load_module_from_path(
    "model", _project_root / "src" / "models" / "model.py"
)
load_political_classifier = _model_module.load_political_classifier

# Configuration
CONFIG = {
    "model_path": "models/best_model/best_model.pt",
    "model_name": "roberta-base",
    "max_length": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

LABEL_MAP = {0: "center", 1: "left", 2: "right"}


class PoliticalClassifier:
    """Classifier for political stance prediction."""

    def __init__(self, config):
        """Initialize the classifier by loading model and tokenizer."""
        self.device = config["device"]
        self.max_length = config["max_length"]

        print(f"Loading model on {self.device}...")

        # Load model architecture
        self.model, self.tokenizer = load_political_classifier(
            model_name=config["model_name"], num_labels=3, device=self.device
        )

        # Load trained weights
        checkpoint_path = _project_root / config["model_path"]
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Set to evaluation mode
        self.model.eval()

        print("Model loaded successfully!\n")

    def predict(self, text):
        """
        Predict political stance of given text.

        Parameters:
        - text: String containing political article/statement

        Returns:
        - Dictionary with prediction, confidence, and all probabilities
        """
        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Convert to probabilities
        probabilities = torch.softmax(logits, dim=-1)
        probabilities = probabilities.cpu().numpy()[0]

        # Get prediction
        predicted_class = probabilities.argmax()
        predicted_label = LABEL_MAP[predicted_class]
        confidence = probabilities[predicted_class]

        return {
            "prediction": predicted_label,
            "confidence": float(confidence),
            "probabilities": {
                "center": float(probabilities[0]),
                "left": float(probabilities[1]),
                "right": float(probabilities[2]),
            },
        }

    def predict_batch(self, texts):
        """Predict on multiple texts at once (more efficient)."""
        # Tokenize all texts
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Convert to probabilities
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        # Build results
        results = []
        for probs in probabilities:
            predicted_class = probs.argmax()
            results.append(
                {
                    "prediction": LABEL_MAP[predicted_class],
                    "confidence": float(probs[predicted_class]),
                    "probabilities": {
                        "center": float(probs[0]),
                        "left": float(probs[1]),
                        "right": float(probs[2]),
                    },
                }
            )

        return results


def display_result(text, result):
    """Pretty print the prediction result."""
    print("=" * 60)
    print("ECHOCHECKER - POLITICAL STANCE INFERENCE")
    print("=" * 60)

    # Show truncated input
    display_text = text[:200] + "..." if len(text) > 200 else text
    print(f"\nInput: {display_text}")

    # Show prediction
    prediction = result["prediction"].upper()
    confidence = result["confidence"] * 100
    print(f"\n>>> Prediction: {prediction}")
    print(f">>> Confidence: {confidence:.1f}%")

    # Show all probabilities with visual bars
    print("\nAll probabilities:")
    for label, prob in result["probabilities"].items():
        bar = "█" * int(prob * 20)
        print(f"  {label:>6}: {prob * 100:5.1f}% {bar}")

    print("=" * 60)


def interactive_mode(classifier):
    """Run inference interactively."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter text to classify (or 'quit' to exit)")
    print("Tip: Paste longer text and press Enter twice\n")

    while True:
        try:
            text = input("Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if text.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if len(text) < 10:
            print("Text too short. Please enter a longer text.\n")
            continue

        result = classifier.predict(text)
        display_result(text, result)
        print()


def file_mode(classifier, file_path):
    """Read text from file and classify."""
    path = Path(file_path)

    if not path.exists():
        print(f"Error: File not found: {file_path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print(f"Read {len(text):,} characters from {file_path}\n")

    result = classifier.predict(text)
    display_result(text, result)


def demo_mode(classifier):
    """Run demo with example texts."""
    print("\n" + "=" * 60)
    print("DEMO MODE - Testing with example texts")
    print("=" * 60 + "\n")

    examples = [
        (
            "Left-leaning",
            "Universal healthcare is a fundamental human right. We must expand social programs "
            "and increase taxes on the wealthy to reduce inequality and support working families.",
        ),
        (
            "Right-leaning",
            "Lower taxes and reduced government regulation will stimulate economic growth. "
            "We must protect Second Amendment rights and secure our borders.",
        ),
        (
            "Center/Neutral",
            "The bipartisan committee heard arguments from both sides of the aisle before "
            "reaching a compromise on the infrastructure spending bill.",
        ),
    ]

    for label, text in examples:
        print(f"--- Example: {label} ---")
        result = classifier.predict(text)
        display_result(text, result)
        print()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="EchoChecker - Classify political stance of text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/inference.py --text "Your political text here"
  python scripts/inference.py --file article.txt
  python scripts/inference.py --interactive
  python scripts/inference.py --demo
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", "-t", type=str, help="Text to classify")
    group.add_argument(
        "--file", "-f", type=str, help="Path to file containing text to classify"
    )
    group.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )
    group.add_argument(
        "--demo", "-d", action="store_true", help="Run demo with example texts"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load model
    classifier = PoliticalClassifier(CONFIG)

    # Run appropriate mode
    if args.interactive:
        interactive_mode(classifier)
    elif args.demo:
        demo_mode(classifier)
    elif args.file:
        file_mode(classifier, args.file)
    elif args.text:
        result = classifier.predict(args.text)
        display_result(args.text, result)


if __name__ == "__main__":
    main()
