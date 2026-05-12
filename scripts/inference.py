from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification

from echocheck.config import settings


class PoliticalClassifier:
    """Loads a Trainer-saved model directory and classifies text."""

    def __init__(self, model_source: str | None = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = settings.max_length
        self.labels = settings.labels
        self.id2label = settings.id2label

        source = model_source or str(settings.eval_model_dir)
        print(f"Loading model from {source} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = RobertaForSequenceClassification.from_pretrained(source).to(self.device)
        self.model.eval()
        print("Model loaded.\n")

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        results = []
        for probs in probabilities:
            predicted_class = int(probs.argmax())
            results.append(
                {
                    "prediction": self.id2label[predicted_class],
                    "confidence": float(probs[predicted_class]),
                    "probabilities": {
                        label: float(probs[i]) for i, label in enumerate(self.labels)
                    },
                }
            )
        return results


def display_result(text, result):
    print("=" * 60)
    print("ECHOCHECKER - POLITICAL STANCE INFERENCE")
    print("=" * 60)

    display_text = text[:200] + "..." if len(text) > 200 else text
    print(f"\nInput: {display_text}")

    prediction = result["prediction"].upper()
    confidence = result["confidence"] * 100
    print(f"\n>>> Prediction: {prediction}")
    print(f">>> Confidence: {confidence:.1f}%")

    print("\nAll probabilities:")
    for label, prob in result["probabilities"].items():
        bar = "█" * int(prob * 20)
        print(f"  {label:>6}: {prob * 100:5.1f}% {bar}")

    print("=" * 60)


def interactive_mode(classifier):
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter text to classify (or 'quit' to exit)\n")

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
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        return

    with path.open(encoding="utf-8") as f:
        text = f.read().strip()

    print(f"Read {len(text):,} characters from {file_path}\n")
    result = classifier.predict(text)
    display_result(text, result)


def demo_mode(classifier):
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
    parser = argparse.ArgumentParser(
        description="EchoChecker — classify political stance of text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/inference.py --text "Your political text here"
  python scripts/inference.py --file article.txt
  python scripts/inference.py --interactive
  python scripts/inference.py --demo
  python scripts/inference.py --demo --model alxdev/echocheck-political-stance
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", "-t", type=str, help="Text to classify")
    group.add_argument("--file", "-f", type=str, help="Path to file containing text to classify")
    group.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    group.add_argument("--demo", "-d", action="store_true", help="Run demo with example texts")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model source (local dir or HF Hub repo ID). Defaults to eval_model_dir in config.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    classifier = PoliticalClassifier(model_source=args.model)

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
