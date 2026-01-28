import argparse
from pathlib import Path
import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer


def upload_model(
    checkpoint_path: str,
    repo_name: str,
    private: bool = False,
):
    """
    Upload trained model to HuggingFace Hub.

    Parameters:
    - checkpoint_path: Path to best_model.pt
    - repo_name: Name for the HuggingFace repo (e.g., "echochecker-political-stance")
    - private: Whether to make the repo private (default: False/public)
    """

    print("=" * 60)
    print("UPLOADING MODEL TO HUGGINGFACE HUB")
    print("=" * 60)

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=3, problem_type="single_label_classification"
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    model.config.id2label = {0: "center", 1: "left", 2: "right"}
    model.config.label2id = {"center": 0, "left": 1, "right": 2}

    model.push_to_hub(
        repo_name,
        private=private,
        commit_message="Upload trained EchoChecker political stance classifier",
    )

    tokenizer.push_to_hub(repo_name, private=private, commit_message="Upload tokenizer")

    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE")
    print("=" * 60)
    print(f"https://huggingface.co/alxdev/{repo_name}")


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/best_model/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="echochecker-political-stance",
        help="Name for the HuggingFace repository",
    )
    parser.add_argument(
        "--private", action="store_true", help="Make the repository private"
    )

    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    upload_model(
        checkpoint_path=args.checkpoint, repo_name=args.repo_name, private=args.private
    )

    return 0


if __name__ == "__main__":
    main()
