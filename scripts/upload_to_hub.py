from __future__ import annotations

import argparse
from pathlib import Path

from echocheck.config import settings
from transformers import AutoTokenizer, RobertaForSequenceClassification


def upload_model(
    model_dir: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Update model",
    token: str | None = None,
):
    """Upload a Trainer-saved model directory to the Hub."""
    print("=" * 60)
    print("UPLOADING MODEL TO HUGGINGFACE HUB")
    print("=" * 60)
    print(f"Source: {model_dir}")
    print(f"Target: {repo_id} (private={private})")

    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model.push_to_hub(repo_id, private=private, commit_message=commit_message, token=token)
    tokenizer.push_to_hub(repo_id, private=private, commit_message=commit_message, token=token)

    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE")
    print("=" * 60)
    print(f"https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload a model directory to the HuggingFace Hub")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(settings.final_model_dir),
        help="Directory saved by the Trainer (defaults to settings.final_model_dir)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=settings.hf_hub_repo_id,
        help="Destination repo ID on the Hub (defaults to settings.hf_hub_repo_id)",
    )
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    parser.add_argument(
        "--message",
        type=str,
        default="Update model",
        help="Commit message for both model and tokenizer pushes",
    )

    args = parser.parse_args()

    if not Path(args.model_dir).exists():
        print(f"Error: model directory not found: {args.model_dir}")
        return 1

    upload_model(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.message,
        token=settings.hf_token,
    )
    return 0


if __name__ == "__main__":
    main()
