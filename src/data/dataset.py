import torch
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PoliticalDataset(Dataset):
    """
    Pytorch Dataset for political stance classification.

    This class:
    - Loads articles from JSON files
    - Tokenizes text using RoBERTa tokenizer
    - Converts labels to numbers
    - Returns data in format PyTorch expects
    """

    def __init__(self, json_file_path, tokenizer_name="roberta-base", max_length=512):
        """
        Constructor class

        Params:
        - json_file_path: Path to your JSON file (train.json, val.json, or test.json)
        - tokenizer_name: Which tokenizer to use (roberta-base)
        - max_length: Maximum tokens per article (512 is RoBERTa's limit)
        """

        self.json_file_path = json_file_path
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.label_map = {"center": 0, "left": 1, "right": 2}

        with open(json_file_path, "r") as file:
            self.articles = json.load(file)

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, index):
        """
        Get one article and prepare it for training.

        This is called automatically by PyTorch when it needs data

        Params:
        - index: Which article to get (0, 1, 2, 3...)

        Returns:
        - Dictionary with:
            - input_ids: Tokenized text as nums
            - attention_mask: Which tokens are real vs padding
            - labels: Label as num (0, 1 or 2)
        """

        article = self.articles[index]

        text = article["text"]
        label_str = article["label"]

        label_num = self.label_map[label_str]

        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_num, dtype=torch.long),
        }
