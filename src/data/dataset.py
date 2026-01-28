import torch
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PoliticalDatasetJSONL(Dataset):
    """
    Memory-efficient PyTorch Dataset that reads from JSONL files.

    Loads article metadata (file positions) but not the actual text,
    which is read on-demand during training.
    """

    def __init__(
        self,
        jsonl_file_path,
        tokenizer=None,
        tokenizer_name="roberta-base",
        max_length=512,
    ):
        """
        Params:
        - jsonl_file_path: Path to JSONL file (one JSON object per line)
        - tokenizer: Pre-loaded tokenizer (optional)
        - tokenizer_name: Which tokenizer to use if tokenizer not provided
        - max_length: Maximum tokens per article (512 is RoBERTa's limit)
        """
        self.jsonl_file_path = jsonl_file_path
        self.max_length = max_length

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.label_map = {"center": 0, "left": 1, "right": 2}
        print(f"Indexing {jsonl_file_path}...", end=" ", flush=True)
        self.line_positions = []
        with open(jsonl_file_path, "rb") as f:
            pos = 0
            for line in f:
                if line.strip():
                    self.line_positions.append(pos)
                pos += len(line)

        print(f"({len(self.line_positions):,} articles)")

        self.file = open(jsonl_file_path, "r", encoding="utf-8")

    def __len__(self):
        return len(self.line_positions)

    def __getitem__(self, index):
        self.file.seek(self.line_positions[index])
        line = self.file.readline()
        article = json.loads(line)

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

    def __del__(self):
        if hasattr(self, "file") and self.file:
            self.file.close()


class PoliticalDataset(Dataset):
    """
    Standard PyTorch Dataset for political stance classification.
    Loads all data into memory - use PoliticalDatasetJSONL for large files.
    """

    def __init__(
        self,
        json_file_path,
        tokenizer=None,
        tokenizer_name="roberta-base",
        max_length=512,
    ):
        """
        Params:
        - json_file_path: Path to JSON file
        - tokenizer: Pre-loaded tokenizer (optional)
        - tokenizer_name: Which tokenizer to use if tokenizer not provided
        - max_length: Maximum tokens per article (512 is RoBERTa's limit)
        """
        self.json_file_path = json_file_path
        self.max_length = max_length

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.label_map = {"center": 0, "left": 1, "right": 2}

        print(f"Loading {json_file_path}...", end=" ", flush=True)

        if json_file_path.endswith(".jsonl"):
            self.articles = []
            with open(json_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.articles.append(json.loads(line))
        else:
            with open(json_file_path, "r", encoding="utf-8") as file:
                self.articles = json.load(file)

        print(f"({len(self.articles):,} articles)")

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, index):
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
