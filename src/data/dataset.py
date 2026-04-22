from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from echocheck.config import settings


class PoliticalDatasetJSONL(Dataset):
    """Memory-efficient PyTorch Dataset that reads from JSONL files.

    Builds a byte-offset index on init (one entry per article) but keeps only
    file handles — text is read and tokenized on-demand in `__getitem__`.
    """

    def __init__(
        self,
        jsonl_file_path,
        tokenizer=None,
        tokenizer_name: str | None = None,
        max_length: int | None = None,
    ):
        self.jsonl_file_path = jsonl_file_path
        self.max_length = max_length if max_length is not None else settings.max_length

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or settings.base_model)

        self.label_map = settings.label2id
        print(f"Indexing {jsonl_file_path}...", end=" ", flush=True)
        self.line_positions = []
        with Path(jsonl_file_path).open("rb") as f:
            pos = 0
            for line in f:
                if line.strip():
                    self.line_positions.append(pos)
                pos += len(line)

        print(f"({len(self.line_positions):,} articles)")

        self.file = Path(jsonl_file_path).open(encoding="utf-8")

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
