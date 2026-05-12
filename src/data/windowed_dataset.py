"""Sliding-window dataset for long-document classification"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from echocheck.config import settings

class WindowedPoliticalDataset(Dataset):
    def __init__(
        self,
        jsonl_file_path,
        tokenizer=None,
        tokenizer_name: str | None = None,
        window_size: int = 256,
        stride: int = 64
    ):
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or settings.base_model)

        self.label_map = settings.label2id

        articles: list[dict] = []
        with Path(jsonl_file_path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    articles.append(json.loads(line))

        texts = [a["text"] for a in articles]
        self.article_labels: list[int] = [self.label_map[a["label"]] for a in articles]

        print(f"Tokenizing {len(articles):,} articles with sliding windows "
              f"(window={window_size}, overlap={stride})...")

        encoding = self.tokenizer(
            texts,
            max_length=window_size,
            truncation=True,
            return_overflowing_tokens=True,
            stride=stride,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=True,
        )

        self.input_ids = encoding["input_ids"]
        self.attention_mask = encoding["attention_mask"]
        self.article_ids: list[int] = encoding["overflow_to_sample_mapping"].tolist()

        n_articles = len(articles)
        n_windows = self.input_ids.shape[0]

        print(
            f"{n_windows:,} windows from {n_articles:,} articles "
            f"(avg {n_windows / max(n_articles, 1):.2f} windows/article)"
        )

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):  # type: ignore[override]
        article_id = self.article_ids[idx]
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": torch.tensor(self.article_labels[article_id], dtype=torch.long),
        }