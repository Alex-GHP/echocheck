# EchoChecker: Political Stance and Ideology Classification using Advanced NLP Techniques

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.57+-orange.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Academic Information

| | |
|---|---|
| **Student** | Alexandru-Gabriel Morariu |
| **Supervisor** | Dr. Ing. Lucian Ștefăniță Grigore |
| **Course** | Artificial Intelligence |
| **Institution** | "Titu Maiorescu" University, Faculty of Computer Science |
| **Project Type** | Bachelor's Thesis |
| **Graduation Year** | 2026 |

---

## Project Description

**EchoCheck** is a deep learning-based system for automatic classification of political stance in news articles and political text. The system uses a fine-tuned RoBERTa transformer model to classify text into three categories:

- **Left** - Progressive political stance
- **Center** - Moderate political stance  
- **Right** - Conservative political stance

---

## Related Repositories

| Repository | Description |
|------------|-------------|
| **This Repository** | Model training, evaluation, and inference |
| [EchoChecker Web App](https://www.google.com) | Full-stack Web Application |
| [HuggingFace Model](https://huggingface.co/alxdev/echocheck-political-stance) | Hosted trained model |

---

## Model Performance

### Test Set Results (233,158 articles)

| Metric | Score |
|--------|-------|
| **Accuracy** | 95.50% |
| **Macro F1** | 95.49% |
| **Weighted F1** | 95.49% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Center | 0.949 | 0.955 | 0.952 | 77,220 |
| Left | 0.953 | 0.964 | 0.959 | 77,951 |
| Right | 0.963 | 0.945 | 0.954 | 77,987 |

### Confusion Matrix

|  | Predicted Center | Predicted Left | Predicted Right |
|--|------------------|----------------|-----------------|
| **Actual Center** | 73,756 | 1,890 | 1,574 |
| **Actual Left** | 1,543 | 75,164 | 1,244 |
| **Actual Right** | 2,426 | 1,826 | 73,735 |

---

## Tech Stack

- **Deep Learning Framework:** PyTorch 2.10+
- **NLP Library:** HuggingFace Transformers 4.57+
- **Base Model:** RoBERTa-base (125M parameters)
- **Package Manager:** UV
- **Hardware Used:** NVIDIA RTX 4070 (12GB VRAM), 64GB RAM DDR5

---

## Project Structure

```
echocheck/
├── src/                          
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset classes
│   │   └── create_dataloaders.py # DataLoader factory
│   └── models/
│       └── model.py             # Model loading utilities
│
├── scripts/                      # Executable scripts
│   ├── preprocess.py            # Data preprocessing
│   ├── convert_to_jsonl.py      # JSON to JSONL conversion
│   ├── train.py                 # Model training
│   ├── evaluate.py              # Test set evaluation
│   ├── inference.py             # Single text prediction
│   └── upload_to_hub.py         # HuggingFace upload
│
├── evaluation_results.json       # Official test results
├── pyproject.toml               # Project dependencies
└── README.md                    # This file
```

---

## Replication Guide

Steps to replicate this project from scratch.

### Prerequisites

- Python 3.14+

Recommended hardware:

- NVIDIA GPU with 8GB+ VRAM (or CPU with 32GB+ RAM)
- 64GB+ RAM recommended for full dataset
- ~50GB disk space for data and models

### Step 1: Clone the Repository

```bash
git clone https://github.com/AlexMoraworworworwor/echocheck.git
cd echocheck
```

### Step 2: Install Dependencies

```bash
# Install UV package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### Step 3: Download the Dataset

The model was trained on the **BIGNEWSBLN** dataset from HuggingFace.

**Dataset:** [https://github.com/launchnlp/POLITICS](https://github.com/launchnlp/POLITICS)

Download and place the data files in the `data/` directory:

```
data/
├── data_center.json    # Center-aligned articles
├── data_left.json      # Left-aligned articles
└── data_right.json     # Right-aligned articles
```

**Dataset Statistics:**
- Total articles: ~2.3 million
- Center: ~777,000 articles
- Left: ~778,000 articles
- Right: ~778,000 articles

### Step 4: Preprocess the Data

Split the data into train/validation/test sets (80/10/10):

```bash
uv run python scripts/preprocess.py
```

**Output:** Creates `processed_data/` with:
- `train.json` (1,865,241 articles)
- `val.json` (233,153 articles)
- `test.json` (233,158 articles)

### Step 5: Convert to JSONL Format (Memory Efficient)

Convert JSON to JSONL for memory-efficient training:

```bash
uv run python scripts/convert_to_jsonl.py
```

**Output:** Creates `processed_data_jsonl/` with:
- `train.jsonl`
- `val.jsonl`
- `test.jsonl`

### Step 6: Train the Model

```bash
uv run python scripts/train.py
```

**Training Configuration:**
- Batch size: 24
- Learning rate: 2e-5
- Epochs: 3
- Warmup: 10% of total steps

**Expected Duration:** ~30-40 hours on RTX 4070

**Output:** 
- `models/best_model/best_model.pt` - Best checkpoint
- `models/checkpoints/` - All epoch checkpoints

### Step 7: Evaluate on Test Set

```bash
uv run python scripts/evaluate.py
```

**Output:** `evaluation_results.json` with official metrics

### Step 8: Test with Inference Script

```bash
# Interactive mode
uv run python scripts/inference.py --interactive

# Single text
uv run python scripts/inference.py --text "Your political text here"

# Demo with examples
uv run python scripts/inference.py --demo
```

---

## Scripts Reference

| Script | Purpose | Command |
|--------|---------|---------|
| `preprocess.py` | Split data into train/val/test | `uv run python scripts/preprocess.py` |
| `convert_to_jsonl.py` | Convert to memory-efficient format | `uv run python scripts/convert_to_jsonl.py` |
| `train.py` | Train the model | `uv run python scripts/train.py` |
| `evaluate.py` | Evaluate on test set | `uv run python scripts/evaluate.py` |
| `inference.py` | Predict on new text | `uv run python scripts/inference.py --interactive` |
| `upload_to_hub.py` | Upload to HuggingFace | `uv run python scripts/upload_to_hub.py` |

---

## Quick Start (Using Pre-trained Model)

If you just want to use the trained model without retraining:

```python
from transformers import RobertaForSequenceClassification, AutoTokenizer
import torch

# Load from HuggingFace Hub
model = RobertaForSequenceClassification.from_pretrained(
    "alxdev/echocheck-political-stance"
)
tokenizer = AutoTokenizer.from_pretrained(
    "alxdev/echocheck-political-stance"
)

# Classify text
text = "The government should increase social spending to help working families."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    prediction = probs.argmax().item()

labels = {0: "center", 1: "left", 2: "right"}
print(f"Prediction: {labels[prediction]}")
print(f"Confidence: {probs[0][prediction]:.2%}")
```

---

## Acknowledgments

- **BIGNEWSBLN Dataset** by Yang et al. for providing the training data
- **HuggingFace** for the Transformers library and model hosting
- **"Titu Maiorescu" University** for supporting this research
- **Dr. Ing. Lucian Ștefăniță Grigore** for supervision and guidance

---

## Contact

For questions about this project:

- **Student:** Alexandru-Gabriel Morariu
- **Email:** [alex.morariu.dev@gmail.com](alex.morariu.dev@gmail.com)

---