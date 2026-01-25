import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer


def test_minimal():
    """Test with minimal data created in memory."""

    test_articles = [
        {
            "text": "This is a test article about political issues and government policies.",
            "label": "center"
        },
        {
            "text": "This article discusses progressive policies and social justice.",
            "label": "left"
        },
        {
            "text": "This article covers conservative values and economic freedom.",
            "label": "right"
        }
    ]
    
    print("\n1. Testing tokenizer loading...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return 1
    
    print("\n2. Testing tokenization...")
    label_map = {"center": 0, "left": 1, "right": 2}
    
    for i, article in enumerate(test_articles):
        text = article["text"]
        label_str = article["label"]
        label_num = label_map[label_str]
        
        print(f"\nArticle {i+1}:")
        print(f"Text: {text[:50]}...")
        print(f"Label: {label_str} ({label_num})")
        
        tokenized = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)
        label_tensor = torch.tensor(label_num, dtype=torch.long)
        
        print(f"Tokenized - Shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Label tensor: {label_tensor}")
    
    print("\n" + "=" * 60)
    print("All minimal tests passed!")
    print("=" * 60)
    print("\n💡 This confirms tokenization works.")
    print("If this works, your dataset class should work too!")
    
    return 0


if __name__ == "__main__":
    exit(test_minimal())
