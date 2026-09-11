"""Task 1: BERT text encoder / tag-classifier and its Dataset."""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256


def get_tokenizer(model_name=MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


class TagTextDataset(Dataset):
    """Tokenizes (text, label) pairs on the fly for the BERT tag/genre classifier."""

    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BertTagClassifier(nn.Module):
    """t = BERT_CLS(X_text); y_hat = sigma(W t + b)  (Task 1 spec, Sec 4.1)."""

    def __init__(self, model_name=MODEL_NAME, num_classes=8, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0, :]
        cls_token = self.dropout(cls_token)
        return self.classifier(cls_token)
