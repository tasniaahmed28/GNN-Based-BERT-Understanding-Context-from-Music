"""Evaluation functions for each model type, plus a metrics-saving helper."""
import json
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_text_classifier(model, loader, device):
    """Task 1: BERT classifier eval. Returns macro_f1, micro_f1, preds, labels."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    micro_f1 = f1_score(all_labels, all_preds, average="micro")
    return macro_f1, micro_f1, all_preds, all_labels


def evaluate_graph_classifier(model, loader, device):
    """Task 2: GraphSAGE / CNN-style classifier eval (graph batches)."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    return f1_score(all_labels, all_preds, average="macro"), f1_score(all_labels, all_preds, average="micro")


def evaluate_cnn_classifier(model, loader, device):
    """Task 2 baseline (B2): CNN over mel-spectrogram tensors."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    return f1_score(all_labels, all_preds, average="macro"), f1_score(all_labels, all_preds, average="micro")


def evaluate_fusion_classifier(model, loader, device):
    """Task 3: fusion model eval (graph + text batches)."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for g_batch, ids_batch, mask_batch, label_batch in loader:
            g_batch, ids_batch, mask_batch, label_batch = (
                g_batch.to(device), ids_batch.to(device), mask_batch.to(device), label_batch.to(device)
            )
            logits = model(g_batch, ids_batch, mask_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_batch.cpu().numpy())
    return f1_score(all_labels, all_preds, average="macro"), f1_score(all_labels, all_preds, average="micro")


def compute_precision_recall(labels, preds):
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    return precision, recall


def save_metrics(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {path}")
