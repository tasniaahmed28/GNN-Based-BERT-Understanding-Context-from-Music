"""Reusable training-loop functions shared across Tasks 1-4."""
from tqdm.auto import tqdm
from torch.optim import AdamW
import torch.nn as nn
from evaluate import evaluate_text_classifier, evaluate_graph_classifier, evaluate_fusion_classifier


def train_text_classifier(model, train_loader, val_loader, device, epochs=4, lr=2e-5):
    """Task 1: BERT tag/genre classifier training loop."""
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_macro_f1": [], "val_micro_f1": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_macro_f1, val_micro_f1, _, _ = evaluate_text_classifier(model, val_loader, device)
        history["train_loss"].append(total_loss / len(train_loader))
        history["val_macro_f1"].append(val_macro_f1)
        history["val_micro_f1"].append(val_micro_f1)
        print(f"Epoch {epoch+1}: val_macro_f1={val_macro_f1:.4f} val_micro_f1={val_micro_f1:.4f}")
    return history


def train_graph_classifier(model, train_loader, val_loader, device, epochs=30, lr=1e-3):
    """Task 2: GraphSAGE genre classifier training loop."""
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_macro_f1": [], "val_micro_f1": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_macro_f1, val_micro_f1 = evaluate_graph_classifier(model, val_loader, device)
        history["train_loss"].append(total_loss / len(train_loader))
        history["val_macro_f1"].append(val_macro_f1)
        history["val_micro_f1"].append(val_micro_f1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: val_macro_f1={val_macro_f1:.4f} val_micro_f1={val_micro_f1:.4f}")
    return history


def train_fusion_model(model, train_loader, val_loader, device, ckpt_path, epochs=4, lr=2e-5):
    """Tasks 3: fusion model training loop (works for both FusionModel and EarlyConcatFusion)."""
    import torch
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_macro_f1": [], "val_micro_f1": []}
    best_val_f1 = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for g_batch, ids_batch, mask_batch, label_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            g_batch, ids_batch, mask_batch, label_batch = (
                g_batch.to(device), ids_batch.to(device), mask_batch.to(device), label_batch.to(device)
            )
            optimizer.zero_grad()
            logits = model(g_batch, ids_batch, mask_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_macro_f1, val_micro_f1 = evaluate_fusion_classifier(model, val_loader, device)
        history["train_loss"].append(total_loss / len(train_loader))
        history["val_macro_f1"].append(val_macro_f1)
        history["val_micro_f1"].append(val_micro_f1)

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(model.state_dict(), ckpt_path)
        print(f"Epoch {epoch+1}: val_macro_f1={val_macro_f1:.4f} val_micro_f1={val_micro_f1:.4f}")
    return history, best_val_f1


def train_contrastive(model, train_loader, device, epochs=15, lr=2e-5):
    """Task 4: dual-encoder contrastive training loop (InfoNCE)."""
    from contrastive import info_nce_loss
    optimizer = AdamW(model.parameters(), lr=lr)
    history = {"train_loss": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for g_batch, ids_batch, mask_batch in train_loader:
            g_batch, ids_batch, mask_batch = g_batch.to(device), ids_batch.to(device), mask_batch.to(device)
            optimizer.zero_grad()
            g_embed = model.encode_graph(g_batch)
            t_embed = model.encode_text(ids_batch, mask_batch)
            loss = info_nce_loss(g_embed, t_embed)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
    return history
