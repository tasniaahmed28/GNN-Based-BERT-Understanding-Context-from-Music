"""Task 4: dual-encoder contrastive model + InfoNCE loss + retrieval metrics (Algorithm 4)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn import GraphSAGE, global_mean_pool
from transformers import AutoModel

TEMPERATURE = 0.07


class DualEncoder(nn.Module):
    """Projects graph and text embeddings into a shared, L2-normalized space."""

    def __init__(self, bert_name, gnn_in=12, gnn_hidden=64, gnn_layers=3, proj_dim=128, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        bert_dim = self.bert.config.hidden_size
        self.sage = GraphSAGE(in_channels=gnn_in, hidden_channels=gnn_hidden, num_layers=gnn_layers, dropout=dropout)
        self.text_proj = nn.Linear(bert_dim, proj_dim)
        self.graph_proj = nn.Linear(gnn_hidden, proj_dim)

    def encode_text(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        t = out.last_hidden_state[:, 0, :]
        return F.normalize(self.text_proj(t), dim=-1)

    def encode_graph(self, graph_batch):
        node_h = self.sage(graph_batch.x, graph_batch.edge_index)
        g = global_mean_pool(node_h, graph_batch.batch)
        return F.normalize(self.graph_proj(g), dim=-1)


def contrastive_collate(batch):
    graphs_batch = PyGBatch.from_data_list([item["graph"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return graphs_batch, input_ids, attention_mask


def info_nce_loss(g_embed, t_embed, temperature=TEMPERATURE):
    logits = g_embed @ t_embed.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.T, labels)
    return (loss_g2t + loss_t2g) / 2


def compute_retrieval_metrics(loader, model, device, k_values=(1, 5, 10)):
    """Audio<->caption R@K, per the spec's Task 4 evaluation."""
    model.eval()
    all_g_embeds, all_t_embeds = [], []
    with torch.no_grad():
        for g_batch, ids_batch, mask_batch in loader:
            g_batch, ids_batch, mask_batch = g_batch.to(device), ids_batch.to(device), mask_batch.to(device)
            all_g_embeds.append(model.encode_graph(g_batch).cpu())
            all_t_embeds.append(model.encode_text(ids_batch, mask_batch).cpu())

    all_g_embeds = torch.cat(all_g_embeds, dim=0)
    all_t_embeds = torch.cat(all_t_embeds, dim=0)
    sim_matrix = all_g_embeds @ all_t_embeds.T
    N = sim_matrix.size(0)

    def recall_at_k(sim, k):
        topk = sim.topk(k, dim=1).indices
        correct = (topk == torch.arange(N).unsqueeze(1)).any(dim=1)
        return correct.float().mean().item()

    results = {}
    for k in k_values:
        results[f"audio2caption_R@{k}"] = recall_at_k(sim_matrix, k)
        results[f"caption2audio_R@{k}"] = recall_at_k(sim_matrix.T, k)
    return results, sim_matrix
