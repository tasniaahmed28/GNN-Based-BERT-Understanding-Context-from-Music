"""Task 3: GNN-BERT fusion -- cross-attention variant and early-concat variant."""
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn import GraphSAGE, global_mean_pool
from transformers import AutoModel


class FusionModel(nn.Module):
    """Cross-attention fusion: graph embedding attends over BERT token embeddings
    (spec Algorithm 3): z = CONCAT(g, A*Htext); y_hat = sigma(Wz)."""

    def __init__(self, bert_name, gnn_in=12, gnn_hidden=64, gnn_layers=3, num_classes=8, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        bert_dim = self.bert.config.hidden_size
        self.sage = GraphSAGE(in_channels=gnn_in, hidden_channels=gnn_hidden, num_layers=gnn_layers, dropout=dropout)
        self.graph_proj = nn.Linear(gnn_hidden, bert_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=bert_dim, num_heads=4, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert_dim * 2, num_classes)

    def forward(self, graph_batch, input_ids, attention_mask):
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        Htext = text_out.last_hidden_state

        node_h = self.sage(graph_batch.x, graph_batch.edge_index)
        g = global_mean_pool(node_h, graph_batch.batch)
        g_proj = self.graph_proj(g).unsqueeze(1)

        key_padding_mask = (attention_mask == 0)
        attended, _ = self.cross_attn(query=g_proj, key=Htext, value=Htext, key_padding_mask=key_padding_mask)
        attended = attended.squeeze(1)

        z = torch.cat([g_proj.squeeze(1), attended], dim=1)
        return self.classifier(self.dropout(z))


class EarlyConcatFusion(nn.Module):
    """Ablation: simple concat of graph embedding + BERT CLS token, no attention."""

    def __init__(self, bert_name, gnn_in=12, gnn_hidden=64, gnn_layers=3, num_classes=8, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        bert_dim = self.bert.config.hidden_size
        self.sage = GraphSAGE(in_channels=gnn_in, hidden_channels=gnn_hidden, num_layers=gnn_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert_dim + gnn_hidden, num_classes)

    def forward(self, graph_batch, input_ids, attention_mask):
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        t = text_out.last_hidden_state[:, 0, :]

        node_h = self.sage(graph_batch.x, graph_batch.edge_index)
        g = global_mean_pool(node_h, graph_batch.batch)

        z = torch.cat([g, t], dim=1)
        return self.classifier(self.dropout(z))


class FusionDataset(TorchDataset):
    """Wraps paired (graph, text, label) dicts for the fusion models above."""

    def __init__(self, paired_list):
        self.data = paired_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def fusion_collate(batch):
    graphs_batch = PyGBatch.from_data_list([item["graph"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return graphs_batch, input_ids, attention_mask, labels
