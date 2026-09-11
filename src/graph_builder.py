"""Build segment graphs (nodes = chroma segments, edges = temporal + similarity)."""
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity

SIM_THRESHOLD = 0.9  # cosine similarity threshold for extra similarity edges


def build_segment_graph(chroma_feats, label, sim_threshold=SIM_THRESHOLD):
    """chroma_feats: (n_segments, 12) array. Returns a PyG Data object."""
    n = chroma_feats.shape[0]
    x = torch.tensor(chroma_feats, dtype=torch.float)

    edge_list = []
    # temporal adjacency: i <-> i+1
    for i in range(n - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])

    # similarity edges: cosine sim > threshold (excluding self and already-adjacent)
    sim_matrix = cosine_similarity(chroma_feats)
    for i in range(n):
        for j in range(i + 1, n):
            if j != i + 1 and sim_matrix[i, j] > sim_threshold:
                edge_list.append([i, j])
                edge_list.append([j, i])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)
