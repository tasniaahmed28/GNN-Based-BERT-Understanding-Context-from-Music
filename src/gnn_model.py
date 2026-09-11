"""Task 2: GraphSAGE genre classifier, plus the CNN mel-spectrogram baseline (B2)."""
import torch.nn as nn
from torch_geometric.nn import GraphSAGE, global_mean_pool


class GraphSAGEClassifier(nn.Module):
    """GraphSAGE encoder + mean-pool readout + linear head (spec Sec 4.2)."""

    def __init__(self, in_channels=12, hidden_channels=64, num_layers=3, num_classes=8, dropout=0.3):
        super().__init__()
        self.sage = GraphSAGE(
            in_channels=in_channels, hidden_channels=hidden_channels,
            num_layers=num_layers, dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        h = self.sage(x, edge_index)
        g = global_mean_pool(h, batch)
        return self.classifier(g)


class MelCNN(nn.Module):
    """Baseline B2: simple 2D-conv stack over mel-spectrograms, no graph/text."""

    def __init__(self, num_classes=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.conv(x))
