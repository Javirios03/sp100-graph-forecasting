import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT_LSTM(nn.Module):
    def __init__(
        self,
        in_feats,
        lstm_hidden=64,
        gat_hidden=64,
        num_classes=3,
        edge_dim=5,
        heads=4,
        dropout=0.2,
    ):
        super().__init__()

        # --- Temporal encoder ---
        self.lstm = nn.LSTM(
            input_size=in_feats,
            hidden_size=lstm_hidden,
            batch_first=True,
        )

        # --- Graph layers ---
        self.gat1 = GATConv(
            lstm_hidden,
            gat_hidden,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        self.gat2 = GATConv(
            gat_hidden * heads,
            gat_hidden,
            heads=1,
            edge_dim=edge_dim,
            dropout=dropout,
        )

        # --- Classifier ---
        self.mlp = nn.Sequential(
            nn.Linear(gat_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x: (N, W, F)
        """

        # --- LSTM ---
        lstm_out, _ = self.lstm(x)         # (N, W, H)
        h = lstm_out[:, -1, :]             # (N, H)

        # --- GAT ---
        h = self.gat1(h, edge_index, edge_attr)
        h = F.elu(h)

        h = self.gat2(h, edge_index, edge_attr)
        h = F.elu(h)

        # --- Output ---
        out = self.mlp(h)

        return out