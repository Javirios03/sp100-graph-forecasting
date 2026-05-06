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
        lstm_layers=2,
        bidirectional=True,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        lstm_out_dim = lstm_hidden * 2 if bidirectional else lstm_hidden

        # Temporal encoder
        self.lstm = nn.LSTM(
            input_size=in_feats,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Temporal attention over LSTM output
        self.temporal_attn = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_dim // 2, 1),
        )

        # Skip connection
        self.skip = nn.Linear(lstm_out_dim, gat_hidden)

        # Graph layers
        self.gat1 = GATConv(
            lstm_out_dim,
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

        # Batch Normalizations
        self.bn1 = nn.BatchNorm1d(gat_hidden * heads)
        self.bn2 = nn.BatchNorm1d(gat_hidden)

        # Classifier
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

        # LSTM
        lstm_out, _ = self.lstm(x)         # (N, W, H)
        
        # Temporal attention
        attn_scores = self.temporal_attn(lstm_out)  # (N, W, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        h = (lstm_out * attn_weights).sum(dim=1)  # (N, H * 2 if bidirectional else H)

        h_skip = self.skip(h)  # (N, gat_hidden)

        # GAT
        h = self.gat1(h, edge_index, edge_attr)
        h = self.bn1(h)
        h = F.elu(h)

        h = self.gat2(h, edge_index, edge_attr)
        h = self.bn2(h)
        h = F.elu(h)

        # Recover residual
        h = h + h_skip  # (N, gat_hidden)

        # Output
        out = self.mlp(h)

        return out