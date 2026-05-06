from pathlib import Path

START_DATE = "2021-01-01"  # YYYY-MM-DD
END_DATE = "2025-12-31"  # YYYY-MM-DD
INTERVAL = "1d"
PATH_TICKERS = Path("src/eda/sp100_output/constant_companies.csv")

# Target parameters
EPSILON = 0.01  # Threshold for movement classification
HORIZON = 5  # Number of days to look ahead for target calculation
WINDOW = 20  # Number of past days to use for feature calculation

# Split dates
TRAIN_END = "2023-12-31"
VAL_END = "2024-12-31"

# Features to use in each model
FEATURE_COLS = [
    "adj_close",
    "volume_norm",
    "log_ret_1d",
    "ma_5",
    "ma_20",
    "rsi_14",
    "roll_vol_20",
]

# Graph construction parameters
TOP_K_CORR = 5  # Number of top correlated neighbors to connect in the correlation graph
TOP_K_DIV = 5  # Number of top similar neighbors to connect in the JS divergence graph
N_BINS = 30  # Number of bins to use when creating histograms for JS divergence calculation

MAX_SECTOR_EDGES = 5 # Maximum number of edges to add based on sector similarity (to avoid overly dense graphs)

# GNN Model
GNN_MODEL = {
    "lstm_hidden": 64,
    "gat_hidden": 64,
    "heads": 4,
    "dropout": 0.3,
    "lstm_layers": 2,
    "bidirectional": True,
}

TRAINING = {
    "lr": 6e-4,
    "weight_decay": 1e-4,
    "epochs": 150,
    "batch_size": 1,  # Snapshot-based training
    "grad_clip": 1.0,
    "early_stopping": False,
    "patience": 40,
    "seed": 42,
}

GRAPH_TYPE = "corr"  # "corr" or "js"
USE_EDGE_ATTR = True  # Whether to use edge attributes in the GNN
