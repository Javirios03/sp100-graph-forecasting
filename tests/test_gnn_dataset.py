import pytest
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


@pytest.fixture(scope="session")
def load_gnn_data():
    snapshots = pd.read_parquet(PROCESSED_DIR / "gnn_snapshots_index.parquet")
    X = np.load(PROCESSED_DIR / "X_gnn.npy")
    y = np.load(PROCESSED_DIR / "y_gnn.npy")
    return snapshots, X, y


def test_shapes(load_gnn_data):
    snapshots, X, y = load_gnn_data
    n_snap, n_nodes, window, n_feat = X.shape
    
    assert y.shape == (n_snap, n_nodes)
    assert n_nodes == 88
    assert window == 20
    assert n_feat == 7
    assert len(snapshots) == n_snap
    assert (snapshots["num_nodes"].unique() == [88]).all()
    assert (snapshots["split"].isin(["train", "val", "test"])).all()


def test_snapshot_alignment(load_gnn_data):
    snapshots, X, y = load_gnn_data
    n_snap, n_nodes, _, _ = X.shape
    
    # cada snapshot debe tener 88 nodos con targets válidos
    for i in range(min(50, n_snap)):  # muestreo rápido
        uniques = np.unique(y[i])
        assert any(cl in uniques for cl in [-1, 0, 1])
        assert np.isfinite(X[i]).all()  # no NaN en features
        assert (y[i] != -99).sum() > 70  # al menos 80% nodos con target válido


def test_train_val_no_invalid_targets(load_gnn_data):
    snapshots, X, y = load_gnn_data
    mask_train_val = snapshots["split"].isin(["train", "val"]).to_numpy()
    
    train_val_y = y[mask_train_val]
    assert (train_val_y != -99).all()
    assert np.isfinite(X[mask_train_val]).all()

def test_ticker_to_node_is_consistent():
    mapping = pd.read_parquet(PROCESSED_DIR / "ticker_to_node.parquet")
    assert mapping["ticker"].nunique() == 88
    assert mapping["node_id"].nunique() == 88
    assert mapping["node_id"].min() == 0
    assert mapping["node_id"].max() == 87

def test_corr_edges_valid_and_weight_ranges():
    edges = pd.read_parquet(PROCESSED_DIR / "graph_corr_edges.parquet")
    # Rango de nodos
    assert edges[["src", "dst"]].min().min() >= 0
    assert edges[["src", "dst"]].max().max() < 88

    # Tipos de arista
    assert set(edges["edge_type"].unique()) <= {"correlation", "sector"}

    corr = edges[edges["edge_type"] == "correlation"]["weight"]
    sect = edges[edges["edge_type"] == "sector"]["weight"]
    assert corr.min() >= 0.32 and corr.max() <= 1.0
    assert np.allclose(sect.unique(), 1.0)

def test_js_edges_valid_and_weight_distance_ranges():
    edges = pd.read_parquet(PROCESSED_DIR / "graph_div_edges.parquet")
    assert set(edges["edge_type"].unique()) <= {"js_divergence", "sector"}

    js = edges[edges["edge_type"] == "js_divergence"]
    sect = edges[edges["edge_type"] == "sector"]

    # Pesos
    assert js["weight"].min() >= 0.85 and js["weight"].max() <= 1.0
    assert np.allclose(sect["weight"].unique(), 1.0)

    # Distancias JS y de sector en el mismo espacio que edge_attr
    assert js["distance"].between(0.0, 0.2).all()
    assert np.allclose(sect["distance"].unique(), 0.0)

def test_gnn_snapshots_and_tensors_shapes():
    idx = pd.read_parquet(PROCESSED_DIR / "gnn_snapshots_index.parquet")
    X = np.load(PROCESSED_DIR / "X_gnn.npy")
    y = np.load(PROCESSED_DIR / "y_gnn.npy")

    # Shapes esperadas
    assert X.shape[0] == len(idx)
    assert y.shape[0] == len(idx)
    assert X.shape[1] == 88
    assert X.shape[2:] == (20, 7)
    assert y.shape[1] == 88

    # num_nodes en metadata coherente
    assert idx["num_nodes"].nunique() == 1
    assert idx["num_nodes"].iloc[0] == 88

def test_splits_match_temporal_and_panel():
    idx = pd.read_parquet(PROCESSED_DIR / "gnn_snapshots_index.parquet")
    panel = pd.read_parquet(PROCESSED_DIR / "panel_with_splits.parquet")

    # Para un subconjunto de fechas, los splits deben coincidir
    sample_dates = idx["date"].sample(10, random_state=42)
    for d in sample_dates:
        snap_split = idx.loc[idx["date"] == d, "split"].iloc[0]
        panel_splits = panel.loc[panel["date"] == d, "split"].unique()
        assert len(panel_splits) == 1
        assert panel_splits[0] == snap_split

def test_no_nans_in_X_gnn():
    X = np.load(PROCESSED_DIR / "X_gnn.npy")
    assert np.isfinite(X).all()
