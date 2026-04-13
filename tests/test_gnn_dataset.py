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