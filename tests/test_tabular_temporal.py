import pytest
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def test_tabular_dataset():
    df = pd.read_parquet(PROCESSED_DIR / "tabular_dataset.parquet")
    
    assert df["ticker"].nunique() > 50
    assert "target_class" in df.columns
    assert df["target_class"].isna().sum() == 0
    assert df["target_class"].isin([-1, 0, 1]).all()
    assert "split" in df.columns
    assert df["split"].isin(["train", "val", "test"]).all()
    
    # features esperadas
    feature_cols = [c for c in df.columns if "_mean_20" in c or "_last" in c]
    assert len(feature_cols) == 14  # 7 features * 2 estadísticas
    assert df[feature_cols].isna().sum().sum() == 0


def test_temporal_dataset():
    meta = pd.read_parquet(PROCESSED_DIR / "temporal_index.parquet")
    X = np.load(PROCESSED_DIR / "X_temporal.npy")
    y = np.load(PROCESSED_DIR / "y_temporal.npy")
    
    assert len(meta) == len(X) == len(y)
    assert X.shape[1:] == (20, 7)
    assert y.shape == (len(meta),)
    assert meta["target_class"].isin([-1, 0, 1]).all()
    assert meta["split"].isin(["train", "val", "test"]).all()
    assert np.isfinite(X).all()