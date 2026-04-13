import pytest
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
INTERIM_DIR = Path("data/interim")


def test_panel_chain():
    # panel base -> features -> targets -> splits
    base = pd.read_parquet(INTERIM_DIR / "base_panel.parquet")
    features = pd.read_parquet(INTERIM_DIR / "features_panel.parquet")
    targets = pd.read_parquet(PROCESSED_DIR / "panel_with_targets.parquet")
    splits = pd.read_parquet(PROCESSED_DIR / "panel_with_splits.parquet")
    
    # ticker consistente
    n_tickers = base["ticker"].nunique()
    assert features["ticker"].nunique() == n_tickers
    assert targets["ticker"].nunique() == n_tickers
    assert splits["ticker"].nunique() == n_tickers

    # targets válidos (acepta tanto int como float)
    valid_targets = splits["target_class"].dropna()
    assert valid_targets.isin([-1, -1.0, 0, 0.0, 1, 1.0]).all()
    assert splits["split"].isin(["train", "val", "test"]).all()