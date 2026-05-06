import pytest
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

REQUIRED_FILES = [
    PROCESSED_DIR / "ticker_to_node.parquet",
    PROCESSED_DIR / "graph_corr_edges.parquet",
    PROCESSED_DIR / "graph_div_edges.parquet",
]

pytestmark = pytest.mark.skipif(
    any(not path.exists() for path in REQUIRED_FILES),
    reason="integration graph artifacts are missing; run uv run python -m src.data.run_pipeline --from-step 09_build_graphs",
)


@pytest.fixture(scope="session")
def load_data():
    mapping = pd.read_parquet(PROCESSED_DIR / "ticker_to_node.parquet")
    corr_edges = pd.read_parquet(PROCESSED_DIR / "graph_corr_edges.parquet")
    js_edges = pd.read_parquet(PROCESSED_DIR / "graph_div_edges.parquet")
    return mapping, corr_edges, js_edges


def test_mapping_consistency(load_data):
    mapping, _, _ = load_data
    assert mapping["ticker"].nunique() == 88
    assert mapping["node_id"].nunique() == 88
    assert mapping["node_id"].min() == 0
    assert mapping["node_id"].max() == 87
    assert len(mapping) == 88
    assert mapping["ticker"].isna().sum() == 0
    assert mapping["node_id"].is_monotonic_increasing


def test_edges_node_ids(load_data):
    mapping, corr_edges, js_edges = load_data
    valid_ids = set(mapping["node_id"].tolist())
    
    for edges, name in [(corr_edges, "corr"), (js_edges, "js")]:
        assert set(edges["src"]).issubset(valid_ids)
        assert set(edges["dst"]).issubset(valid_ids)
        assert len(edges) > 100  # debe haber suficientes edges
        assert edges["src"].min() >= 0
        assert edges["dst"].max() < 88


def test_edges_ticker_mapping(load_data):
    mapping, corr_edges, js_edges = load_data
    node_to_ticker = dict(zip(mapping["node_id"], mapping["ticker"]))
    
    for edges, name in [(corr_edges.head(1000), "corr"), (js_edges.head(1000), "js")]:
        check = edges.assign(
            src_ticker_map=lambda d: d["src"].map(node_to_ticker),
            dst_ticker_map=lambda d: d["dst"].map(node_to_ticker),
        )
        assert (check["src_ticker"] == check["src_ticker_map"]).all()
        assert (check["dst_ticker"] == check["dst_ticker_map"]).all()
        assert check["src_ticker"].isna().sum() == 0
        assert check["dst_ticker"].isna().sum() == 0
