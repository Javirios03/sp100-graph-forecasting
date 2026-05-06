import importlib

import numpy as np
import pandas as pd


graphs = importlib.import_module("src.data.09_build_graphs")


def test_graph_edges_include_ticker_labels():
    df = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "date": pd.to_datetime(["2021-01-01", "2021-01-02"] * 3),
            "split": ["train"] * 6,
            "sector": ["Tech", "Tech", "Tech", "Tech", "Finance", "Finance"],
            "log_ret_1d": [0.01, 0.02, 0.02, 0.03, -0.01, -0.02],
        }
    )
    mapping = graphs.build_ticker_mapping(df)

    corr_edges = graphs.build_corr_edges(df, mapping, top_k=1)

    assert {"src_ticker", "dst_ticker"}.issubset(corr_edges.columns)
    assert corr_edges["src_ticker"].notna().all()
    assert corr_edges["dst_ticker"].notna().all()
    assert corr_edges["weight"].between(0.0, 1.0).all()


def test_sector_edges_use_configurable_limit():
    meta = pd.DataFrame(
        {
            "sector": ["Tech", "Tech", "Tech", "Finance"],
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "node_id": [0, 1, 2, 3],
        }
    )

    sector_edges = graphs.build_sector_edges(meta, max_edges_per_node=1)

    assert {"src_ticker", "dst_ticker"}.issubset(sector_edges.columns)
    assert np.allclose(sector_edges["distance"].unique(), 0.0)
    assert sector_edges.groupby("src").size().max() == 1
