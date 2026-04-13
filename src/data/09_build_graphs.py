import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from src.utils import PROCESSED_DIR
from config import TOP_K_CORR, TOP_K_DIV, N_BINS


def validate_input(df: pd.DataFrame) -> None:
    required_cols = ["ticker", "date", "split", "sector", "log_ret_1d"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_ticker_mapping(df: pd.DataFrame) -> pd.DataFrame:
    tickers = sorted(df["ticker"].dropna().unique().tolist())
    mapping = pd.DataFrame(
        {"ticker": tickers, "node_id": range(len(tickers))}
    )
    return mapping


def build_sector_edges(meta: pd.DataFrame) -> set:
    edges = set()
    for sector, group in meta.groupby("sector"):
        nodes = group["node_id"].tolist()
        for i in nodes:
            for j in nodes:
                if i != j:
                    edges.add((i, j))
    return edges


def build_corr_edges(train_df: pd.DataFrame, mapping: pd.DataFrame, top_k: int) -> pd.DataFrame:
    pivot = train_df.pivot(index="date", columns="ticker", values="log_ret_1d")
    corr = pivot.corr()

    ticker_to_node = dict(zip(mapping["ticker"], mapping["node_id"]))
    edges = []

    for ticker in corr.columns:
        s = corr[ticker].drop(labels=[ticker]).dropna()
        top_neighbors = s.sort_values(ascending=False).head(top_k)

        src = ticker_to_node[ticker]
        for nbr_ticker, weight in top_neighbors.items():
            dst = ticker_to_node[nbr_ticker]
            edges.append(
                {
                    "src": src,
                    "dst": dst,
                    "src_ticker": ticker,
                    "dst_ticker": nbr_ticker,
                    "weight": float(weight),
                    "edge_type": "correlation",
                }
            )

    return pd.DataFrame(edges)


def build_js_edges(train_df: pd.DataFrame, mapping: pd.DataFrame, top_k: int, n_bins: int) -> pd.DataFrame:
    ticker_to_node = dict(zip(mapping["ticker"], mapping["node_id"]))
    returns_by_ticker = {
        ticker: grp["log_ret_1d"].dropna().values
        for ticker, grp in train_df.groupby("ticker")
    }

    all_returns = np.concatenate([x for x in returns_by_ticker.values() if len(x) > 0])
    hist_range = (np.nanmin(all_returns), np.nanmax(all_returns))

    histograms = {}
    for ticker, values in returns_by_ticker.items():
        hist, _ = np.histogram(values, bins=n_bins, range=hist_range, density=True)
        hist = hist + 1e-12
        hist = hist / hist.sum()
        histograms[ticker] = hist

    rows = []
    tickers = sorted(histograms.keys())

    for ticker in tickers:
        distances = []
        for other in tickers:
            if ticker == other:
                continue
            dist = float(jensenshannon(histograms[ticker], histograms[other]))
            distances.append((other, dist))

        top_neighbors = sorted(distances, key=lambda x: x[1])[:top_k]
        src = ticker_to_node[ticker]

        for nbr_ticker, dist in top_neighbors:
            dst = ticker_to_node[nbr_ticker]
            rows.append(
                {
                    "src": src,
                    "dst": dst,
                    "src_ticker": ticker,
                    "dst_ticker": nbr_ticker,
                    "weight": float(1.0 / (1.0 + dist)),
                    "distance": dist,
                    "edge_type": "js_divergence",
                }
            )

    return pd.DataFrame(rows)


def add_sector_edges(edge_df: pd.DataFrame, mapping: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    sector_edges = build_sector_edges(meta)
    existing = set(zip(edge_df["src"], edge_df["dst"]))

    rows = edge_df.to_dict("records")
    node_to_ticker = dict(zip(mapping["node_id"], mapping["ticker"]))

    for src, dst in sector_edges:
        if (src, dst) not in existing:
            rows.append(
                {
                    "src": src,
                    "dst": dst,
                    "src_ticker": node_to_ticker[src],
                    "dst_ticker": node_to_ticker[dst],
                    "weight": 1.0,
                    "edge_type": "sector",
                }
            )

    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(PROCESSED_DIR / "panel_with_splits.parquet")
    df["date"] = pd.to_datetime(df["date"])

    validate_input(df)

    mapping = build_ticker_mapping(df)
    meta = (
        df[["ticker", "sector"]]
        .drop_duplicates(subset=["ticker"])
        .merge(mapping, on="ticker", how="inner")
    )

    train_df = df[df["split"] == "train"].copy()

    corr_edges = build_corr_edges(train_df, mapping, TOP_K_CORR)
    corr_edges = add_sector_edges(corr_edges, mapping, meta)

    js_edges = build_js_edges(train_df, mapping, TOP_K_DIV, N_BINS)
    js_edges = add_sector_edges(js_edges, mapping, meta)

    mapping.to_parquet(PROCESSED_DIR / "ticker_to_node.parquet", index=False)
    corr_edges.to_parquet(PROCESSED_DIR / "graph_corr_edges.parquet", index=False)
    js_edges.to_parquet(PROCESSED_DIR / "graph_div_edges.parquet", index=False)

    print(f"Saved ticker mapping with {len(mapping)} nodes")
    print(f"Saved correlation graph with {len(corr_edges)} edges")
    print(f"Saved divergence graph with {len(js_edges)} edges")


if __name__ == "__main__":
    main()