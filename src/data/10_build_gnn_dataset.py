import numpy as np
import pandas as pd

from src.utils import PROCESSED_DIR
from config import WINDOW, FEATURE_COLS


def validate_input(df: pd.DataFrame, mapping: pd.DataFrame) -> None:
    required_cols = ["ticker", "date", "split", "target_class"] + FEATURE_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "ticker" not in mapping.columns or "node_id" not in mapping.columns:
        raise ValueError("ticker_to_node mapping is malformed.")


def main():
    df = pd.read_parquet(PROCESSED_DIR / "panel_with_splits.parquet")
    df["date"] = pd.to_datetime(df["date"])

    mapping = pd.read_parquet(PROCESSED_DIR / "ticker_to_node.parquet")
    validate_input(df, mapping)

    ticker_to_node = dict(zip(mapping["ticker"], mapping["node_id"]))
    all_tickers = mapping.sort_values("node_id")["ticker"].tolist()
    n_nodes = len(all_tickers)
    n_features = len(FEATURE_COLS)

    snapshot_rows = []
    X_snapshots = []
    y_snapshots = []

    grouped = {
        ticker: grp.sort_values("date").reset_index(drop=True).copy()
        for ticker, grp in df.groupby("ticker", sort=True)
    }

    common_dates = sorted(set.intersection(*[
        set(grp["date"].iloc[WINDOW - 1 :].tolist())
        for grp in grouped.values()
    ]))

    for snapshot_id, date_ref in enumerate(common_dates):
        X_nodes = np.full((n_nodes, WINDOW, n_features), np.nan, dtype=np.float32)
        y_nodes = np.full((n_nodes,), -99, dtype=np.int64)
        split_value = None
        valid_snapshot = True

        for ticker in all_tickers:
            grp = grouped[ticker]
            idx_list = grp.index[grp["date"] == date_ref].tolist()

            if not idx_list:
                valid_snapshot = False
                break

            idx = idx_list[0]
            if idx < WINDOW - 1:
                valid_snapshot = False
                break

            window_df = grp.iloc[idx - WINDOW + 1 : idx + 1]
            row_now = grp.iloc[idx]
            node_id = ticker_to_node[ticker]

            if window_df[FEATURE_COLS].isna().any().any():
                valid_snapshot = False
                break
            if pd.isna(row_now["target_class"]):
                valid_snapshot = False
                break

            X_nodes[node_id] = window_df[FEATURE_COLS].to_numpy(dtype=np.float32)
            y_nodes[node_id] = int(row_now["target_class"])

            if split_value is None:
                split_value = row_now["split"]
            elif split_value != row_now["split"]:
                valid_snapshot = False
                break

        if not valid_snapshot:
            continue

        X_snapshots.append(X_nodes)
        y_snapshots.append(y_nodes)
        snapshot_rows.append(
            {
                "snapshot_id": snapshot_id,
                "date": date_ref,
                "split": split_value,
                "num_nodes": n_nodes,
            }
        )

    if not X_snapshots:
        raise ValueError("No valid GNN snapshots were created.")

    X = np.stack(X_snapshots, axis=0)
    y = np.stack(y_snapshots, axis=0)
    snapshot_index = pd.DataFrame(snapshot_rows)

    np.save(PROCESSED_DIR / "X_gnn.npy", X)
    np.save(PROCESSED_DIR / "y_gnn.npy", y)
    snapshot_index.to_parquet(PROCESSED_DIR / "gnn_snapshots_index.parquet", index=False)

    print(f"Saved X_gnn with shape {X.shape}")
    print(f"Saved y_gnn with shape {y.shape}")
    print(f"Saved snapshot index with {len(snapshot_index)} rows")


if __name__ == "__main__":
    main()