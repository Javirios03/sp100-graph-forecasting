import numpy as np
import pandas as pd

from src.utils import PROCESSED_DIR
from config import WINDOW, FEATURE_COLS


def validate_input(df: pd.DataFrame) -> None:
    required_cols = ["ticker", "date", "target_class", "split"] + FEATURE_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate (ticker, date) rows found.")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("'date' must be datetime.")


def main():
    input_path = PROCESSED_DIR / "panel_with_splits.parquet"

    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])

    validate_input(df)

    X_list = []
    y_list = []
    meta_rows = []

    for ticker, group in df.groupby("ticker", sort=True):
        group = group.sort_values("date").reset_index(drop=True).copy()

        for idx in range(WINDOW - 1, len(group)):
            row_now = group.iloc[idx]
            window_df = group.iloc[idx - WINDOW + 1 : idx + 1]

            if window_df[FEATURE_COLS].isna().any().any():
                continue
            if pd.isna(row_now["target_class"]):
                continue

            X_window = window_df[FEATURE_COLS].to_numpy(dtype=np.float32)
            y_value = int(row_now["target_class"])

            X_list.append(X_window)
            y_list.append(y_value)
            meta_rows.append(
                {
                    "sample_id": len(meta_rows),
                    "ticker": ticker,
                    "date": row_now["date"],
                    "split": row_now["split"],
                    "target_class": y_value,
                }
            )

    if not X_list:
        raise ValueError("Temporal dataset is empty.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    meta = pd.DataFrame(meta_rows)

    if "ticker" not in meta.columns:
        raise ValueError("Ticker missing from temporal metadata.")

    np.save(PROCESSED_DIR / "X_temporal.npy", X)
    np.save(PROCESSED_DIR / "y_temporal.npy", y)
    meta.to_parquet(PROCESSED_DIR / "temporal_index.parquet", index=False)

    print(f"Saved X_temporal with shape {X.shape}")
    print(f"Saved y_temporal with shape {y.shape}")
    print(f"Saved temporal_index with {len(meta)} rows")


if __name__ == "__main__":
    main()