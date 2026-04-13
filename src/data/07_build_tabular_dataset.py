import pandas as pd
import numpy as np

from src.utils import PROCESSED_DIR
from config import WINDOW, FEATURE_COLS


def build_examples(group: pd.DataFrame, ticker: str) -> pd.DataFrame:
    group = group.sort_values("date").reset_index(drop=True).copy()
    rows = []

    for idx in range(WINDOW - 1, len(group)):
        row_now = group.iloc[idx]
        window_df = group.iloc[idx - WINDOW + 1 : idx + 1]

        if window_df[FEATURE_COLS].isna().any().any():
            continue
        if pd.isna(row_now["target_class"]):
            continue

        row = {
            "ticker": row_now["ticker"],
            "date": row_now["date"],
            "split": row_now["split"],
            "target_class": row_now["target_class"],
            "sector": row_now["sector"],
            "market_cap": row_now["market_cap"],
        }

        for col in FEATURE_COLS:
            row[f"{col}_mean_{WINDOW}"] = window_df[col].mean()
            row[f"{col}_std_{WINDOW}"] = window_df[col].std()
            row[f"{col}_last"] = window_df[col].iloc[-1]
            row[f"{col}_min_{WINDOW}"] = window_df[col].min()
            row[f"{col}_max_{WINDOW}"] = window_df[col].max()

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    df = pd.read_parquet(PROCESSED_DIR / "panel_with_splits.parquet")
    df["date"] = pd.to_datetime(df["date"])

    out = []
    for ticker, group in df.groupby("ticker"):
        out.append(build_examples(group, ticker))

    result = pd.concat(out, axis=0, ignore_index=True)

    output_path = PROCESSED_DIR / "tabular_dataset.parquet"
    result.to_parquet(output_path, index=False)
    print(f"Saved tabular dataset to {output_path}")


if __name__ == "__main__":
    main()