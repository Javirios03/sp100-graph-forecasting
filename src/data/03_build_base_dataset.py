from pathlib import Path
import pandas as pd

from src.utils import RAW_DIR, INTERIM_DIR

def main():
    prices = pd.read_parquet(RAW_DIR / "prices_raw.parquet")
    meta = pd.read_parquet(RAW_DIR / "ticker_metadata.parquet")

    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.drop_duplicates(subset=["date", "ticker"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    panel = prices.merge(meta, on="ticker", how="left")

    panel = panel[panel["adj_close"].notna()].copy()
    panel = panel[panel["adj_close"] > 0].copy()
    panel = panel[panel["volume"].notna()].copy()

    output_path = INTERIM_DIR / "base_panel.parquet"
    panel.to_parquet(output_path, index=False)
    print(f"Saved {len(panel)} rows to {output_path}")


if __name__ == "__main__":
    main()