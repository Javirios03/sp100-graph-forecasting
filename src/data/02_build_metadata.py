from pathlib import Path
import pandas as pd
import yfinance as yf
from time import sleep

from src.utils import TICKERS, RAW_DIR


def fetch_metadata(ticker):
    tk = yf.Ticker(ticker)
    info = tk.info
    return {
        "ticker": ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "short_name": info.get("shortName"),
        "currency": info.get("currency"),
    }


def main():
    rows = []
    for ticker in TICKERS:
        try:
            rows.append(fetch_metadata(ticker))
            sleep(0.2)
        except Exception as e:
            rows.append(
                {
                    "ticker": ticker,
                    "sector": None,
                    "industry": None,
                    "market_cap": None,
                    "short_name": None,
                    "currency": None,
                }
            )
            print(f"Warning: metadata failed for {ticker}: {e}")

    df = pd.DataFrame(rows)
    output_path = RAW_DIR / "ticker_metadata.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Saved metadata to {output_path}")


if __name__ == "__main__":
    main()