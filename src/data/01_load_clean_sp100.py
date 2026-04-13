from tracemalloc import start

import yfinance as yf
import pandas as pd
# from pathlib import Path

from config import START_DATE, END_DATE, INTERVAL
from src.utils import RAW_DIR, TICKERS

def download_sp100_data(tickers, start_date=START_DATE, end_date=END_DATE, interval=INTERVAL):
    """Download historical stock data for the S&P 100 tickers.

    Args:
        tickers (list): A list of S&P 100 tickers.
        start_date (str): The start date for the data in YYYY-MM-DD format.
        end_date (str): The end date for the data in YYYY-MM-DD format.
        interval (str): The data interval (e.g., '1d', '1h').
    Returns:
        df: A DataFrame containing the historical stock data for the S&P 100 tickers.
    """
    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=True,
        group_by='column',
        threads=True
    )

    if df.empty:
        print("No data downloaded. Please check the tickers and date range.")
        return None
    
    df = df.stack(level=1).reset_index()
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    df = df.rename(columns={'adj_close': 'adj_close', 'level_1': 'ticker'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
    
    expected_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns in the downloaded data: {missing_cols}")
    return df

if __name__ == "__main__":
    print("Using a total of {} tickers: {}".format(len(TICKERS), TICKERS))

    output_path = RAW_DIR / "prices_raw.parquet"

    df = download_sp100_data(tickers=TICKERS)
    if df is not None:
        print(f"Downloaded data for {len(df['ticker'].unique())} tickers and {len(df)} rows.")
        # Why Parquet instead of CSV? Parquet is a columnar storage format that is more efficient for large datasets, both in terms of storage space and read/write performance. It also preserves data types better than CSV.
        df.to_parquet(
            output_path,
            index=False,
            engine='pyarrow'
        )
        print(f"Data saved to {output_path}")