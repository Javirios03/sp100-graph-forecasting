import numpy as np
import pandas as pd

from src.utils import INTERIM_DIR, TICKERS


def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_features(group: pd.DataFrame, ticker: str) -> pd.DataFrame:
    group = group.sort_values("date").copy()

    group["log_ret_1d"] = np.log(group["adj_close"] / group["adj_close"].shift(1))
    group["adj_close_ret_1d"] = group["adj_close"].pct_change()

    group["ma_5"] = group["adj_close"].rolling(5).mean() / group["adj_close"]
    group["ma_20"] = group["adj_close"].rolling(20).mean() / group["adj_close"]

    group["volume_ma_20"] = group["volume"].rolling(20).mean()
    group["volume_norm"] = group["volume"] / group["volume_ma_20"]

    group["roll_vol_20"] = group["log_ret_1d"].rolling(20).std()
    group["rsi_14"] = compute_rsi(group["adj_close"], window=14)

    group["ticker"] = ticker

    return group


def main():
    df = pd.read_parquet(INTERIM_DIR / "base_panel.parquet")
    df["date"] = pd.to_datetime(df["date"])

    if "ticker" not in df.columns:
        raise ValueError("Expected 'ticker' column in the base panel")
    
    out = []
    for ticker, group in df.groupby("ticker", sort=True):
        out.append(add_features(group, ticker))

    features = pd.concat(out, axis=0, ignore_index=True)

    if "ticker" not in features.columns:
        raise ValueError("Expected 'ticker' column in the features panel")
    
    # Since the currency is the same for all tickers, we can drop it
    features = features.drop(columns=["currency"], errors="ignore")

    output_path = INTERIM_DIR / "features_panel.parquet"
    features.to_parquet(output_path, index=False)
    print(f"Saved features to {output_path}")


if __name__ == "__main__":
    main()