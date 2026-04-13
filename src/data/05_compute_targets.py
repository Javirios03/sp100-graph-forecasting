import pandas as pd

from src.utils import INTERIM_DIR, PROCESSED_DIR
from config import EPSILON, HORIZON


def classify_return(x, epsilon):
    if pd.isna(x):
        return pd.NA
    if x > epsilon:
        return 1
    if x < -epsilon:
        return -1
    return 0


def add_targets(group: pd.DataFrame, ticker: str) -> pd.DataFrame:
    group = group.sort_values("date").copy()

    future_returns = [group["log_ret_1d"].shift(-i) for i in range(1, HORIZON + 1)]
    group["future_log_ret_5d"] = sum(future_returns)
    group["target_class"] = group["future_log_ret_5d"].apply(lambda x: classify_return(x, EPSILON))

    group["ticker"] = ticker

    return group


def main():
    df = pd.read_parquet(INTERIM_DIR / "features_panel.parquet")
    df["date"] = pd.to_datetime(df["date"])

    out = []
    for ticker, group in df.groupby("ticker", sort=True):
        out.append(add_targets(group, ticker))

    result = pd.concat(out, axis=0, ignore_index=True)

    output_path = PROCESSED_DIR / "panel_with_targets.parquet"
    result.to_parquet(output_path, index=False)
    print(f"Saved targets to {output_path}")
    # print(f"Rows: {len(result)}, Columns: {len(result.columns)}")
    # print("Target class distribution (excluding NANs):")
    # print(result["target_class"].value_counts(dropna=True).sort_index())


if __name__ == "__main__":
    main()