import pandas as pd

from src.utils import PROCESSED_DIR
from config import TRAIN_END, VAL_END


def assign_split(date):
    if date <= pd.Timestamp(TRAIN_END):
        return "train"
    if date <= pd.Timestamp(VAL_END):
        return "val"
    return "test"


def main():
    df = pd.read_parquet(PROCESSED_DIR / "panel_with_targets.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["split"] = df["date"].apply(assign_split)

    output_path = PROCESSED_DIR / "panel_with_splits.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Saved splits to {output_path}")


if __name__ == "__main__":
    main()