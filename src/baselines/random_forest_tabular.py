import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import PROCESSED_DIR

df = pd.read_parquet(PROCESSED_DIR / "tabular_dataset.parquet")

# Preparar X/y
meta_cols = ["ticker", "date", "split", "sector", "market_cap", "target_class"]
feature_cols = [c for c in df.columns if c not in meta_cols]
X = df[feature_cols].fillna(0)
y = df["target_class"]

print(f"Dataset shape: {X.shape}")
print("Target distribution:")
print(y.value_counts(normalize=True).sort_index().round(3))

# RF baseline
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# CV en train
train_mask = df["split"] == "train"
X_train, y_train = X[train_mask], y[train_mask]

cv_scores = cross_val_score(rf, X_train, y_train, cv=StratifiedKFold(5), 
                           scoring="f1_macro", n_jobs=-1)
print(f"\nF1-macro CV train: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Fit final y predicción val
rf.fit(X_train, y_train)
val_mask = df["split"] == "val"
X_val, y_val = X[val_mask], y[val_mask]
y_pred_val = rf.predict(X_val)

print("\nF1-macro val:")
print(classification_report(y_val, y_pred_val))
print("\nConfusion matrix val:")
print(confusion_matrix(y_val, y_pred_val))

# Feature importance top 10
importances = pd.Series(rf.feature_importances_, index=feature_cols)
print("\nTop 10 features:")
print(importances.sort_values(ascending=False).head(10))