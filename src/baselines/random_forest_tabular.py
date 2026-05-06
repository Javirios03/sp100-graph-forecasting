import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.utils import PROCESSED_DIR


CLASS_NAMES = ["down", "neutral", "up"]
CLASS_VALUES = [-1, 0, 1]


def compute_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_VALUES,
        zero_division=0,
    )
    precision_macro, recall_macro, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            class_name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, class_name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_VALUES).tolist(),
        "labels": CLASS_NAMES,
    }

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

test_mask = df["split"] == "test"
X_test, y_test = X[test_mask], y[test_mask]
y_pred_test = rf.predict(X_test)

print("\nF1-macro val:")
print(classification_report(y_val, y_pred_val))
print("\nConfusion matrix val:")
print(confusion_matrix(y_val, y_pred_val))

print("\nF1-macro test:")
print(classification_report(y_test, y_pred_test))
print("\nConfusion matrix test:")
print(confusion_matrix(y_test, y_pred_test))

# Feature importance top 10
importances = pd.Series(rf.feature_importances_, index=feature_cols)
print("\nTop 10 features:")
print(importances.sort_values(ascending=False).head(10))

results = {
    "experiment": "random_forest",
    "model_type": "random_forest",
    "best_val_f1": float(f1_score(y_val, y_pred_val, average="macro", zero_division=0)),
    "test_f1": float(f1_score(y_test, y_pred_test, average="macro", zero_division=0)),
    "metrics": {
        "val": compute_metrics(y_val, y_pred_val),
        "test": compute_metrics(y_test, y_pred_test),
    },
    "cv_train_f1_macro_mean": float(cv_scores.mean()),
    "cv_train_f1_macro_std": float(cv_scores.std()),
    "feature_importance": {
        feature: float(value)
        for feature, value in importances.sort_values(ascending=False).items()
    },
}

Path("results").mkdir(exist_ok=True)
with open(Path("results") / "random_forest.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)
