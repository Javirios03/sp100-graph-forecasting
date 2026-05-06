import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_OUTPUT_DIR = Path("results")


def load_tensorboard_history(run_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return []

    event_files = list(run_dir.glob("events.out.tfevents*"))
    if not event_files:
        return []

    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))

    values_by_epoch = {}
    if "Loss/train" in scalar_tags:
        for event in accumulator.Scalars("Loss/train"):
            values_by_epoch.setdefault(event.step, {})["loss"] = float(event.value)
    if "F1/val" in scalar_tags:
        for event in accumulator.Scalars("F1/val"):
            values_by_epoch.setdefault(event.step, {})["val_f1"] = float(event.value)

    return [
        {"epoch": int(epoch), **values}
        for epoch, values in sorted(values_by_epoch.items())
    ]


def load_tensorboard_test_f1(run_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None

    if not list(run_dir.glob("events.out.tfevents*")):
        return None

    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    if "F1/test" not in scalar_tags:
        return None

    scalars = accumulator.Scalars("F1/test")
    return float(scalars[-1].value) if scalars else None


def infer_model_name(result, fallback_name):
    model_type = result.get("model_type")
    if model_type:
        return str(model_type).upper()

    graph_type = result.get("graph_type")
    if graph_type:
        graph_label = "Correlation" if graph_type == "corr" else graph_type.upper()
        return f"GNN ({graph_label})"

    name = result.get("experiment", fallback_name)
    return name.replace("_", " ").title()


def normalize_result(result, fallback_name):
    history = result.get("history", [])
    best_val = result.get("best_val_f1")
    test_f1 = result.get("test_f1")

    metrics = result.get("metrics", {})
    if test_f1 is None and "test" in metrics:
        test_f1 = metrics["test"].get("f1_macro")
    if best_val is None and history:
        val_values = [row.get("val_f1") for row in history if row.get("val_f1") is not None]
        best_val = max(val_values) if val_values else None

    return {
        "id": fallback_name,
        "experiment": result.get("experiment", fallback_name),
        "model": infer_model_name(result, fallback_name),
        "best_epoch": result.get("best_epoch"),
        "best_val_f1": best_val,
        "test_f1": test_f1,
        "history": history,
        "metrics": metrics,
    }


def load_results(results_dir, runs_dir):
    loaded = {}

    for path in sorted(results_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            loaded[path.stem] = normalize_result(json.load(f), path.stem)

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.exists() else []:
        if run_dir.name in loaded:
            continue
        history = load_tensorboard_history(run_dir)
        test_f1 = load_tensorboard_test_f1(run_dir)
        if history or test_f1 is not None:
            loaded[run_dir.name] = normalize_result(
                {
                    "experiment": run_dir.name,
                    "history": history,
                    "test_f1": test_f1,
                },
                run_dir.name,
            )

    return list(loaded.values())


def save_current_figure(output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    return png_path, pdf_path


def build_history_frame(results):
    rows = []
    for result in results:
        for point in result["history"]:
            epoch = point.get("epoch")
            if epoch is None:
                continue
            rows.append(
                {
                    "model": result["model"],
                    "experiment": result["experiment"],
                    "epoch": epoch,
                    "train_loss": point.get("loss"),
                    "val_f1": point.get("val_f1"),
                }
            )
    return pd.DataFrame(rows)


def plot_validation_f1(history_df, output_dir):
    df = history_df.dropna(subset=["val_f1"])
    if df.empty:
        return []

    plt.figure(figsize=(8.5, 4.8))
    sns.lineplot(data=df, x="epoch", y="val_f1", hue="model", linewidth=2.2)
    best_rows = df.loc[df.groupby("model")["val_f1"].idxmax()]
    sns.scatterplot(
        data=best_rows,
        x="epoch",
        y="val_f1",
        hue="model",
        legend=False,
        s=90,
        edgecolor="black",
        linewidth=0.7,
    )
    plt.title("Validation macro-F1 over training")
    plt.xlabel("Epoch")
    plt.ylabel("Validation macro-F1")
    plt.ylim(bottom=0)
    plt.grid(alpha=0.25)
    return save_current_figure(output_dir, "validation_f1_by_epoch")


def plot_training_loss(history_df, output_dir):
    df = history_df.dropna(subset=["train_loss"])
    if df.empty:
        return []

    plt.figure(figsize=(8.5, 4.8))
    sns.lineplot(data=df, x="epoch", y="train_loss", hue="model", linewidth=2.2)
    plt.title("Training loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.grid(alpha=0.25)
    return save_current_figure(output_dir, "training_loss_by_epoch")


def build_summary_frame(results):
    rows = []
    for result in results:
        rows.append(
            {
                "model": result["model"],
                "experiment": result["experiment"],
                "best_epoch": result["best_epoch"],
                "best_val_f1": result["best_val_f1"],
                "test_f1": result["test_f1"],
            }
        )
    return pd.DataFrame(rows)


def plot_f1_bars(summary_df, output_dir):
    df = summary_df.melt(
        id_vars=["model", "experiment"],
        value_vars=["best_val_f1", "test_f1"],
        var_name="metric",
        value_name="macro_f1",
    ).dropna(subset=["macro_f1"])
    if df.empty:
        return []

    df["metric"] = df["metric"].map(
        {"best_val_f1": "Best validation F1", "test_f1": "Test F1"}
    )

    plt.figure(figsize=(8.0, 4.8))
    ax = sns.barplot(data=df, x="model", y="macro_f1", hue="metric")
    ax.bar_label(ax.containers[0], fmt="%.3f", padding=3)
    if len(ax.containers) > 1:
        ax.bar_label(ax.containers[1], fmt="%.3f", padding=3)
    plt.title("Best validation and test macro-F1")
    plt.xlabel("")
    plt.ylabel("Macro-F1")
    plt.ylim(0, max(0.45, df["macro_f1"].max() * 1.22))
    plt.grid(axis="y", alpha=0.25)
    return save_current_figure(output_dir, "best_val_and_test_f1")


def plot_best_epoch(summary_df, output_dir):
    df = summary_df.dropna(subset=["best_epoch"])
    if df.empty:
        return []

    plt.figure(figsize=(7.0, 4.2))
    ax = sns.barplot(data=df, x="model", y="best_epoch")
    ax.bar_label(ax.containers[0], fmt="%.0f", padding=3)
    plt.title("Epoch selected by validation macro-F1")
    plt.xlabel("")
    plt.ylabel("Best epoch")
    plt.grid(axis="y", alpha=0.25)
    return save_current_figure(output_dir, "best_epoch_by_model")


def plot_per_class_f1(results, output_dir):
    rows = []
    for result in results:
        test_metrics = result.get("metrics", {}).get("test", {})
        for class_name, values in test_metrics.get("per_class", {}).items():
            rows.append(
                {
                    "model": result["model"],
                    "class": class_name,
                    "f1": values.get("f1"),
                }
            )
    df = pd.DataFrame(rows).dropna() if rows else pd.DataFrame()
    if df.empty:
        return []

    plt.figure(figsize=(8.0, 4.8))
    ax = sns.barplot(data=df, x="class", y="f1", hue="model")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3)
    plt.title("Test F1 by class")
    plt.xlabel("Class")
    plt.ylabel("F1")
    plt.ylim(0, max(0.45, df["f1"].max() * 1.25))
    plt.grid(axis="y", alpha=0.25)
    return save_current_figure(output_dir, "test_f1_by_class")


def plot_confusion_matrices(results, output_dir):
    saved = []
    for result in results:
        test_metrics = result.get("metrics", {}).get("test", {})
        matrix = test_metrics.get("confusion_matrix")
        labels = test_metrics.get("labels", ["down", "neutral", "up"])
        if not matrix:
            continue

        matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
        plt.figure(figsize=(5.2, 4.6))
        sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Test confusion matrix: {result['model']}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        stem = f"confusion_matrix_{result['experiment']}".replace(" ", "_")
        saved.extend(save_current_figure(output_dir, stem))
    return saved


def main():
    parser = argparse.ArgumentParser(description="Generate report figures from result JSONs and TensorBoard runs.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    results = load_results(args.results_dir, args.runs_dir)
    if not results:
        raise SystemExit("No result JSONs or TensorBoard runs found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    history_df = build_history_frame(results)
    summary_df = build_summary_frame(results)
    summary_path = args.output_dir / "model_results_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    saved_paths = [summary_path]
    if not history_df.empty:
        history_path = args.output_dir / "training_history.csv"
        history_df.to_csv(history_path, index=False)
        saved_paths.append(history_path)

    for paths in [
        plot_validation_f1(history_df, args.output_dir),
        plot_training_loss(history_df, args.output_dir),
        plot_f1_bars(summary_df, args.output_dir),
        plot_best_epoch(summary_df, args.output_dir),
        plot_per_class_f1(results, args.output_dir),
        plot_confusion_matrices(results, args.output_dir),
    ]:
        saved_paths.extend(paths)

    print("Generated files:")
    for path in saved_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
