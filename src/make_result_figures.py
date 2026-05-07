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
        model_labels = {
            "random_forest": "Random Forest",
            "lstm": "LSTM",
        }
        return model_labels.get(str(model_type), str(model_type).replace("_", " ").title())

    graph_type = result.get("graph_type")
    if graph_type:
        graph_label = "Correlation" if graph_type == "corr" else graph_type.upper()
        return f"GNN ({graph_label})"

    name = result.get("experiment", fallback_name)
    return name.replace("_", " ").title()


def model_order_key(model_name):
    order = {
        "Random Forest": 0,
        "LSTM": 1,
        "GNN (Correlation)": 2,
        "GNN (JS)": 3,
    }
    return order.get(model_name, 99)


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


def save_figure(fig, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
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
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("model", key=lambda s: s.map(model_order_key)).reset_index(drop=True)
    return df


def build_test_metrics_frame(results):
    rows = []
    for result in results:
        test_metrics = result.get("metrics", {}).get("test", {})
        if not test_metrics:
            continue
        rows.append(
            {
                "model": result["model"],
                "experiment": result["experiment"],
                "accuracy": test_metrics.get("accuracy"),
                "precision_macro": test_metrics.get("precision_macro"),
                "recall_macro": test_metrics.get("recall_macro"),
                "f1_macro": test_metrics.get("f1_macro"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("model", key=lambda s: s.map(model_order_key)).reset_index(drop=True)
    return df


def build_per_class_metrics_frame(results):
    rows = []
    for result in results:
        test_metrics = result.get("metrics", {}).get("test", {})
        for class_name, values in test_metrics.get("per_class", {}).items():
            rows.append(
                {
                    "model": result["model"],
                    "experiment": result["experiment"],
                    "class": class_name,
                    "precision": values.get("precision"),
                    "recall": values.get("recall"),
                    "f1": values.get("f1"),
                    "support": values.get("support"),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("model", key=lambda s: s.map(model_order_key)).reset_index(drop=True)
    return df


def save_latex_table(df, output_path, caption, label):
    pretty = df.copy()
    pretty.columns = [
        col.replace("_", " ").title().replace("F1", "F1")
        for col in pretty.columns
    ]
    latex = pretty.to_latex(
        index=False,
        float_format="%.3f",
        caption=caption,
        label=label,
        escape=True,
    )
    output_path.write_text(latex, encoding="utf-8")


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


def plot_confusion_matrices_grid(results, output_dir):
    results_with_cm = [
        result
        for result in results
        if result.get("metrics", {}).get("test", {}).get("confusion_matrix")
    ]
    by_model = {result["model"]: result for result in results_with_cm}
    ordered_models = ["Random Forest", "LSTM", "GNN (Correlation)", "GNN (JS)"]
    ordered_results = [by_model[model] for model in ordered_models if model in by_model]
    if len(ordered_results) < 2:
        return []

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.2))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")

    for ax, result in zip(axes, ordered_results):
        test_metrics = result["metrics"]["test"]
        labels = test_metrics.get("labels", ["down", "neutral", "up"])
        matrix = pd.DataFrame(
            test_metrics["confusion_matrix"],
            index=[label.title() for label in labels],
            columns=[label.title() for label in labels],
        )
        ax.axis("on")
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(result["model"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle("Test confusion matrices", fontsize=14, y=1.02)
    return save_figure(fig, output_dir, "confusion_matrices_2x2")


def plot_training_curves_row(history_df, output_dir):
    if history_df.empty:
        return []

    df = history_df.copy()
    df = df.sort_values("model", key=lambda s: s.map(model_order_key))
    has_loss = df["train_loss"].notna().any()
    has_val = df["val_f1"].notna().any()
    if not has_loss and not has_val:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    if has_loss:
        sns.lineplot(
            data=df.dropna(subset=["train_loss"]),
            x="epoch",
            y="train_loss",
            hue="model",
            linewidth=2.0,
            ax=axes[0],
        )
        axes[0].set_title("Training loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.25)
    else:
        axes[0].axis("off")

    if has_val:
        val_df = df.dropna(subset=["val_f1"])
        sns.lineplot(
            data=val_df,
            x="epoch",
            y="val_f1",
            hue="model",
            linewidth=2.0,
            ax=axes[1],
        )
        best_rows = val_df.loc[val_df.groupby("model")["val_f1"].idxmax()]
        sns.scatterplot(
            data=best_rows,
            x="epoch",
            y="val_f1",
            hue="model",
            legend=False,
            s=70,
            edgecolor="black",
            linewidth=0.6,
            ax=axes[1],
        )
        axes[1].set_title("Validation macro-F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1")
        axes[1].set_ylim(bottom=0)
        axes[1].grid(alpha=0.25)
    else:
        axes[1].axis("off")

    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title("")

    return save_figure(fig, output_dir, "training_curves_row")


def plot_model_comparison_row(results, summary_df, output_dir):
    if summary_df.empty:
        return []

    per_class_rows = []
    for result in results:
        test_metrics = result.get("metrics", {}).get("test", {})
        for class_name, values in test_metrics.get("per_class", {}).items():
            per_class_rows.append(
                {
                    "model": result["model"],
                    "class": class_name.title(),
                    "f1": values.get("f1"),
                }
            )

    score_df = summary_df.melt(
        id_vars=["model", "experiment"],
        value_vars=["best_val_f1", "test_f1"],
        var_name="metric",
        value_name="macro_f1",
    ).dropna(subset=["macro_f1"])
    if score_df.empty:
        return []

    score_df["metric"] = score_df["metric"].map(
        {"best_val_f1": "Best validation F1", "test_f1": "Test F1"}
    )
    score_df = score_df.sort_values("model", key=lambda s: s.map(model_order_key))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = sns.barplot(data=score_df, x="model", y="macro_f1", hue="metric", ax=axes[0])
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    axes[0].set_title("Overall macro-F1")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Macro-F1")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].set_ylim(0, max(0.45, score_df["macro_f1"].max() * 1.23))
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(title="")

    per_class_df = pd.DataFrame(per_class_rows).dropna() if per_class_rows else pd.DataFrame()
    if not per_class_df.empty:
        per_class_df = per_class_df.sort_values("model", key=lambda s: s.map(model_order_key))
        ax = sns.barplot(data=per_class_df, x="class", y="f1", hue="model", ax=axes[1])
        axes[1].set_title("Test F1 by class")
        axes[1].set_xlabel("Class")
        axes[1].set_ylabel("F1")
        axes[1].set_ylim(0, max(0.45, per_class_df["f1"].max() * 1.23))
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend(title="", fontsize=8)
    else:
        axes[1].axis("off")

    return save_figure(fig, output_dir, "model_comparison_row")


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
    test_metrics_df = build_test_metrics_frame(results)
    per_class_metrics_df = build_per_class_metrics_frame(results)

    summary_path = args.output_dir / "model_results_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    saved_paths = [summary_path]
    if not test_metrics_df.empty:
        test_metrics_path = args.output_dir / "test_metrics_summary.csv"
        test_metrics_latex_path = args.output_dir / "test_metrics_summary.tex"
        test_metrics_df.to_csv(test_metrics_path, index=False)
        save_latex_table(
            test_metrics_df.drop(columns=["experiment"]),
            test_metrics_latex_path,
            "Test metrics for all models.",
            "tab:test_metrics",
        )
        saved_paths.extend([test_metrics_path, test_metrics_latex_path])

    if not per_class_metrics_df.empty:
        per_class_metrics_path = args.output_dir / "test_per_class_metrics.csv"
        per_class_metrics_latex_path = args.output_dir / "test_per_class_metrics.tex"
        per_class_metrics_df.to_csv(per_class_metrics_path, index=False)
        save_latex_table(
            per_class_metrics_df.drop(columns=["experiment"]),
            per_class_metrics_latex_path,
            "Per-class test metrics for all models.",
            "tab:test_per_class_metrics",
        )
        saved_paths.extend([per_class_metrics_path, per_class_metrics_latex_path])

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
        plot_confusion_matrices_grid(results, args.output_dir),
        plot_training_curves_row(history_df, args.output_dir),
        plot_model_comparison_row(results, summary_df, args.output_dir),
    ]:
        saved_paths.extend(paths)

    print("Generated files:")
    for path in saved_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
