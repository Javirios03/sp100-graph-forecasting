import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.utils import PROCESSED_DIR
from src.gnn.gat import GAT_LSTM
from config import GNN_MODEL, TRAINING


# ------------------------
# DATA LOADING
# ------------------------

def load_data():
    X = np.load(PROCESSED_DIR / "X_gnn.npy")
    y = np.load(PROCESSED_DIR / "y_gnn.npy")

    snapshot_index = pd.read_parquet(PROCESSED_DIR / "gnn_snapshots_index.parquet")

    edge_index_corr = torch.load(PROCESSED_DIR / "edge_index_corr.pt")
    edge_attr_corr = torch.load(PROCESSED_DIR / "edge_attr_corr.pt")

    edge_index_js = torch.load(PROCESSED_DIR / "edge_index_js.pt")
    edge_attr_js = torch.load(PROCESSED_DIR / "edge_attr_js.pt")

    return X, y, snapshot_index, edge_index_corr, edge_attr_corr, edge_index_js, edge_attr_js


# ------------------------
# SPLITS
# ------------------------

def get_splits(snapshot_index):
    train_idx = snapshot_index[snapshot_index["split"] == "train"].index.values
    val_idx   = snapshot_index[snapshot_index["split"] == "val"].index.values
    test_idx  = snapshot_index[snapshot_index["split"] == "test"].index.values
    return train_idx, val_idx, test_idx


# ------------------------
# TRAIN / EVAL
# ------------------------

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def remap_targets(y):
    y_remap = y.copy()
    y_remap[y == -1] = 0
    y_remap[y == 0] = 1
    y_remap[y == 1] = 2
    return y_remap


def validate_splits(train_idx, val_idx, test_idx):
    missing = []
    if len(train_idx) == 0:
        missing.append("train")
    if len(val_idx) == 0:
        missing.append("val")
    if len(test_idx) == 0:
        missing.append("test")
    if missing:
        raise ValueError(f"Empty split(s) found: {missing}")


def train_epoch(model, optimizer, criterion, X, y, indices, edge_index, edge_attr, device, grad_clip=None):
    model.train()

    total_loss = 0

    for t in np.random.permutation(indices):
        x_t = torch.tensor(X[t], dtype=torch.float32).to(device)
        y_t = torch.tensor(y[t], dtype=torch.long).to(device)

        optimizer.zero_grad()

        out = model(x_t, edge_index, edge_attr)
        loss = criterion(out, y_t)

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(indices)


CLASS_NAMES = ["down", "neutral", "up"]


def collect_predictions(model, X, y, indices, edge_index, edge_attr, device):
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for t in indices:
            x_t = torch.tensor(X[t], dtype=torch.float32).to(device)
            y_t = torch.tensor(y[t], dtype=torch.long).to(device)

            out = model(x_t, edge_index, edge_attr)
            preds = out.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_true.append(y_t.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    return all_true, all_preds


def compute_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )

    precision_macro = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )[0]
    recall_macro = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )[1]

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
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "labels": CLASS_NAMES,
    }


def evaluate(model, X, y, indices, edge_index, edge_attr, device):
    y_true, y_pred = collect_predictions(model, X, y, indices, edge_index, edge_attr, device)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

def run_experiment(graph_type, use_edge_attr, exp_name, training_overrides=None, model_overrides=None):
    '''
    Runs a full training and evaluation cycle for a given set of hyperparameters.

    Args:
        graph_type (str): Which graph to use ("js" or "corr")
        use_edge_attr (bool): Whether to use edge attributes or not.
        exp_name (str): Name of the experiment (used for logging and saving).
        training_overrides (dict): Optional dict to override default training hyperparameters.
        model_overrides (dict): Optional dict to override default model hyperparameters.

    Returns:
        dict: A dictionary containing the results of the experiment (best_val_f1, test_f1, etc.).
    '''
    training_cfg = TRAINING.copy()
    model_cfg = GNN_MODEL.copy()

    print(f"Running experiment: {exp_name}")
    print(f"Using the following configuration:")
    print(f"Graph type: {graph_type}")
    print(f"Use edge attributes: {use_edge_attr}")
    print(f"Training parameters: {training_cfg}")
    print(f"Model parameters: {model_cfg}")

    if training_overrides:
        training_cfg.update(training_overrides)
    if model_overrides:
        model_cfg.update(model_overrides)

    if training_cfg.get("patience") is None:
        training_cfg["patience"] = training_cfg["epochs"]

    set_seed(training_cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y, snapshot_index, ei_corr, ea_corr, ei_js, ea_js = load_data()

    y = remap_targets(y)

    train_idx, val_idx, test_idx = get_splits(snapshot_index)
    validate_splits(train_idx, val_idx, test_idx)

    # -------- GRAPH SELECTION --------
    if graph_type == "corr":
        edge_index, edge_attr = ei_corr, ea_corr
    elif graph_type == "js":
        edge_index, edge_attr = ei_js, ea_js
    else:
        raise ValueError("graph_type must be either 'corr' or 'js'")

    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)

    if not use_edge_attr:
        edge_attr = None

    # -------- MODEL --------
    model = GAT_LSTM(
        in_feats=X.shape[-1],
        lstm_hidden=model_cfg["lstm_hidden"],
        gat_hidden=model_cfg["gat_hidden"],
        heads=model_cfg["heads"],
        dropout=model_cfg["dropout"],
        lstm_layers=model_cfg["lstm_layers"],
        bidirectional=model_cfg["bidirectional"],
        edge_dim=edge_attr.shape[1] if edge_attr is not None else None,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["lr"],
        weight_decay=training_cfg["weight_decay"],
    )

    # -------- LOGGING --------
    log_dir = Path("runs") / exp_name
    writer = SummaryWriter(log_dir=log_dir)

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    checkpoint_path = model_dir / f"{exp_name}.pt"
    best_val = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    history = []

    # -------- TRAIN LOOP --------
    for epoch in range(training_cfg["epochs"]):

        loss = train_epoch(
            model,
            optimizer,
            criterion,
            X,
            y,
            train_idx,
            edge_index,
            edge_attr,
            device,
            grad_clip=training_cfg.get("grad_clip"),
        )
        val_f1 = evaluate(model, X, y, val_idx, edge_index, edge_attr, device)

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)

        print(f"[{exp_name}] Epoch {epoch} | Loss {loss:.4f} | Val F1 {val_f1:.4f}")

        history.append({
            "epoch": epoch,
            "loss": loss,
            "val_f1": val_f1
        })

        if val_f1 > best_val:
            best_val = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            if (
                training_cfg.get("early_stopping", True)
                and epochs_no_improve >= training_cfg["patience"]
            ):
                print(f"[{exp_name}] Early stopping at epoch {epoch}")
                break

    # -------- TEST --------
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    val_true, val_pred = collect_predictions(model, X, y, val_idx, edge_index, edge_attr, device)
    test_true, test_pred = collect_predictions(model, X, y, test_idx, edge_index, edge_attr, device)
    val_metrics = compute_metrics(val_true, val_pred)
    test_metrics = compute_metrics(test_true, test_pred)
    test_f1 = test_metrics["f1_macro"]

    writer.add_scalar("F1/test", test_f1, 0)
    writer.add_scalar("Accuracy/test", test_metrics["accuracy"], 0)
    writer.add_scalar("Precision/test_macro", test_metrics["precision_macro"], 0)
    writer.add_scalar("Recall/test_macro", test_metrics["recall_macro"], 0)

    writer.add_hparams(
        {
            "graph_type": graph_type,
            "use_edge_attr": use_edge_attr,
            "lr": training_cfg["lr"],
            "weight_decay": training_cfg["weight_decay"],
            "grad_clip": training_cfg.get("grad_clip", 0.0) or 0.0,
            "early_stopping": training_cfg.get("early_stopping", True),
            "patience": training_cfg["patience"],
            "epochs": training_cfg["epochs"],
            "lstm_hidden": model_cfg["lstm_hidden"],
            "gat_hidden": model_cfg["gat_hidden"],
            "lstm_layers": model_cfg["lstm_layers"],
            "bidirectional": model_cfg["bidirectional"],
        },
        {
            "hparam/test_f1": test_f1,
            "hparam/best_val_f1": best_val,
        }
    )

    writer.close()

    # -------- SAVE RESULTS --------
    results = {
        "experiment": exp_name,
        "graph_type": graph_type,
        "use_edge_attr": use_edge_attr,
        "best_val_f1": best_val,
        "best_epoch": best_epoch,
        "test_f1": test_f1,
        "history": history,
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
        },
    }

    Path("results").mkdir(exist_ok=True)

    with open(Path("results") / f"{exp_name}.json", "w") as f:
        json.dump(results, f, indent=4)

    return results

# def main():
#     writer = SummaryWriter(log_dir="runs/gnn_training")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     X, y, snapshot_index, ei_corr, ea_corr, ei_js, ea_js = load_data()
#     train_idx, val_idx, test_idx = get_splits(snapshot_index)

#     # Selección de grafo
#     if GRAPH_TYPE == "corr":
#         edge_index, edge_attr = ei_corr.to(device), ea_corr.to(device)
#     else:
#         edge_index, edge_attr = ei_js.to(device), ea_js.to(device)

#     # Modelo
#     model = GAT_LSTM(
#         in_feats=X.shape[-1],
#         lstm_hidden=GNN_MODEL["lstm_hidden"],
#         gat_hidden=GNN_MODEL["gat_hidden"],
#         heads=GNN_MODEL["heads"],
#         dropout=GNN_MODEL["dropout"],
#         edge_dim=edge_attr.shape[1] if USE_EDGE_ATTR else None,
#     ).to(device)

#     if not USE_EDGE_ATTR:
#         edge_attr = None

#     optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING["lr"], weight_decay=TRAINING["weight_decay"])

#     best_val = 0

#     for epoch in range(TRAINING["epochs"]):
#         loss = train_epoch(model, optimizer, X, y, train_idx, edge_index, edge_attr, device)
#         val_f1 = evaluate(model, X, y, val_idx, edge_index, edge_attr, device)

#         print(f"[{GRAPH_TYPE}] Epoch {epoch} | Loss {loss:.4f} | Val F1 {val_f1:.4f}")

#         writer.add_scalar(f"{GRAPH_TYPE}/loss", loss, epoch)
#         writer.add_scalar(f"{GRAPH_TYPE}/val_f1", val_f1, epoch)

#         if val_f1 > best_val:
#             best_val = val_f1
#             torch.save(model.state_dict(), f"best_model_{GRAPH_TYPE}.pt")

#     test_f1 = evaluate(model, X, y, test_idx, edge_index, edge_attr, device)
#     print(f"Test F1 ({GRAPH_TYPE}): {test_f1:.4f}")
#     writer.close()
