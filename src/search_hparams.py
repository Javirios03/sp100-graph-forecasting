from itertools import product
from src.train_gnn import run_experiment
from pathlib import Path
import json

GRAPH_TYPES = ["js", "corr"]
LR_VALUES = [5e-4, 8e-4, 1e-3]
WD_VALUES = [0, 1e-4]
EPOCHS_VALUES = [80]
LSTM_HIDDEN = [64, 128]
GAT_HIDDEN = [64, 128]

results_dir = Path("results")
existing_results = {p.stem for p in results_dir.glob("*.json")}

all_experiments = []
new_results = []
for graph_type, lr, wd, epochs, lstm_h, gat_h in product(
    GRAPH_TYPES, LR_VALUES, WD_VALUES, EPOCHS_VALUES, LSTM_HIDDEN, GAT_HIDDEN
):
    exp_name = f"{graph_type}_edge_True_lr{lr}_wd{wd}_ep{epochs}_l{lstm_h}_g{gat_h}"
    all_experiments.append(exp_name)
    if exp_name in existing_results:
        print(f"Skipping {exp_name} (already exists)")
        continue
    training_overrides = {
        "lr": lr,
        "weight_decay": wd,
        "epochs": epochs,
    }
    model_overrides = {
        "lstm_hidden": lstm_h,
        "gat_hidden": gat_h,
    }

    result = run_experiment(
        graph_type=graph_type,
        use_edge_attr=True,
        exp_name=exp_name,
        training_overrides=training_overrides,
        model_overrides=model_overrides,
    )
    new_results.append(result)

print("\nSUMMARY (sorted by best_val_f1):")
all_results = []
for json_path in results_dir.glob("*.json"):
    with open(json_path, "r") as f:
        all_results.append(json.load(f))

for r in sorted(all_results, key=lambda x: x["best_val_f1"], reverse=True):
    print(r)
