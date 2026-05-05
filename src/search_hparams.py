from itertools import product
from src.train_gnn import run_experiment

LR_VALUES = [1e-4, 5e-4, 1e-3]
WD_VALUES = [0, 1e-4, 1e-3]
EPOCHS_VALUES = [20, 40]
LSTM_HIDDEN = [32, 64]
GAT_HIDDEN = [32, 64]

experiments = []
for lr, wd, epochs, lstm_h, gat_h in product(
    LR_VALUES, WD_VALUES, EPOCHS_VALUES, LSTM_HIDDEN, GAT_HIDDEN
):
    training_overrides = {
        "lr": lr,
        "weight_decay": wd,
        "epochs": epochs,
    }
    model_overrides = {
        "lstm_hidden": lstm_h,
        "gat_hidden": gat_h,
    }
    exp_name = f"js_edge_True_lr{lr}_wd{wd}_ep{epochs}_l{lstm_h}_g{gat_h}"

    result = run_experiment(
        graph_type="js",
        use_edge_attr=True,
        exp_name=exp_name,
        training_overrides=training_overrides,
        model_overrides=model_overrides,
    )
    experiments.append(result)

print("\nSUMMARY (sorted by best_val_f1):")
for r in sorted(experiments, key=lambda x: x["best_val_f1"], reverse=True):
    print(r)