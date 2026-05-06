import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import PROCESSED_DIR, MODELS_DIR


CLASS_NAMES = ["down", "neutral", "up"]
CLASS_VALUES = [0, 1, 2]


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

print("Cargando dataset temporal...")
meta = pd.read_parquet(PROCESSED_DIR / "temporal_index.parquet")
X = np.load(PROCESSED_DIR / "X_temporal.npy")
y = np.load(PROCESSED_DIR / "y_temporal.npy")

print(f"Dataset shape: X={X.shape}, y={y.shape}")

# Map Targets
y_mapped = np.where(y == -1, 0, y+1)
print("Targets mapeados:", np.unique(y_mapped, return_counts=True))

class StockLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=True,  # ← FIX #1
                           dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size*2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        hn = torch.mean(out, dim=1)
        out = self.relu(self.fc1(hn))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        return self.fc3(out)

# Datos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_mask = meta["split"] == "train"
val_mask = meta["split"] == "val"
test_mask = meta["split"] == "test"

X_train, y_train = X[train_mask], y_mapped[train_mask]
X_val, y_val = X[val_mask], y_mapped[val_mask]
X_test, y_test = X[test_mask], y_mapped[test_mask]

# Tensores + VALIDATION LOADER (FIX #5)
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)
y_val_t = torch.LongTensor(y_val).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.LongTensor(y_test).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)  # ← NUEVO!
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

# Entrenamiento
model = StockLSTM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
class_weights = torch.FloatTensor([1.25, 1.6, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

print("\nEntrenando LSTM (30 epochs)...")
best_f1, epochs_no_improve = 0, 0
best_epoch = -1
patience = 7
history = []

for epoch in range(30):
    # Train
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    # Val (FIX #2, #3)
    model.eval()
    val_loss = 0
    val_preds, val_true = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_true.extend(batch_y.cpu().numpy())
    
    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    train_loss = total_loss / len(train_loader)
    val_f1 = classification_report(
        val_true,
        val_preds,
        output_dict=True,
        zero_division=0,
    )["macro avg"]["f1-score"]
    history.append({
        "epoch": epoch,
        "loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_f1": float(val_f1),
    })
    
    # F1 cada 5 epochs (FIX #3)
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODELS_DIR / "lstm_temporal.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# sFinal Evaluation - Batch to allow GPU inference
print("\nCargando mejor modelo...")
model.load_state_dict(torch.load(MODELS_DIR / "lstm_temporal.pth"))

# Liberar memoria
torch.cuda.empty_cache()

model.eval()

# Eval TRAIN por batches (FIX OOM)
train_preds, train_true = [], []
with torch.no_grad():
    for i in range(0, len(X_train_t), 1024):  # Batches de 1024
        batch = X_train_t[i:i+1024]
        pred = torch.argmax(model(batch), dim=1).cpu().numpy()
        train_preds.extend(pred)
        train_true.extend(y_train_t[i:i+1024].cpu().numpy())

# Eval VAL por batches
val_preds, val_true = [], []
with torch.no_grad():
    for i in range(0, len(X_val_t), 1024):
        batch = X_val_t[i:i+1024]
        pred = torch.argmax(model(batch), dim=1).cpu().numpy()
        val_preds.extend(pred)
        val_true.extend(y_val_t[i:i+1024].cpu().numpy())

# Eval TEST por batches
test_preds, test_true = [], []
with torch.no_grad():
    for i in range(0, len(X_test_t), 1024):
        batch = X_test_t[i:i+1024]
        pred = torch.argmax(model(batch), dim=1).cpu().numpy()
        test_preds.extend(pred)
        test_true.extend(y_test_t[i:i+1024].cpu().numpy())

# Map back
def map_back(preds):
    return np.where(np.array(preds) == 0, -1, np.array(preds)-1)

print("\n" + "="*60)
print("LSTM TEMPORAL RESULTS (FINAL)")
print("="*60)

print("\nF1-macro VAL:")
print(classification_report(np.array(val_true), val_preds, 
                          target_names=["Down (0)", "Neutral (1)", "Up (2)"], 
                          zero_division=0))

cm_val = confusion_matrix(np.array(val_true), val_preds)
print("\nConfusion Matrix VAL:")
print(cm_val)

print("\nF1-macro TEST:")
print(classification_report(np.array(test_true), test_preds,
                          target_names=["Down (0)", "Neutral (1)", "Up (2)"],
                          zero_division=0))

cm_test = confusion_matrix(np.array(test_true), test_preds)
print("\nConfusion Matrix TEST:")
print(cm_test)

# Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Down", "Neutral", "Up"],
            yticklabels=["Down", "Neutral", "Up"])
plt.title(f"LSTM Confusion Matrix (Val) F1={best_f1:.3f}")
plt.ylabel("Real")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

print(f"\n✅ LSTM F1-macro FINAL: {best_f1:.4f} (vs RF 0.34)")
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

results = {
    "experiment": "lstm_temporal",
    "model_type": "lstm",
    "best_val_f1": float(best_f1),
    "best_epoch": int(best_epoch),
    "test_f1": float(f1_score(test_true, test_preds, average="macro", zero_division=0)),
    "history": history,
    "metrics": {
        "train": compute_metrics(train_true, train_preds),
        "val": compute_metrics(val_true, val_preds),
        "test": compute_metrics(test_true, test_preds),
    },
}

Path("results").mkdir(exist_ok=True)
with open(Path("results") / "lstm_temporal.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)
