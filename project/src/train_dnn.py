"""Train and evaluate a feed-forward DNN baseline (PyTorch), matching the
classical model comparison in train_classical.py."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from preprocess import load_and_split

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

RANDOM_STATE = 42
EPOCHS = 30
BATCH_SIZE = 512
LEARNING_RATE = 1e-3


class DiabetesNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_tensor_loader(X, y, batch_size, shuffle):
    X_t = torch.tensor(X.values, dtype=torch.float32)
    y_t = torch.tensor(y.values, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    torch.manual_seed(RANDOM_STATE)
    device = get_device()
    print(f"Using device: {device}")

    X_train, X_test, y_train, y_test, _ = load_and_split()

    train_loader = to_tensor_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32).to(device)

    model = DiabetesNet(X_train.shape[1]).to(device)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        epoch_loss /= len(train_loader.dataset)
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{EPOCHS} - loss: {epoch_loss:.4f}")

    elapsed = time.time() - start

    model.eval()
    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    scores = {
        "DNN (Feed-forward)": {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "auc": roc_auc_score(y_test, probs),
            "train_time_s": elapsed,
        }
    }
    results_df = pd.DataFrame(scores).T[["accuracy", "f1", "auc", "train_time_s"]]
    results_df.to_csv(RESULTS_DIR / "dnn_results.csv")

    torch.save(model.state_dict(), MODELS_DIR / "dnn.pt")

    print(f"\nTrained in {elapsed:.1f}s")
    print(results_df.round(4).to_string())
    print(f"\nSaved results to {RESULTS_DIR}, model to {MODELS_DIR}")


if __name__ == "__main__":
    main()
