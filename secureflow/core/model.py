"""secureflow.core.model — Tabular fraud detection neural network.

A lightweight feed-forward network sized for tabular mobile money
transaction features. Designed to run on operator infrastructure
without requiring GPUs.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from secureflow.core.data import FEATURE_COLUMNS


class FraudDetector(nn.Module):
    """Feed-forward fraud detector for mobile money transactions.

    Input:  14 transaction features (see FEATURE_COLUMNS in data.py)
    Output: 2-class logits (legitimate / fraud)

    Lightweight enough to train and serve on any operator's existing
    infrastructure. Inference latency typically < 50ms per transaction.
    """

    def __init__(self, input_dim: int = len(FEATURE_COLUMNS),
                 hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return fraud probability (0..1) for each transaction."""
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)[:, 1]

    def risk_score(self, x: torch.Tensor) -> torch.Tensor:
        """Return integer risk score 0–100 for each transaction."""
        return (self.predict_proba(x) * 100).int()


def make_model(seed: int = 42) -> FraudDetector:
    """Create a fresh fraud detector with a fixed random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    return FraudDetector()


def get_weights(model: nn.Module) -> list[np.ndarray]:
    """Extract model weights as NumPy arrays (Flower's exchange format)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Load NumPy weight arrays back into a model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train(model: FraudDetector, train_loader, val_loader, epochs: int,
          learning_rate: float, device: torch.device) -> dict:
    """Train the fraud detector on local transaction data.

    Uses a class-weighted loss to handle the heavy class imbalance
    inherent to fraud (typical 0.13–5% fraud rate in PaySim).
    """
    model.to(device)

    # Estimate class weights from this batch loader
    fraud_count = 0
    legit_count = 0
    for _, y in train_loader:
        fraud_count += int((y == 1).sum())
        legit_count += int((y == 0).sum())
    total = fraud_count + legit_count
    if fraud_count == 0:
        class_weights = torch.tensor([1.0, 1.0])
    else:
        class_weights = torch.tensor([
            total / (2 * legit_count) if legit_count > 0 else 1.0,
            total / (2 * fraud_count),
        ], dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    val_loss, val_metrics = test(model, val_loader, device)
    return {"val_loss": val_loss, **val_metrics}


def test(model: FraudDetector, loader, device: torch.device
         ) -> tuple[float, dict]:
    """Evaluate the fraud detector. Returns (loss, metrics).

    Metrics: precision, recall, fpr, accuracy.
    """
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            preds = logits.argmax(dim=1)

            tp += int(((preds == 1) & (y == 1)).sum())
            fp += int(((preds == 1) & (y == 0)).sum())
            fn += int(((preds == 0) & (y == 1)).sum())
            tn += int(((preds == 0) & (y == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, {
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "accuracy": accuracy,
    }
