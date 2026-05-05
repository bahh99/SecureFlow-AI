"""secureflow.core.client_app — Federated learning client.

Each mobile money operator runs this client locally. It:
  1. Receives the global model from the federation server
  2. Trains the model on the operator's own transaction data
  3. Applies differential privacy to gradients
  4. Sends only the DP-protected, encrypted update to the server

Raw transaction data NEVER leaves the operator's infrastructure.
"""

from __future__ import annotations

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from secureflow.core.data import load_partition
from secureflow.core.model import (
    FraudDetector, get_weights, set_weights, train, test,
)
from secureflow.core.defence import apply_differential_privacy


class SecureFlowClient(NumPyClient):
    """Mobile money operator node — trains locally, shares only DP-protected updates."""

    def __init__(self, train_loader, test_loader,
                 local_epochs: int, learning_rate: float,
                 dp_enabled: bool, dp_noise_std: float, dp_clip_value: float):
        self.model = FraudDetector()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dp_enabled = dp_enabled
        self.dp_noise_std = dp_noise_std
        self.dp_clip_value = dp_clip_value

    def fit(self, parameters, config):
        """Train locally and return DP-protected model update."""
        set_weights(self.model, parameters)

        results = train(
            self.model, self.train_loader, self.test_loader,
            epochs=self.local_epochs,
            learning_rate=self.learning_rate,
            device=self.device,
        )

        weights = get_weights(self.model)

        # Apply differential privacy before sharing
        if self.dp_enabled:
            weights = apply_differential_privacy(
                weights,
                noise_std=self.dp_noise_std,
                clip_value=self.dp_clip_value,
            )
            print("[Defense] Differential privacy applied "
                  f"(noise_std={self.dp_noise_std}, clip={self.dp_clip_value})")

        return weights, len(self.train_loader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the global model on this operator's local test data."""
        set_weights(self.model, parameters)
        loss, metrics = test(self.model, self.test_loader, self.device)
        return loss, len(self.test_loader.dataset), metrics


def client_fn(context: Context):
    """Construct a SecureFlowClient from Flower's run context."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    cfg = context.run_config
    is_demo = cfg.get("is-demo", True)

    train_loader, test_loader = load_partition(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=cfg["batch-size"],
        is_demo=is_demo,
    )

    return SecureFlowClient(
        train_loader=train_loader,
        test_loader=test_loader,
        local_epochs=cfg["local-epochs"],
        learning_rate=cfg["learning-rate"],
        dp_enabled=cfg.get("dp-enabled", True),
        dp_noise_std=cfg.get("dp-noise-std", 0.05),
        dp_clip_value=cfg.get("dp-clip-value", 0.1),
    ).to_client()


# Flower ClientApp with SecAgg+ encrypted aggregation enabled
app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod],
)
