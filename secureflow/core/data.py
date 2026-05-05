"""secureflow.core.data — PaySim dataset loading and federated partitioning.

PaySim is a synthetic mobile money dataset released by Lopez-Rojas et al.
that simulates real African mobile money transactions, including fraud
patterns. It is the gold standard for mobile money fraud detection research.

Dataset: https://www.kaggle.com/datasets/ealaxi/paysim1

Original columns:
    step          — Hour of simulation (0..743, ~31 days)
    type          — Transaction type (PAYMENT, TRANSFER, CASH_OUT, ...)
    amount        — Transaction amount
    nameOrig      — Sender ID
    oldbalanceOrg — Sender balance before transaction
    newbalanceOrig— Sender balance after transaction
    nameDest      — Receiver ID
    oldbalanceDest— Receiver balance before transaction
    newbalanceDest— Receiver balance after transaction
    isFraud       — 1 if transaction is fraud, 0 otherwise
    isFlaggedFraud— Rule-based flag (transfers > 200K)

We engineer a compact set of features focused on signals SecureFlow AI
would actually use in production deployment with an operator's API.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


# Engineered feature columns — these are the inputs to the fraud model.
# In production these come from the operator's transaction stream API;
# for the demo they are derived from PaySim columns.
FEATURE_COLUMNS = [
    "amount",
    "type_TRANSFER",     # one-hot: TRANSFER (highest fraud rate)
    "type_CASH_OUT",     # one-hot: CASH_OUT (second-highest fraud rate)
    "type_PAYMENT",      # one-hot: PAYMENT
    "type_DEBIT",        # one-hot: DEBIT
    "type_CASH_IN",      # one-hot: CASH_IN
    "old_balance_orig",
    "new_balance_orig",
    "balance_change_orig",   # newbalanceOrig - oldbalanceOrg
    "old_balance_dest",
    "new_balance_dest",
    "balance_change_dest",   # newbalanceDest - oldbalanceDest
    "hour_of_day",           # step % 24
    "amount_to_balance_ratio",
]

LABEL_COLUMN = "isFraud"

PAYSIM_TRANSACTION_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"]


# ── Path resolution ─────────────────────────────────────────────────────────

def find_paysim_csv() -> Path | None:
    """Locate PaySim CSV file by checking common paths.

    Returns Path if found, None otherwise.
    Users place the file at one of these locations:
      - data/paysim.csv (recommended)
      - data/PS_20174392719_1491204439457_log.csv (original Kaggle filename)
      - PAYSIM_PATH environment variable
    """
    env_path = os.environ.get("PAYSIM_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    candidates = [
        Path("data/paysim.csv"),
        Path("data/PS_20174392719_1491204439457_log.csv"),
        Path("PS_20174392719_1491204439457_log.csv"),
        Path("paysim.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ── Synthetic fallback (so demos work without a Kaggle download) ────────────

def generate_synthetic_paysim(n_rows: int = 50_000, fraud_rate: float = 0.013,
                              seed: int = 42) -> pd.DataFrame:
    """Generate PaySim-shaped synthetic data with realistic fraud patterns.

    Used when the real PaySim CSV is not available. Mirrors PaySim's column
    structure so all downstream code is identical. Fraud rate matches PaySim's
    real ~0.13% baseline (with some boost for visible demo signals).
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    rows = []

    # Legitimate transactions
    for _ in range(n_legit):
        tx_type = rng.choice(PAYSIM_TRANSACTION_TYPES,
                             p=[0.08, 0.35, 0.34, 0.01, 0.22])
        amount = float(rng.lognormal(np.log(150), 1.2))
        old_bal = max(0, float(rng.lognormal(np.log(2000), 1.5)))
        new_bal_orig = max(0, old_bal - amount) if tx_type in ("TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT") else old_bal + amount
        old_bal_dest = max(0, float(rng.lognormal(np.log(1500), 1.5)))
        new_bal_dest = old_bal_dest + amount if tx_type in ("TRANSFER", "PAYMENT") else old_bal_dest
        rows.append({
            "step": int(rng.integers(0, 744)),
            "type": tx_type,
            "amount": amount,
            "oldbalanceOrg": old_bal,
            "newbalanceOrig": new_bal_orig,
            "oldbalanceDest": old_bal_dest,
            "newbalanceDest": new_bal_dest,
            "isFraud": 0,
        })

    # Fraud transactions (always TRANSFER or CASH_OUT in real PaySim)
    for _ in range(n_fraud):
        tx_type = rng.choice(["TRANSFER", "CASH_OUT"], p=[0.5, 0.5])
        old_bal = max(100, float(rng.lognormal(np.log(8000), 1.0)))
        amount = old_bal * rng.uniform(0.85, 1.0)  # fraud drains account
        new_bal_orig = max(0, old_bal - amount)
        old_bal_dest = float(rng.lognormal(np.log(500), 1.5))
        new_bal_dest = 0.0  # destinations often zero out (mule accounts)
        rows.append({
            "step": int(rng.integers(0, 744)),
            "type": tx_type,
            "amount": amount,
            "oldbalanceOrg": old_bal,
            "newbalanceOrig": new_bal_orig,
            "oldbalanceDest": old_bal_dest,
            "newbalanceDest": new_bal_dest,
            "isFraud": 1,
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


# ── Feature engineering ─────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw PaySim columns into the model's feature columns.

    Produces a DataFrame with exactly FEATURE_COLUMNS + LABEL_COLUMN.
    """
    out = pd.DataFrame()

    # Direct features
    out["amount"] = df["amount"].astype(float)
    out["old_balance_orig"] = df["oldbalanceOrg"].astype(float)
    out["new_balance_orig"] = df["newbalanceOrig"].astype(float)
    out["old_balance_dest"] = df["oldbalanceDest"].astype(float)
    out["new_balance_dest"] = df["newbalanceDest"].astype(float)

    # Engineered features
    out["balance_change_orig"] = out["new_balance_orig"] - out["old_balance_orig"]
    out["balance_change_dest"] = out["new_balance_dest"] - out["old_balance_dest"]
    out["hour_of_day"] = (df["step"] % 24).astype(float)
    out["amount_to_balance_ratio"] = out["amount"] / (out["old_balance_orig"] + 1.0)

    # One-hot transaction type
    for t in PAYSIM_TRANSACTION_TYPES:
        out[f"type_{t}"] = (df["type"] == t).astype(float)

    # Reorder to match FEATURE_COLUMNS exactly
    out = out[FEATURE_COLUMNS]
    out[LABEL_COLUMN] = df["isFraud"].astype(int)
    return out


def normalize_features(df: pd.DataFrame, fit_on: pd.DataFrame | None = None) -> pd.DataFrame:
    """Z-score normalize the feature columns. Pass `fit_on` to use stats from
    a different DataFrame (e.g. fit on train, apply to test)."""
    source = fit_on if fit_on is not None else df
    out = df.copy()
    for col in FEATURE_COLUMNS:
        mean = source[col].mean()
        std = source[col].std()
        if std > 1e-8:
            out[col] = (df[col] - mean) / std
        else:
            out[col] = 0.0
    return out


# ── Federated partitioning ──────────────────────────────────────────────────

def load_paysim(max_rows: int | None = None, fraud_rate_target: float = 0.05,
                seed: int = 42) -> pd.DataFrame:
    """Load PaySim from disk if available, else generate synthetic data.

    Args:
        max_rows: cap dataset size (None = full). For demos, 50K–100K is plenty.
        fraud_rate_target: if loading real PaySim (~0.13% fraud), upsample fraud
            cases to this rate for visible demo signals. Set None to disable.
        seed: reproducibility seed.

    Returns:
        DataFrame with raw PaySim columns: step, type, amount, oldbalanceOrg,
        newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud.
    """
    csv_path = find_paysim_csv()

    if csv_path is None:
        print("[SecureFlow] PaySim CSV not found — generating synthetic data.")
        print("              To use real PaySim: download from")
        print("              https://www.kaggle.com/datasets/ealaxi/paysim1")
        print("              and place at data/paysim.csv")
        return generate_synthetic_paysim(
            n_rows=max_rows or 50_000,
            fraud_rate=fraud_rate_target or 0.013,
            seed=seed,
        )

    print(f"[SecureFlow] Loading PaySim from {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
             "oldbalanceDest", "newbalanceDest", "isFraud"]]

    if max_rows is not None and len(df) > max_rows:
        # Keep all fraud + sample legit
        fraud_df = df[df["isFraud"] == 1]
        legit_df = df[df["isFraud"] == 0].sample(
            n=min(max_rows - len(fraud_df), len(df) - len(fraud_df)),
            random_state=seed,
        )
        df = pd.concat([fraud_df, legit_df]).sample(frac=1.0, random_state=seed)

    if fraud_rate_target is not None:
        actual_rate = df["isFraud"].mean()
        if actual_rate < fraud_rate_target:
            fraud_df = df[df["isFraud"] == 1]
            legit_df = df[df["isFraud"] == 0]
            target_legit = int(len(fraud_df) * (1 - fraud_rate_target) / fraud_rate_target)
            legit_df = legit_df.sample(n=min(target_legit, len(legit_df)),
                                        random_state=seed)
            df = pd.concat([fraud_df, legit_df]).sample(frac=1.0, random_state=seed)

    return df.reset_index(drop=True)


def partition_for_federation(df: pd.DataFrame, num_partitions: int,
                              seed: int = 42) -> list[pd.DataFrame]:
    """Split a PaySim DataFrame into `num_partitions` operator-shaped shards.

    Stratifies by fraud label so each operator sees similar fraud rates —
    matches the realistic case where every operator has fraud problems.
    """
    rng = np.random.default_rng(seed)

    fraud_idx = df.index[df["isFraud"] == 1].to_numpy().copy()
    legit_idx = df.index[df["isFraud"] == 0].to_numpy().copy()
    rng.shuffle(fraud_idx)
    rng.shuffle(legit_idx)

    fraud_chunks = np.array_split(fraud_idx, num_partitions)
    legit_chunks = np.array_split(legit_idx, num_partitions)

    partitions = []
    for i in range(num_partitions):
        idx = np.concatenate([fraud_chunks[i], legit_chunks[i]])
        rng.shuffle(idx)
        partitions.append(df.loc[idx].reset_index(drop=True))
    return partitions


# ── Federated client interface (the one client_app.py uses) ─────────────────

def load_partition(partition_id: int, num_partitions: int, batch_size: int,
                    is_demo: bool = False, seed: int = 42
                    ) -> tuple[DataLoader, DataLoader]:
    """Load this client's slice of PaySim as PyTorch DataLoaders.

    Used by FlowerClient.client_fn() to give each federated node its own
    private partition of the dataset. In production this would be replaced
    by a connector to the operator's transaction API.

    Args:
        partition_id: which client (0..num_partitions-1)
        num_partitions: total number of clients in the federation
        batch_size: training batch size
        is_demo: if True, use a small (fast) dataset
        seed: reproducibility

    Returns:
        (train_loader, test_loader)
    """
    max_rows = 5_000 if is_demo else 50_000

    df = load_paysim(max_rows=max_rows, seed=seed)
    partitions = partition_for_federation(df, num_partitions, seed=seed)
    my_partition = partitions[partition_id]

    feats = engineer_features(my_partition)

    # 80/20 train/test split, normalize using train statistics
    n_train = int(len(feats) * 0.8)
    train_df = feats.iloc[:n_train]
    test_df = feats.iloc[n_train:]

    train_norm = normalize_features(train_df, fit_on=train_df)
    test_norm = normalize_features(test_df, fit_on=train_df)

    train_X = torch.tensor(train_norm[FEATURE_COLUMNS].values, dtype=torch.float32)
    train_y = torch.tensor(train_norm[LABEL_COLUMN].values, dtype=torch.long)
    test_X = torch.tensor(test_norm[FEATURE_COLUMNS].values, dtype=torch.float32)
    test_y = torch.tensor(test_norm[LABEL_COLUMN].values, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(train_X, train_y),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_X, test_y),
                             batch_size=batch_size)
    return train_loader, test_loader


if __name__ == "__main__":
    # Sanity check
    df = load_paysim(max_rows=10_000)
    print(f"Loaded {len(df):,} transactions, fraud rate: {df['isFraud'].mean():.2%}")
    parts = partition_for_federation(df, 3)
    for i, p in enumerate(parts):
        print(f"  Operator {i}: {len(p):,} transactions, "
              f"fraud rate: {p['isFraud'].mean():.2%}")
    feats = engineer_features(parts[0])
    print(f"\nEngineered features ({len(FEATURE_COLUMNS)} columns):")
    print(feats.head(3))
