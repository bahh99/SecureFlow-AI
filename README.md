<div align="center">

# 🛡️ SecureFlow AI

### Privacy-Preserving Federated Fraud Detection for Mobile Money

**Real-time fraud detection trained collaboratively across mobile money operators — without ever sharing transaction data.**

[![Status](https://img.shields.io/badge/Status-Prototype-orange?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)](LICENSE)
[![Privacy](https://img.shields.io/badge/Privacy-By_Design-green?style=for-the-badge)]()
[![Dataset](https://img.shields.io/badge/Dataset-PaySim-purple?style=for-the-badge)]()

</div>

---

## What this does

SecureFlow AI lets multiple mobile money operators jointly train a fraud detection model — each on their own private transactions — without any operator ever seeing another's data.

The privacy guarantee is mathematical, not contractual: federated learning + differential privacy + SecAgg+ encrypted aggregation make the data impossible to extract from gradient updates.

The training and evaluation use **PaySim**, the standard public benchmark for mobile money fraud research.

---

## Quick start

```bash
# Install dependencies
pip install -e .

# Run a federated training simulation (3 operator nodes, 3 rounds)
flwr run . local-simulation --stream
```

That's it. The simulation will spin up 3 simulated operator nodes, train a fraud model collaboratively on PaySim data, apply differential privacy to all gradient updates, aggregate them via SecAgg+, and report fraud detection metrics at each round.

**No PaySim download required** — synthetic fallback data is generated automatically.
For real PaySim: see [data/README.md](data/README.md) (one-time Kaggle download).

---

## Project structure

```
secureflow-ai/
├── pyproject.toml            ← Flower config + dependencies
├── data/                     ← PaySim CSV (gitignored)
│   └── README.md
└── secureflow/
    └── core/
        ├── data.py           ← PaySim loading + federated partitioning
        ├── model.py          ← Tabular fraud detector (PyTorch)
        ├── client_app.py     ← Federated client (one per operator)
        ├── server_app.py     ← Federation server (SecAgg+)
        └── defence/
            └── noise.py      ← DP noise + gradient clipping
```

---

## Configuration

All settings live in `pyproject.toml` under `[tool.flwr.app.config]`:

| Setting | Default | What it does |
|---|---|---|
| `num-clients` | 3 | Operator nodes in the federation |
| `num-server-rounds` | 5 | Training rounds per simulation |
| `local-epochs` | 3 | Local training epochs per round |
| `dp-enabled` | true | Apply differential privacy to gradients |
| `dp-noise-std` | 0.05 | DP noise std (higher = more private) |
| `dp-clip-value` | 0.1 | Gradient clip threshold |
| `is-demo` | true | Small dataset (5K/operator) for fast demos |

To run with 5 operators on the full dataset:
```toml
num-clients = 5
is-demo = false
```
Then update `[tool.flwr.federations.local-simulation]` to match: `options.num-supernodes = 5`.

---

## Data: PaySim

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) is the gold-standard public mobile money fraud dataset. SecureFlow AI engineers 14 features from PaySim's transaction columns:

- Transaction amount and balance changes
- One-hot transaction types (TRANSFER, CASH_OUT, PAYMENT, DEBIT, CASH_IN)
- Hour-of-day from the simulation step
- Amount-to-balance ratio (a key fraud signal — fraudsters drain accounts)

Real PaySim has ~0.13% fraud; for demo visibility we upsample to 5%.

---

## Architecture

```
              ┌─────────────────────────────┐
              │   Federation Server         │
              │   SecAgg+ encrypted         │
              │   aggregation               │
              └────────────┬────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐        ┌────▼────┐        ┌───▼─────┐
   │Operator │        │Operator │        │Operator │
   │   A     │        │   B     │        │   C     │
   │         │        │         │        │         │
   │ Local   │        │ Local   │        │ Local   │
   │ training│        │ training│        │ training│
   │ + DP    │        │ + DP    │        │ + DP    │
   └─────────┘        └─────────┘        └─────────┘
   PaySim
   shard 1            shard 2             shard 3

   Raw data           Raw data            Raw data
   never leaves       never leaves        never leaves
```

---

## Privacy guarantee

Three layers:

1. **Federated learning** — operator data never moves. Each operator trains locally on its own transactions.

2. **Differential privacy** — every gradient update is clipped and noised before being shared. This means even if an attacker intercepts an update, they cannot recover individual transactions.

3. **SecAgg+ secure aggregation** — operator updates are cryptographically masked so the federation server only sees the aggregate, never individual contributions.

To verify privacy holds, toggle `dp-enabled = false` in `pyproject.toml` and observe that fraud detection still works but raw gradients become attackable.

---

## Pilot context

This codebase is the foundation for SecureFlow AI's planned 2026 pilot with mobile money operators in **Sierra Leone**. The federated architecture extends naturally across Orange Group's 17 African and Middle-Eastern markets without any cross-operator data sharing.

---

## Research foundation

The privacy-preserving core is grounded in peer-reviewed federated learning research and was developed through the founder's MSc thesis at ELTE University, Budapest:

📄 [github.com/bahh99/msc-thesis-federated-learning-privacy](https://github.com/bahh99/msc-thesis-federated-learning-privacy)

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Contact

**Founder:** Umaru Bah
📧 r11rbq@inf.elte.hu
🐙 [@bahh99](https://github.com/bahh99)
🌍 Hamburg, Germany
