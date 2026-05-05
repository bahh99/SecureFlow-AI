<div align="center">

# 🛡️ SecureFlow AI

### Privacy-Preserving Federated Fraud Detection for Mobile Money

**Real-time fraud detection trained collaboratively across mobile money operators — without ever sharing transaction data.**

[![Status](https://img.shields.io/badge/Status-Prototype-orange?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=for-the-badge)](LICENSE)
[![Privacy](https://img.shields.io/badge/Privacy-By_Design-green?style=for-the-badge)]()
[![Dataset](https://img.shields.io/badge/Dataset-PaySim-purple?style=for-the-badge)]()
[![Precision](https://img.shields.io/badge/Precision-89.7%25-brightgreen?style=for-the-badge)]()
[![Recall](https://img.shields.io/badge/Recall-83%25-brightgreen?style=for-the-badge)]()

</div>

---

## Demo

**📹 [Watch the demo video](https://www.loom.com/share/98ab55cf38144bf5b074750dbfe83ae9)** — federated training running live across three Sierra Leone operator nodes

**🌐 [Try the interactive demo](https://bahh99.github.io/SecureFlow-AI/demo/)** — visualise the federation, privacy, and fraud detection in your browser

---

## What this does

SecureFlow AI lets multiple mobile money operators jointly train a fraud detection model, each on their own private transactions, without any operator ever seeing another's data.

The privacy guarantee is mathematical, not contractual. Federated learning combined with differential privacy and SecAgg+ encrypted aggregation makes transaction data impossible to extract from gradient updates.

Training and evaluation use **PaySim**, the standard public benchmark for mobile money fraud research, derived from real African mobile money transaction logs.

**Validated results on real PaySim data:**

| Round | Precision | Recall | False Positive Rate |
|---|---|---|---|
| 1 | 97.7% | 74.7% | 0.3% |
| 2 | 88.9% | 88.7% | 2.1% |
| 3 | 73.3% | 93.6% | 6.6% |
| 4 | 76.0% | 95.3% | 5.8% |
| **5** | **89.7%** | **83.0%** | **1.8%** |

3 operator nodes. 5 training rounds. Differential privacy enabled. **Total raw transaction data shared between operators: 0 bytes.**

---

## Quick start

```bash
# Install dependencies
pip install -e .

# Run a federated training simulation (3 operator nodes, 5 rounds)
flwr run . local-simulation --stream
```

The simulation spins up 3 operator nodes, trains a fraud model collaboratively on PaySim data, applies differential privacy to all gradient updates, aggregates them via SecAgg+, and reports fraud detection metrics at each round.

**No PaySim download needed** — synthetic fallback data is generated automatically. For real PaySim: see [data/README.md](data/README.md).

---

## Project structure

```
SecureFlow-AI/
├── pyproject.toml              ← Flower config + dependencies
├── demo/
│   └── index.html              ← Interactive browser demo
├── data/                       ← PaySim CSV (gitignored)
│   └── README.md
└── secureflow/
    └── core/
        ├── data.py             ← PaySim loading + federated partitioning
        ├── model.py            ← Tabular fraud detector (PyTorch)
        ├── client_app.py       ← Federated client (one per operator)
        ├── server_app.py       ← Federation server (SecAgg+)
        └── defence/
            └── noise.py        ← DP noise + gradient clipping
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
| `dp-noise-std` | 0.05 | DP noise standard deviation |
| `dp-clip-value` | 0.1 | Gradient clipping threshold |
| `is-demo` | true | Small dataset (5K rows per operator) for fast demos |

To run with 5 operators on the full dataset:
```toml
num-clients = 5
is-demo = false
```
Then update `options.num-supernodes = 5` in the federations section to match.

---

## Data: PaySim

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) is the standard public dataset for mobile money fraud research. SecureFlow AI engineers 14 features from PaySim's transaction columns:

- Transaction amount and balance changes (before and after)
- One-hot transaction types: TRANSFER, CASH_OUT, PAYMENT, DEBIT, CASH_IN
- Hour-of-day derived from the simulation step
- Amount-to-balance ratio (fraudsters tend to drain accounts completely)

Real PaySim has roughly 0.13% fraud rate. For demo visibility the synthetic fallback upsamples to 5%.

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
   │Freetown │        │   Bo    │        │ Kenema  │
   │         │        │         │        │         │
   │ Local   │        │ Local   │        │ Local   │
   │ training│        │ training│        │ training│
   │ + DP    │        │ + DP    │        │ + DP    │
   └─────────┘        └─────────┘        └─────────┘
   PaySim             PaySim              PaySim
   shard 1            shard 2             shard 3

   Raw data           Raw data            Raw data
   never leaves       never leaves        never leaves
```

---

## Privacy guarantee

Three reinforcing layers:

1. **Federated learning** — operator data never moves. Each operator trains locally on its own transactions. Raw data stays on-site.

2. **Differential privacy** — every gradient update is clipped and noised before being shared. Even if an update is intercepted, individual transactions cannot be recovered.

3. **SecAgg+ secure aggregation** — operator updates are cryptographically masked so the federation server only ever sees the aggregate, never any individual operator's contribution.

To see what happens without privacy: set `dp-enabled = false` in `pyproject.toml` and re-run. The fraud detection metrics are similar but the gradient updates become reconstructable.

---

## Pilot context

This codebase is the technical foundation for SecureFlow AI's planned 2026 pilot with mobile money operators in **Sierra Leone**. The federated architecture extends naturally across Orange Group's 17 African and Middle Eastern markets without any cross-operator data sharing.

---

## Research foundation

The privacy-preserving core is grounded in peer-reviewed federated learning research, developed through the founder's MSc thesis at ELTE University, Budapest:

📄 [github.com/bahh99/msc-thesis-federated-learning-privacy](https://github.com/bahh99/msc-thesis-federated-learning-privacy)

Key references: Bonawitz et al. (2017) on Secure Aggregation, McMahan et al. (2017) on Federated Averaging, Abadi et al. (2016) on Differential Privacy, Zhu et al. (2019) on Deep Leakage from Gradients.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Contact

**Founder:** Umaru Bah
📧 r11rbq@inf.elte.hu
🐙 [@bahh99](https://github.com/bahh99)
🌍 Hamburg, Germany
