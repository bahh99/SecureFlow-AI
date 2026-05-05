"""secureflow.core — Federated fraud detection on the PaySim dataset.

Modules:
    data        — PaySim dataset loading and federated partitioning
    model       — Tabular fraud detection neural network
    defence     — Differential privacy mechanisms (noise + clipping)
    client_app  — Federated learning client (per operator node)
    server_app  — Federation server (SecAgg+ aggregation)
"""
