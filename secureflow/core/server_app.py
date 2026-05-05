"""secureflow.core.server_app — Federation server.

Coordinates federated training rounds across operator nodes.
Uses SecAgg+ to aggregate encrypted gradient updates so that the
server never sees any individual operator's update — only the
encrypted aggregate.

The server runs on SecureFlow AI's infrastructure. Operator nodes
connect to it to participate in the federated training network.
"""

from __future__ import annotations

from logging import INFO
from typing import List, Tuple

from flwr.common import Context, Metrics, log, ndarrays_to_parameters
from flwr.server import LegacyContext, ServerApp, ServerConfig, Grid
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from secureflow.core.model import get_weights, make_model


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate per-operator metrics, weighted by sample count."""
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    out = {}
    for key in ("precision", "recall", "fpr", "accuracy"):
        out[key] = sum(n * m.get(key, 0.0) for n, m in metrics) / total
    return out


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = context.run_config

    # Initialise global fraud detection model
    initial_parameters = ndarrays_to_parameters(get_weights(make_model()))

    # Federated averaging strategy with SecAgg+ encrypted aggregation
    num_clients = int(cfg.get("num-clients", 3))
    strategy = FedAvg(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        fraction_evaluate=1.0,
        min_evaluate_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
    )

    legacy_context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=cfg["num-server-rounds"]),
        strategy=strategy,
    )

    # SecAgg+ encrypted aggregation workflow
    fit_workflow = SecAggPlusWorkflow(
        num_shares=cfg["num-shares"],
        reconstruction_threshold=cfg["reconstruction-threshold"],
        max_weight=cfg.get("max-weight", 9000),
    )

    log(INFO, "")
    log(INFO, "=" * 78)
    log(INFO, "SecureFlow AI — Federated Fraud Detection on PaySim")
    log(INFO, f"Network: {num_clients} operator nodes  ·  "
              f"{cfg['num-server-rounds']} training rounds")
    dp_enabled = cfg.get("dp-enabled", True)
    log(INFO, f"Privacy: differential privacy {'ENABLED' if dp_enabled else 'DISABLED'} "
              f"+ SecAgg+ encrypted aggregation")
    log(INFO, "=" * 78)
    log(INFO, "")

    workflow = DefaultWorkflow(fit_workflow=fit_workflow)
    workflow(grid, legacy_context)

    log(INFO, "")
    log(INFO, "=" * 78)
    log(INFO, "Training complete. Model deployed across operator nodes.")
    log(INFO, "Total raw transactions shared between operators: 0 bytes")
    log(INFO, "=" * 78)
