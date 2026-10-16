"""
FedProx Strategy for Flower (Li et al., 2020)
Target: System & Statistical Heterogeneity (Kompensasi Client Drift via Proximal Term)
Mechanism: Menambahkan penalti proksimal (mu/2) * ||w - w^t||^2 pada loss lokal klien.
"""

from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedProxStrategy(FedAvg):
    """
    Flower Strategy implementing FedProx.
    Mengirim parameter proximal_mu ke setiap klien pada tiap ronde training.
    """
    def __init__(
        self,
        proximal_mu: float = 0.01,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.proximal_mu = proximal_mu

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        client_config_pairs = super().configure_fit(server_round, parameters, client_manager)
        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_ins.config, "proximal_mu": self.proximal_mu, "strategy": "fedprox"},
                ),
            )
            for client, fit_ins in client_config_pairs
        ]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            metrics["aggregated_round"] = server_round
            metrics["strategy"] = "fedprox"
        return aggregated_parameters, metrics
