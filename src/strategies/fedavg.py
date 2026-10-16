"""
FedAvg (baseline) Strategy for Flower
Mechanism: Agregasi rata-rata terbobot (jumlah sampel) atas SEMUA parameter model,
tanpa modifikasi loss/optimizer di sisi client. Dipakai sebagai pembanding (baseline).
"""

from typing import Dict, List, Optional, Tuple, Union
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedAvgStrategy(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            metrics["aggregated_round"] = server_round
            metrics["strategy"] = "fedavg"
        return aggregated_parameters, metrics
