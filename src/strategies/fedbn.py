"""
FedBN (Federated Batch Normalization) Strategy for Flower
Target: Feature & Statistical Heterogeneity (Different ECG device calibrations & populations)
Mechanism: Excludes BatchNorm1d layers from global aggregation.
BN layers remain private and locally updated on each client node.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedBNStrategy(FedAvg):
    """
    Flower Strategy implementing FedBN.
    Server only aggregates non-BatchNorm parameters.
    """
    def __init__(
        self,
        bn_keys: Optional[List[str]] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bn_keys = bn_keys or ["bn", "running_mean", "running_var", "num_batches_tracked"]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Melakukan agregasi FedAvg pada parameter model non-BatchNorm.
        """
        if not results:
            return None, {}

        # Panggil agregasi FedAvg standar pada parameter yang dikirim client
        # (Client FedBN telah memfilter parameter BN sebelum pengiriman)
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            metrics["aggregated_round"] = server_round
            metrics["strategy"] = "fedbn"

        return aggregated_parameters, metrics
