"""
SCAFFOLD Strategy for Flower (Karimireddy et al., 2020)
Target: Client Drift Akut pada heterogenitas distribusi ekstrem
Mechanism:
  - Memelihara control variates global (c) di server dan lokal (c_i) di tiap klien.
  - Koreksi gradien pada optimasi lokal: g_i - c_i + c.
  - Update control variates: Delta c_i = c_i^+ - c_i; c <- c + (1/N) * sum(Delta c_i).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    EvaluateIns,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class SCAFFOLDStrategy(FedAvg):
    """
    Flower Strategy implementing SCAFFOLD.
    Parameter yang dikomunikasikan berupa gabungan [W_model, C_control_variate].
    """
    def __init__(
        self,
        total_clients: int = 3,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.total_clients = total_clients
        self.num_model_layers: Optional[int] = None
        self.server_c: Optional[NDArrays] = None
        self.current_weights: Optional[NDArrays] = None

        # Inisialisasi kontrol variat server (c) jika initial_parameters tersedia
        if self.initial_parameters is not None:
            init_ndarrays = parameters_to_ndarrays(self.initial_parameters)
            self.num_model_layers = len(init_ndarrays)
            self.server_c = [np.zeros_like(p) for p in init_ndarrays]
            self.current_weights = init_ndarrays
            # Kirimkan [W_0, C_0] ke ronde 1
            combined = list(init_ndarrays) + list(self.server_c)
            self.initial_parameters = ndarrays_to_parameters(combined)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Pastikan parameter yang dikirim ke klien berisi [W, C_server]."""
        client_config_pairs = super().configure_fit(server_round, parameters, client_manager)
        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_ins.config, "strategy": "scaffold"},
                ),
            )
            for client, fit_ins in client_config_pairs
        ]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Kirim hanya parameter bobot model (tanpa control variate) untuk evaluasi."""
        ndarrays = parameters_to_ndarrays(parameters)
        if self.num_model_layers is not None and len(ndarrays) > self.num_model_layers:
            model_only_params = ndarrays_to_parameters(ndarrays[:self.num_model_layers])
        else:
            model_only_params = parameters

        return super().configure_evaluate(server_round, model_only_params, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Tiap klien mengirim [W_i, Delta_c_i]
        first_res = parameters_to_ndarrays(results[0][1].parameters)
        if self.num_model_layers is None:
            self.num_model_layers = len(first_res) // 2

        M = self.num_model_layers
        if self.server_c is None:
            self.server_c = [np.zeros_like(p) for p in first_res[:M]]

        # Pisahkan bobot model (W) dan delta control variates (Delta c)
        model_results = []
        delta_cs = []
        for client, fit_res in results:
            arrs = parameters_to_ndarrays(fit_res.parameters)
            w_i = arrs[:M]
            delta_c_i = arrs[M:]
            delta_cs.append(delta_c_i)
            # Buat FitRes tiruan dengan hanya bobot model untuk diagregasi FedAvg standar
            res_proxy = FitRes(
                status=fit_res.status,
                parameters=ndarrays_to_parameters(w_i),
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics
            )
            model_results.append((client, res_proxy))

        # 1. Agregasi bobot model global x <- sum(p_i * y_i)
        aggregated_model_params, metrics = super().aggregate_fit(server_round, model_results, failures)
        if aggregated_model_params is None:
            return None, metrics

        aggregated_model_ndarrays = parameters_to_ndarrays(aggregated_model_params)
        self.current_weights = aggregated_model_ndarrays

        # 2. Update control variate global c <- c + (1 / N) * sum(Delta c_i)
        N = max(1, self.total_clients)
        num_sampled = len(delta_cs)
        for layer_idx in range(M):
            sum_delta = np.zeros_like(self.server_c[layer_idx], dtype=np.float64)
            for client_idx in range(num_sampled):
                sum_delta += delta_cs[client_idx][layer_idx].astype(np.float64)
            # c_new = c_old + (1/N) * sum_delta
            delta_layer = (sum_delta / float(N)).astype(self.server_c[layer_idx].dtype)
            self.server_c[layer_idx] = self.server_c[layer_idx] + delta_layer

        # 3. Gabungkan [W_global, C_global] untuk ronde berikutnya
        combined = list(aggregated_model_ndarrays) + list(self.server_c)
        metrics["aggregated_round"] = server_round
        metrics["strategy"] = "scaffold"

        return ndarrays_to_parameters(combined), metrics
