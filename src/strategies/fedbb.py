"""
FedBB: Balanced Loss & Client Balanced Reweighting for Flower
Target: Class & Client Imbalance (Severe minority arrhythmia imbalance across sites)
Mechanism:
  - Client side: Positive-Negative Balanced (PNB) Loss
  - Server side: Client Balanced Reweighting (CBR) using combined sample count (k_b)
    and class balance factor (k_a).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class PNBLoss(nn.Module):
    """
    Positive-Negative Balanced (PNB) Loss for Multi-Label ECG Classification.
    Dynamically weights minority positive classes locally.
    """
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, eps: float = 1e-7):
        super().__init__()
        self.pos_weight = pos_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Compute binary cross entropy with positive reweighting
        loss_pos = -targets * torch.log(probs + self.eps)
        if self.pos_weight is not None:
            loss_pos = loss_pos * self.pos_weight

        loss_neg = -(1.0 - targets) * torch.log(1.0 - probs + self.eps)
        loss = torch.mean(loss_pos + loss_neg)
        return loss


class FedBBStrategy(FedAvg):
    """
    Server strategy implementing Client Balanced Reweighting (CBR).
    Weights each client based on a combination of sample size (k_b)
    and client balance score (k_a).
    """
    def __init__(self, alpha: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha  # Balance weight factor

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Hitung bobot CBR gabungan
        client_weights = []
        client_ndarrays = []

        total_samples = sum(fit_res.num_examples for _, fit_res in results)

        for _, fit_res in results:
            num_examples = fit_res.num_examples
            # Ambil skor keseimbangan lokal dari metrics jika dikirim oleh client
            balance_score = fit_res.metrics.get("balance_score", 1.0)
            
            k_b = num_examples / total_samples
            k_a = balance_score

            combined_weight = (1 - self.alpha) * k_b + self.alpha * k_a
            client_weights.append(combined_weight)
            client_ndarrays.append(parameters_to_ndarrays(fit_res.parameters))

        # Normalisasi bobot
        sum_weights = sum(client_weights)
        norm_weights = [w / sum_weights for w in client_weights]

        # Agregasi terbobot
        # Akumulasi dalam float64 (layer integer, mis. num_batches_tracked BN,
        # tidak bisa menerima hasil float secara in-place)
        orig_dtypes = [param.dtype for param in client_ndarrays[0]]
        weighted_weights: NDArrays = [
            np.zeros(param.shape, dtype=np.float64) for param in client_ndarrays[0]
        ]

        for client_idx, ndarrs in enumerate(client_ndarrays):
            w = norm_weights[client_idx]
            for i, layer in enumerate(ndarrs):
                weighted_weights[i] += layer.astype(np.float64) * w

        # Kembalikan ke dtype asli (dibulatkan untuk tipe integer)
        weighted_weights = [
            np.rint(arr).astype(dt) if np.issubdtype(dt, np.integer) else arr.astype(dt)
            for arr, dt in zip(weighted_weights, orig_dtypes)
        ]

        metrics = {"strategy": "fedbb", "aggregated_round": server_round}
        return ndarrays_to_parameters(weighted_weights), metrics
