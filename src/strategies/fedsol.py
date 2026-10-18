"""
FedSOL: Orthogonal Learning with Proximal Restriction
Target: Client Drift (divergence between local model updates and global objective)
Mechanism:
  - Local Client: Computes proximal penalty gradient against global weights,
    introduces controlled perturbation, and updates local parameters orthogonally
    to the proximal restriction direction.
  - Server: Standard FedAvg aggregation across all model parameters.
"""

import copy
import torch
import torch.nn as nn
from flwr.server.strategy import FedAvg


class FedSOLOptimizer:
    """
    Client-side optimizer helper for Orthogonal Step & Proximal Restriction.
    """
    def __init__(self, model: nn.Module, global_model: nn.Module, lr: float = 0.001, mu: float = 0.01):
        self.model = model
        self.global_model = global_model
        self.lr = lr
        self.mu = mu
        self.base_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def step(self, loss: torch.Tensor):
        self.base_optimizer.zero_grad()
        loss.backward()

        # Orthogonal projection of local gradient w.r.t proximal drift
        with torch.no_grad():
            for param, global_param in zip(self.model.parameters(), self.global_model.parameters()):
                if param.grad is not None:
                    # Direction of drift from global
                    drift_dir = param.data - global_param.data
                    drift_norm = torch.norm(drift_dir) + 1e-8

                    # Orthogonal projection: grad_orth = grad - ((grad . drift) / ||drift||^2) * drift
                    proj = torch.sum(param.grad * drift_dir) / (drift_norm ** 2)
                    param.grad.data = param.grad.data - (proj * drift_dir) + (self.mu * drift_dir)

        self.base_optimizer.step()


class FedSOLStrategy(FedAvg):
    """
    Server strategy for FedSOL. Uses FedAvg aggregation over all model parameters.
    """
    def __init__(self, mu: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu
