#!/usr/bin/env python3
"""
Flower Server Entrypoint for Multi-Dataset ECG FL Benchmark
Supports FedBN, FedBB, and FedSOL strategies.
"""

import os
import sys
from pathlib import Path

# Pastikan root direktori selalu ada di Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters

from src.models.resnet1d import ResNet1D, get_model_parameters
from src.strategies.fedbn import FedBNStrategy
from src.strategies.fedbb import FedBBStrategy
from src.strategies.fedsol import FedSOLStrategy

def get_strategy(strategy_name: str, num_rounds: int, min_fit_clients: int, min_available_clients: int, initial_params):
    strategy_name = strategy_name.lower()

    def fit_config(server_round: int):
        """Kirim instruksi strategi dan hyperparameter ke semua klien dari server pusat"""
        return {
            "current_round": server_round,
            "strategy": strategy_name,
            "local_epochs": 3,
            "batch_size": 32,
            "lr": 0.001
        }

    common_args = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_fit_clients,
        "min_available_clients": min_available_clients,
        "initial_parameters": initial_params,
        "on_fit_config_fn": fit_config,
        "on_evaluate_config_fn": fit_config,
    }

    if strategy_name == "fedbn":
        print("🧠 Initializing FedBN Strategy (Excluding BatchNorm1d from Aggregation)")
        return FedBNStrategy(**common_args)
    elif strategy_name == "fedbb":
        print("⚖️ Initializing FedBB Strategy (CBR Aggregation & Balanced Loss)")
        return FedBBStrategy(alpha=0.5, **common_args)
    elif strategy_name == "fedsol":
        print("📐 Initializing FedSOL Strategy (Orthogonal Optimization Aggregation)")
        return FedSOLStrategy(**common_args)
    else:
        print(f"⚠️ Strategy '{strategy_name}' unrecognized, defaulting to FedAvg")
        return fl.server.strategy.FedAvg(**common_args)

def run_single_strategy(strategy_name: str, num_rounds: int, server_address: str, min_clients: int):
    print(f"\n==================================================================")
    print(f"🏥 Memulai Sesi FL: {strategy_name.upper()} ({num_rounds} Rounds)")
    print(f"==================================================================")

    model = ResNet1D(in_channels=12, num_classes=27)
    exclude_bn = (strategy_name.lower() == "fedbn")
    state_dict = get_model_parameters(model, exclude_bn=exclude_bn)
    initial_ndarrays = [val.cpu().numpy() for val in state_dict.values()]
    initial_params = ndarrays_to_parameters(initial_ndarrays)

    strategy = get_strategy(
        strategy_name=strategy_name,
        num_rounds=num_rounds,
        min_fit_clients=min_clients,
        min_available_clients=min_clients,
        initial_params=initial_params
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )
    print(f"✅ Selesai Sesi Eksperimen: {strategy_name.upper()}!\n")


def main():
    parser = argparse.ArgumentParser(description="Flower ECG FL Server")
    parser.add_argument("--strategy", type=str, default=os.environ.get("STRATEGY", "fedbn"))
    parser.add_argument("--rounds", type=int, default=int(os.environ.get("NUM_ROUNDS", 50)))
    parser.add_argument("--host", type=str, default=os.environ.get("SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=str, default=os.environ.get("SERVER_PORT", "8080"))
    parser.add_argument("--min_clients", type=int, default=int(os.environ.get("MIN_FIT_CLIENTS", 3)))
    args = parser.parse_args()

    server_address = f"{args.host}:{args.port}"

    if args.strategy.lower() == "all":
        print(f"🚀 [BENCHMARK SEMUA STRATEGI] Menjalankan 3 Algoritma Berturutan:")
        print(f"   1. FedBN -> 2. FedBB -> 3. FedSOL (Masing-masing {args.rounds} Rounds)")
        strategies = ["fedbn", "fedbb", "fedsol"]
        for s in strategies:
            run_single_strategy(s, args.rounds, server_address, args.min_clients)
        print("🎉 SELURUH BENCHMARK KOMPARATIF SELESAI (FedBN + FedBB + FedSOL)!")
    else:
        run_single_strategy(args.strategy, args.rounds, server_address, args.min_clients)


if __name__ == "__main__":
    main()
