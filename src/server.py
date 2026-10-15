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

def record_round_start(server_round: int):
    """Mencatat timestamp saat ronde federasi dimulai"""
    import time, json
    timing_file = os.path.join(os.getcwd(), "logs", "round_timing.json")
    try:
        os.makedirs(os.path.dirname(timing_file), exist_ok=True)
        data = {}
        if os.path.exists(timing_file):
            with open(timing_file, "r") as f:
                data = json.load(f)
        now = time.time()
        prev_rnd = str(server_round - 1)
        if prev_rnd in data and "start" in data[prev_rnd] and "duration" not in data[prev_rnd]:
            data[prev_rnd]["end"] = now
            data[prev_rnd]["duration"] = round(now - data[prev_rnd]["start"], 2)

        data[str(server_round)] = {"start": now}
        with open(timing_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def record_round_end(server_round: int):
    """Mencatat timestamp dan menghitung durasi nyata saat ronde federasi selesai diagregasi"""
    import time, json
    timing_file = os.path.join(os.getcwd(), "logs", "round_timing.json")
    try:
        if os.path.exists(timing_file):
            with open(timing_file, "r") as f:
                data = json.load(f)
            rnd_key = str(server_round)
            now = time.time()
            if rnd_key in data and "start" in data[rnd_key]:
                dur = round(now - data[rnd_key]["start"], 2)
                data[rnd_key]["end"] = now
                data[rnd_key]["duration"] = dur
                with open(timing_file, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"⏱️ [SERVER] Ronde {server_round} selesai dalam {dur} detik ({dur/60:.2f} menit).")
    except Exception:
        pass


def get_strategy(strategy_name: str, num_rounds: int, min_fit_clients: int, min_available_clients: int, initial_params):
    strategy_name = strategy_name.lower()

    def fit_config(server_round: int):
        record_round_start(server_round)
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
    }

    if strategy_name == "fedbn":
        print("🧠 Initializing FedBN Strategy (Excluding BatchNorm1d from Aggregation)")
        strategy = FedBNStrategy(**common_args)
    elif strategy_name == "fedbb":
        print("⚖️ Initializing FedBB Strategy (CBR Aggregation & Balanced Loss)")
        strategy = FedBBStrategy(alpha=0.5, **common_args)
    elif strategy_name == "fedsol":
        print("📐 Initializing FedSOL Strategy (Orthogonal Optimization Aggregation)")
        strategy = FedSOLStrategy(**common_args)
    else:
        print(f"⚠️ Strategy '{strategy_name}' unrecognized, defaulting to FedAvg")
        strategy = fl.server.strategy.FedAvg(**common_args)

    orig_aggregate_fit = strategy.aggregate_fit
    def timed_aggregate_fit(server_round, results, failures):
        res = orig_aggregate_fit(server_round, results, failures)
        record_round_end(server_round)
        return res
    strategy.aggregate_fit = timed_aggregate_fit
    return strategy

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
