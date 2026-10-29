#!/usr/bin/env python3
"""
Flower Server Entrypoint for Multi-Dataset ECG FL Benchmark
Strategi diambil dari src/strategies/registry.py (FedAvg, FedBN, FedBB, FedSOL, dst).
Setiap sesi training direkam permanen ke results/runs/<strategi>-<waktu>/ (append-only).
"""

import os
import sys
from pathlib import Path

# Pastikan root direktori selalu ada di Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import signal
import time
import json

import flwr as fl
import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from src.models.resnet1d import ResNet1D, get_model_parameters
from src.strategies.registry import build_strategy, strategy_names, is_registered
from src.results_store import RunRecorder
from src.ckks_client import get_ckks_client


TIMING_FILE = os.path.join(os.getcwd(), "logs", "round_timing.json")
CKKS_METRICS_FILE = os.path.join(os.getcwd(), "logs", "ckks_rounds.json")


def reset_live_timing():
    """Timing LIVE (untuk ETA di dashboard) hanya berlaku per sesi strategi.
    Data permanen ada di results/runs/*/metrics.json, bukan di file ini."""
    try:
        os.makedirs(os.path.dirname(TIMING_FILE), exist_ok=True)
        with open(TIMING_FILE, "w") as f:
            json.dump({}, f)
    except Exception:
        pass
    try:
        if os.path.exists(CKKS_METRICS_FILE):
            os.remove(CKKS_METRICS_FILE)
    except Exception:
        pass


def record_round_start(server_round: int):
    """Mencatat timestamp saat ronde federasi dimulai (live)"""
    try:
        os.makedirs(os.path.dirname(TIMING_FILE), exist_ok=True)
        data = {}
        if os.path.exists(TIMING_FILE):
            with open(TIMING_FILE, "r") as f:
                data = json.load(f)
        now = time.time()
        prev_rnd = str(server_round - 1)
        if prev_rnd in data and "start" in data[prev_rnd] and "duration" not in data[prev_rnd]:
            data[prev_rnd]["end"] = now
            data[prev_rnd]["duration"] = round(now - data[prev_rnd]["start"], 2)

        data[str(server_round)] = {"start": now}
        with open(TIMING_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def record_round_end(server_round: int):
    """Menghitung durasi nyata saat ronde selesai diagregasi (live)"""
    try:
        if os.path.exists(TIMING_FILE):
            with open(TIMING_FILE, "r") as f:
                data = json.load(f)
            rnd_key = str(server_round)
            now = time.time()
            if rnd_key in data and "start" in data[rnd_key]:
                dur = round(now - data[rnd_key]["start"], 2)
                data[rnd_key]["end"] = now
                data[rnd_key]["duration"] = dur
                with open(TIMING_FILE, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"⏱️ [SERVER] Ronde {server_round} selesai dalam {dur} detik ({dur/60:.2f} menit).")
    except Exception:
        pass


def _save_ckks_round(server_round, data):
    try:
        os.makedirs(os.path.dirname(CKKS_METRICS_FILE), exist_ok=True)
        existing = []
        if os.path.exists(CKKS_METRICS_FILE):
            with open(CKKS_METRICS_FILE, "r") as f:
                existing = json.load(f).get("rounds", [])
        data["round"] = server_round
        data["timestamp"] = time.time()
        existing = [r for r in existing if r.get("round") != server_round]
        existing.append(data)
        existing.sort(key=lambda r: r["round"])
        with open(CKKS_METRICS_FILE, "w") as f:
            json.dump({"rounds": existing}, f, indent=2)
    except Exception:
        pass


def _ckks_aggregate(ckks, server_round, results, failures):
    from flwr.common import parameters_to_ndarrays

    total_samples = sum(fr.num_examples for _, fr in results)
    client_chunks = []
    client_weights = []
    shapes_json = None
    num_params = 0
    per_client = []

    for _, fit_res in results:
        m = fit_res.metrics or {}
        chunk_sizes = json.loads(m["ckks_chunk_sizes"])
        ndarrays = parameters_to_ndarrays(fit_res.parameters)
        chunks = [arr.tobytes()[:sz] for arr, sz in zip(ndarrays, chunk_sizes)]
        client_chunks.append(chunks)
        client_weights.append(fit_res.num_examples / total_samples)
        per_client.append({
            "dataset": str(m.get("dataset", "unknown")),
            "encrypt_ms": m.get("ckks_encrypt_ms", 0),
            "bytes_total": m.get("ckks_bytes_total", 0),
            "num_examples": fit_res.num_examples,
        })
        if shapes_json is None:
            shapes_json = m["ckks_shapes"]
            num_params = int(m["ckks_num_params"])

    agg_chunks, agg_metrics = ckks.aggregate(client_chunks, client_weights)
    flat, dec_metrics = ckks.decrypt(agg_chunks, num_params)

    shapes = json.loads(shapes_json)
    arrays = []
    offset = 0
    for shape, dtype_str in shapes:
        size = 1
        for s in shape:
            size *= s
        arr = flat[offset:offset + size].reshape(shape).astype(dtype_str)
        arrays.append(arr)
        offset += size

    metrics = {
        "strategy": results[0][1].metrics.get("strategy", "fedavg"),
        "aggregated_round": server_round,
        "ckks_encrypted": True,
    }
    metrics.update(agg_metrics)
    metrics.update(dec_metrics)

    _save_ckks_round(server_round, {
        "num_clients": len(results),
        "num_params": num_params,
        "num_chunks": len(client_chunks[0]) if client_chunks else 0,
        "total_samples": total_samples,
        "aggregate_ms": agg_metrics.get("ckks_aggregate_ms", 0),
        "decrypt_ms": dec_metrics.get("ckks_decrypt_ms", 0),
        "clients": per_client,
    })

    print(f"🔐 [CKKS] Round {server_round}: aggregate={agg_metrics.get('ckks_aggregate_ms', 0):.0f}ms, "
          f"decrypt={dec_metrics.get('ckks_decrypt_ms', 0):.0f}ms, "
          f"clients={len(results)}, total_samples={total_samples}", flush=True)
    return ndarrays_to_parameters(arrays), metrics


def get_strategy(strategy_name: str, recorder: RunRecorder, num_rounds: int, min_fit_clients: int,
                 min_available_clients: int, initial_params, state_keys):
    strategy_name = strategy_name.lower()

    local_epochs = int(os.environ.get("LOCAL_EPOCHS", 3))
    batch_size = int(os.environ.get("BATCH_SIZE", 32))
    lr = float(os.environ.get("LEARNING_RATE", 0.001))

    def fit_config(server_round: int):
        record_round_start(server_round)
        recorder.start_round(server_round)
        cfg = {
            "current_round": server_round,
            "strategy": strategy_name,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "lr": lr,
        }
        if strategy_name == "fedprox":
            cfg["proximal_mu"] = float(os.environ.get("PROXIMAL_MU", 0.01))
        elif strategy_name == "fedbb":
            cfg["alpha"] = float(os.environ.get("FEDBB_ALPHA", 0.5))
        elif strategy_name == "fedsol":
            cfg["fedsol_mu"] = float(os.environ.get("FEDSOL_MU", 0.01))
        return cfg

    common_args = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_fit_clients,
        "min_available_clients": min_available_clients,
        "initial_parameters": initial_params,
        "on_fit_config_fn": fit_config,
    }

    print(f"🧠 Initializing strategy: {strategy_name.upper()}")
    strategy = build_strategy(strategy_name, **common_args)

    latest = {"params": None}

    orig_aggregate_fit = strategy.aggregate_fit
    def recorded_aggregate_fit(server_round, results, failures):
        ckks = get_ckks_client()
        is_ckks = (
            ckks is not None
            and results
            and results[0][1].metrics
            and results[0][1].metrics.get("ckks_encrypted")
        )

        if is_ckks:
            params, metrics = _ckks_aggregate(ckks, server_round, results, failures)
        else:
            params, metrics = orig_aggregate_fit(server_round, results, failures)

        clients = []
        for _, fit_res in results:
            m = fit_res.metrics or {}
            clients.append({
                "dataset": str(m.get("dataset", "unknown")),
                "num_examples": fit_res.num_examples,
                "train_loss": m.get("loss"),
                "duration": m.get("round_duration"),
            })
        recorder.record_fit(server_round, clients, num_failures=len(failures))
        if params is not None:
            latest["params"] = params
        record_round_end(server_round)
        return params, metrics
    strategy.aggregate_fit = recorded_aggregate_fit

    orig_aggregate_evaluate = strategy.aggregate_evaluate
    def recorded_aggregate_evaluate(server_round, results, failures):
        loss, metrics = orig_aggregate_evaluate(server_round, results, failures)
        if not results:
            return loss, metrics
        clients = []
        for _, ev in results:
            m = ev.metrics or {}
            clients.append({
                "dataset": str(m.get("dataset", "unknown")),
                "num_examples": ev.num_examples,
                "val_loss": ev.loss,
                "f1_macro": m.get("f1_macro"),
                "f1_micro": m.get("f1_micro"),
                "auroc": m.get("auroc"),
                "per_class": {k[len("f1_c_"):]: v for k, v in m.items() if k.startswith("f1_c_")},
            })
        g = recorder.record_eval(server_round, clients, num_failures=len(failures))
        f1 = g.get("f1_macro")
        print(f"📊 [SERVER] Ronde {server_round} | val_loss={g.get('val_loss')} | "
              f"F1-macro={f1} | AUROC={g.get('auroc')}", flush=True)

        # Simpan model global (bobot hasil agregasi ronde ini)
        if latest["params"] is not None:
            try:
                arrays = parameters_to_ndarrays(latest["params"])
                state = {k: torch.from_numpy(np.ascontiguousarray(a)) for k, a in zip(state_keys, arrays[:len(state_keys)])}
                partial = (len(arrays) < len(state_keys)) or (strategy_name == "fedbn")
                recorder.save_model("final", state, server_round, f1, excludes_batchnorm=partial)
                if recorder.is_best(f1):
                    recorder.save_model("best", state, server_round, f1, excludes_batchnorm=partial)
            except Exception as e:
                print(f"⚠️ [SERVER] Gagal menyimpan model ronde {server_round}: {e}", flush=True)
        return loss, metrics
    strategy.aggregate_evaluate = recorded_aggregate_evaluate

    return strategy


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt()


def run_single_strategy(strategy_name: str, num_rounds: int, server_address: str, min_clients: int):
    if not is_registered(strategy_name):
        raise SystemExit(f"❌ Strategi '{strategy_name}' tidak terdaftar. Pilihan: {', '.join(strategy_names())}")

    print(f"\n==================================================================")
    print(f"🏥 Memulai Sesi FL: {strategy_name.upper()} ({num_rounds} Rounds)")
    print(f"==================================================================")

    reset_live_timing()

    seed = int(os.environ.get("SPLIT_SEED", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = ResNet1D(in_channels=12, num_classes=27)
    exclude_bn = (strategy_name.lower() == "fedbn")
    state_dict = get_model_parameters(model, exclude_bn=exclude_bn)
    state_keys = list(state_dict.keys())
    initial_ndarrays = [val.cpu().numpy() for val in state_dict.values()]
    initial_params = ndarrays_to_parameters(initial_ndarrays)

    config = {
        "num_rounds": num_rounds,
        "min_clients": min_clients,
        "local_epochs": int(os.environ.get("LOCAL_EPOCHS", 3)),
        "batch_size": int(os.environ.get("BATCH_SIZE", 32)),
        "learning_rate": float(os.environ.get("LEARNING_RATE", 0.001)),
        "val_fraction": float(os.environ.get("VAL_FRACTION", 0.2)),
        "split_seed": int(os.environ.get("SPLIT_SEED", 42)),
        "model": "resnet1d",
    }
    if strategy_name.lower() == "fedprox":
        config["proximal_mu"] = float(os.environ.get("PROXIMAL_MU", 0.01))
    elif strategy_name.lower() == "fedbb":
        config["alpha"] = float(os.environ.get("FEDBB_ALPHA", 0.5))
    elif strategy_name.lower() == "fedsol":
        config["fedsol_mu"] = float(os.environ.get("FEDSOL_MU", 0.01))
    recorder = RunRecorder(strategy_name.lower(), config)
    print(f"💾 Hasil run disimpan permanen di: {recorder.dir}", flush=True)

    strategy = get_strategy(
        strategy_name=strategy_name,
        recorder=recorder,
        num_rounds=num_rounds,
        min_fit_clients=min_clients,
        min_available_clients=min_clients,
        initial_params=initial_params,
        state_keys=state_keys,
    )

    status = "failed"
    try:
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy
        )
        done = sum(1 for rd in recorder._rounds.values() if "global" in rd)
        status = "completed" if done >= num_rounds else "incomplete"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        recorder.finish(status)
        print(f"🗂️ Run {recorder.run_id}: status={status}", flush=True)
    print(f"✅ Selesai Sesi Eksperimen: {strategy_name.upper()}!\n")


def main():
    signal.signal(signal.SIGTERM, _raise_interrupt)

    parser = argparse.ArgumentParser(description="Flower ECG FL Server")
    parser.add_argument("--strategy", type=str, default=os.environ.get("STRATEGY", "fedbn"))
    parser.add_argument("--rounds", type=int, default=int(os.environ.get("NUM_ROUNDS", 50)))
    parser.add_argument("--host", type=str, default=os.environ.get("SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=str, default=os.environ.get("SERVER_PORT", "8080"))
    parser.add_argument("--min_clients", type=int, default=int(os.environ.get("MIN_FIT_CLIENTS", 3)))
    parser.add_argument("--proximal_mu", type=float, default=None)
    parser.add_argument("--fedbb_alpha", type=float, default=None)
    parser.add_argument("--fedsol_mu", type=float, default=None)
    args = parser.parse_args()

    if args.proximal_mu is not None:
        os.environ["PROXIMAL_MU"] = str(args.proximal_mu)
    if args.fedbb_alpha is not None:
        os.environ["FEDBB_ALPHA"] = str(args.fedbb_alpha)
    if args.fedsol_mu is not None:
        os.environ["FEDSOL_MU"] = str(args.fedsol_mu)

    server_address = f"{args.host}:{args.port}"

    # Dukungan urutan kustom: misal "fedbb,fedsol,fedbn" atau "all" (semua strategi terdaftar)
    if "," in args.strategy:
        strategies = [s.strip().lower() for s in args.strategy.split(",") if s.strip()]
    elif args.strategy.lower() == "all":
        strategies = strategy_names()
    else:
        strategies = None

    if strategies:
        strat_names = " -> ".join([s.upper() for s in strategies])
        print(f"\n🚀 [BENCHMARK MULTI-STRATEGI] Menjalankan {len(strategies)} Algoritma Sesuai Urutan:")
        print(f"   Urutan Eksekusi: {strat_names} (Masing-masing {args.rounds} Rounds)")
        for idx, s in enumerate(strategies):
            run_single_strategy(s, args.rounds, server_address, args.min_clients)
            if idx < len(strategies) - 1:
                next_strat = strategies[idx + 1]
                print(f"\n⏸️ Menunggu 6 detik untuk pelepasan socket gRPC sebelum memulai {next_strat.upper()}...", flush=True)
                time.sleep(6)
        print(f"🎉 SELURUH EKSPERIMEN ({strat_names}) SELESAI!\n")
    else:
        run_single_strategy(args.strategy, args.rounds, server_address, args.min_clients)


if __name__ == "__main__":
    main()
