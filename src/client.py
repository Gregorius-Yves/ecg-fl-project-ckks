#!/usr/bin/env python3
"""
Flower Client Entrypoint for Multi-Dataset ECG FL Benchmark
Integrates ResNet1D, Local BatchNorm preservation for FedBN,
PNB Loss for FedBB, and Orthogonal Optimizer for FedSOL.
"""

import os
import sys
from pathlib import Path

# Pastikan root direktori selalu ada di Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import copy
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.resnet1d import ResNet1D, is_batchnorm_param
from src.strategies.fedbb import PNBLoss
from src.strategies.fedsol import FedSOLOptimizer


class ECGFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, dataset_name: str, strategy: str, batch_size: int = 32, epochs: int = 3, lr: float = 0.001):
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.strategy = strategy.lower()
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet1D(in_channels=12, num_classes=27).to(self.device)
        self.global_model = copy.deepcopy(self.model)

        # Setup Loss function
        if self.strategy == "fedbb":
            self.criterion = PNBLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.strategy == "fedsol":
            self.fedsol_opt = FedSOLOptimizer(self.model, self.global_model, lr=self.lr)

        # Setup data loader
        self.data_dir = os.environ.get("DATA_DIR", f"data/{self.dataset_name}")
        self._init_dataloader()

    def _init_dataloader(self):
        from src.preprocessing.dataset import ECGDataset
        real_dataset = ECGDataset(data_dir=self.data_dir, dataset_name=self.dataset_name)

        if len(real_dataset) == 0:
            error_msg = (
                f"\n❌ [ZERO FABRICATION POLICY] Dataset asli '{self.dataset_name}' tidak ditemukan di '{self.data_dir}'!\n"
                f"   Sistem ini strictly menolak data fabrikasi/dummy.\n"
                f"   Silakan unduh dataset asli PhysioNet terlebih dahulu dengan perintah:\n"
                f"   python3 download_data.py --dataset {self.dataset_name}\n"
            )
            print(error_msg)
            raise FileNotFoundError(error_msg)

        print(f"✅ Memuat {len(real_dataset)} rekaman sinyal EKG asli dari '{self.data_dir}'.")
        self.train_loader = DataLoader(
            real_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

    def get_parameters(self, config):
        """Ambil parameter model untuk dikirim ke server"""
        if self.strategy == "fedbn":
            # FedBN: jangan kirim parameter BatchNorm1d
            return [val.cpu().numpy() for name, val in self.model.state_dict().items() if not is_batchnorm_param(name)]
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Pasang parameter global dari server ke model lokal"""
        state_dict = self.model.state_dict()
        if self.strategy == "fedbn":
            # FedBN: Hanya update layer non-BN
            non_bn_keys = [k for k in state_dict.keys() if not is_batchnorm_param(k)]
            for k, param in zip(non_bn_keys, parameters):
                state_dict[k] = torch.tensor(param).to(self.device)
        else:
            for k, param in zip(state_dict.keys(), parameters):
                state_dict[k] = torch.tensor(param).to(self.device)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        # Otomatis baca instruksi strategi dan hyperparameter dari Server Pusat
        server_strategy = config.get("strategy", self.strategy).lower()
        server_round = config.get("current_round", 1)
        epochs = config.get("local_epochs", self.epochs)

        self.strategy = server_strategy
        print(f"\n📢 [ROUND {server_round}] Instruksi dari Server: Menjalankan Strategi '{self.strategy.upper()}' ({epochs} epoch)")

        # Sesuaikan Loss Function & Optimizer sesuai perintah Server
        if self.strategy == "fedbb":
            self.criterion = PNBLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.set_parameters(parameters)
        if self.strategy == "fedsol":
            self.global_model.load_state_dict(self.model.state_dict())
            if not hasattr(self, "fedsol_opt"):
                self.fedsol_opt = FedSOLOptimizer(self.model, self.global_model, lr=self.lr)

        self.model.train()
        total_loss = 0.0
        for epoch in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                if self.strategy == "fedsol":
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                    self.fedsol_opt.step(loss)
                else:
                    self.optimizer.zero_grad()
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

        num_examples = len(self.train_loader.dataset)
        avg_loss = total_loss / (epochs * len(self.train_loader))
        metrics = {
            "loss": float(avg_loss),
            "client_id": self.client_id,
            "dataset": self.dataset_name,
            "strategy": self.strategy,
            "balance_score": 1.0
        }
        return self.get_parameters(config={}), num_examples, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        return float(avg_loss), len(self.train_loader.dataset), {"val_loss": float(avg_loss)}


def main():
    parser = argparse.ArgumentParser(description="Flower ECG FL Client")
    parser.add_argument("--server", type=str, default=os.environ.get("SERVER_ADDRESS", "10.34.100.114:8080"))
    parser.add_argument("--client_id", type=str, default=os.environ.get("CLIENT_ID", "1"))
    parser.add_argument("--dataset", type=str, default=os.environ.get("DATASET_NAME", "ptbxl"))
    parser.add_argument("--strategy", type=str, default=os.environ.get("STRATEGY", "auto"), help="Strategi FL (otomatis dikontrol oleh server jika 'auto')")
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", 32)))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("LOCAL_EPOCHS", 3)))
    args = parser.parse_args()

    print(f"\n🏥 Starting ECG Client: {args.dataset.upper()} (ID: {args.client_id})")
    print(f"🔗 Target Server: {args.server} | Strategy: {args.strategy.upper()}")

    client = ECGFlowerClient(
        client_id=args.client_id,
        dataset_name=args.dataset,
        strategy=args.strategy,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    # Auto-reconnect loop: Klien tetap standby jika server menjalankan eksperimen berurutan (FedBN -> FedBB -> FedSOL)
    import time
    while True:
        try:
            print("⏳ Menghubungkan ke Flower Server...", flush=True)
            fl.client.start_numpy_client(server_address=args.server, client=client)
            print("🏁 Sesi eksperimen selesai. Standby menunggu ronde/strategi berikutnya...", flush=True)
            time.sleep(3)
        except KeyboardInterrupt:
            print("\n🛑 Klien dihentikan oleh pengguna.")
            break
        except Exception as e:
            print(f"⚠️ Terputus dari server ({e}). Mencoba menyambung kembali dalam 5 detik...", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
