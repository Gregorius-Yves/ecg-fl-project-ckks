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

import threading
import urllib.request
import json
import time
import collections

from src.ckks_client import get_ckks_client


def set_seed(seed: int = 42):
    """Set random seed secara global untuk reproduktibilitas eksperimen."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _effective_cpu_count() -> int:
    """Jumlah core yang benar-benar tersedia (affinity + kuota cgroup/Docker), bisa di-override TORCH_THREADS."""
    env = os.environ.get("TORCH_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 1
    for path in ("/sys/fs/cgroup/cpu.max",):
        try:
            quota, period = open(path).read().split()[:2]
            if quota != "max":
                n = min(n, max(1, int(int(quota) / int(period))))
        except Exception:
            pass
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            n = min(n, max(1, q // p))
    except Exception:
        pass
    return max(1, n)



class _TensorSplit:
    """Subset tensor (signals, labels) yang kompatibel dengan FastTensorLoader."""
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)


class FastTensorLoader:
    """Loader batch langsung dari tensor di RAM (tanpa collate/worker overhead per-sampel)."""
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.signals = dataset.signals
        self.labels = dataset.labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        n = len(self.signals)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.signals)
        idx = torch.randperm(n) if self.shuffle else torch.arange(n)
        for i in range(len(self)):
            b = idx[i * self.batch_size:(i + 1) * self.batch_size]
            yield self.signals[b], self.labels[b]

class HeartbeatThread(threading.Thread):
    def __init__(self, client, server_address: str, interval: float = 2.0):
        super().__init__(daemon=True)
        self.client = client
        server_host = server_address.split(":")[0] if ":" in server_address else server_address
        self.url = f"http://{server_host}:8501/api/client/heartbeat"
        self.interval = interval
        self.running = True
        self.has_logged_success = False

    def run(self):
        print(f"💓 [HEARTBEAT] Memulai pelaporan status ke Dashboard -> {self.url}", flush=True)
        consecutive_errors = 0
        while self.running:
            try:
                # Hanya gunakan log sesi berjalan (in-memory) agar tidak ada log basi dari file lama
                recent_logs = []
                if hasattr(self.client, "log_history") and self.client.log_history:
                    recent_logs = [line if line.endswith("\n") else line + "\n" for line in list(self.client.log_history)[-30:]]

                payload = {
                    "log_version": 2,
                    "client_id": self.client.client_id,
                    "dataset": self.client.dataset_name,
                    "status": getattr(self.client, "current_status", "online"),
                    "detail": getattr(self.client, "current_detail", "Online"),
                    "current_round": getattr(self.client, "current_round", 0),
                    "epoch": getattr(self.client, "current_epoch", 0),
                    "total_epochs": getattr(self.client, "total_epochs", 3),
                    "batch": getattr(self.client, "current_batch", 0),
                    "total_batches": getattr(self.client, "total_batches", 0),
                    "learning_rate": getattr(self.client, "current_lr", getattr(self.client, "lr", 0.001)),
                    "measured_duration": getattr(self.client, "last_round_duration", 0),
                    "elapsed_in_round": round(time.time() - getattr(self.client, "current_round_start_time", time.time()), 1) if getattr(self.client, "current_status", "") == "training" else 0,
                    "logs": "".join(recent_logs)
                }
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    self.url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    if not self.has_logged_success:
                        print(f"✅ [HEARTBEAT] Terhubung ke Dashboard Web UI ({self.url})!", flush=True)
                        self.has_logged_success = True
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors == 1 or consecutive_errors % 30 == 0:
                    print(f"⚠️ [HEARTBEAT] Gagal kirim heartbeat ke Dashboard ({self.url}): {e}", flush=True)
            time.sleep(self.interval)


class ECGFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, dataset_name: str, strategy: str, batch_size: int = 32, epochs: int = 3, lr: float = 0.001):
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.strategy = strategy.lower()
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        # Status tracking untuk Dashboard & estimasi waktu nyata
        self.current_status = "online"
        self.current_detail = "Online (Standby)"
        self.current_round = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.current_round_start_time = 0.0
        self.last_round_duration = 0.0
        self.log_history = collections.deque(maxlen=40)
        self.log_history.append(f"🏥 Klien {self.dataset_name.upper()} (ID: {self.client_id}) aktif & siap.")

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

    def reset_state(self):
        """Reset state internal klien (strategi, model lokal, optimizer) antar sesi strategi."""
        set_seed(int(os.environ.get("SPLIT_SEED", 42)))
        self.strategy = "auto"
        self.current_round = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.current_round_start_time = 0.0
        self.last_round_duration = 0.0
        self.current_status = "online"
        self.current_detail = "Online (Standby)"
        # Reset model lokal ke bobot awal yang baru dan bersih
        self.model = ResNet1D(in_channels=12, num_classes=27).to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if hasattr(self, "fedsol_opt"):
            delattr(self, "fedsol_opt")
        self.client_c_dict = None
        self.server_c = None
        self.delta_c = None

    def _init_dataloader(self):
        from src.preprocessing.dataset import ECGDataset
        real_dataset = ECGDataset(data_dir=self.data_dir, dataset_name=self.dataset_name)

        if len(real_dataset) == 0:
            error_msg = (
                f"\n❌ [ZERO FABRICATION POLICY] File dataset pra-proses '{self.dataset_name}_preprocessed.pt' tidak ditemukan!\n"
                f"   Training hanya memakai tensor hasil preprocessing (tidak membaca .hea/.dat langsung).\n"
                f"   Jalankan: python3 download_data.py --dataset {self.dataset_name}\n"
                f"   lalu:     python3 preprocess_data.py --dataset {self.dataset_name}\n"
            )
            print(error_msg)
            raise FileNotFoundError(error_msg)

        print(f"✅ Memuat {len(real_dataset)} rekaman sinyal EKG asli dari '{self.data_dir}'.")
        # Optimasi RAM: baca & praproses SEKALI ke tensor di RAM, lalu semua epoch/ronde berjalan dari memori
        if not getattr(real_dataset, "use_preprocessed", False):
            real_dataset.cache_in_ram = False  # cache per-worker tidak berguna & boros
            real_dataset.preload_to_ram(num_workers=min(4, os.cpu_count() or 1))
        # Pastikan tensor float32 kontigu
        real_dataset.signals = torch.as_tensor(real_dataset.signals, dtype=torch.float32).contiguous()
        real_dataset.labels = torch.as_tensor(real_dataset.labels, dtype=torch.float32).contiguous()
        n_threads = _effective_cpu_count()
        torch.set_num_threads(n_threads)
        print(f"🧵 PyTorch threads: {n_threads} (cpu_count host={os.cpu_count()})", flush=True)

        # Split train/validasi deterministik (seed tetap -> identik antar strategi & antar run)
        val_fraction = float(os.environ.get("VAL_FRACTION", 0.2))
        split_seed = int(os.environ.get("SPLIT_SEED", 42))
        n_total = len(real_dataset.signals)
        perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(split_seed))
        n_val = max(1, int(round(n_total * val_fraction)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        train_set = _TensorSplit(real_dataset.signals[train_idx].contiguous(), real_dataset.labels[train_idx].contiguous())
        val_set = _TensorSplit(real_dataset.signals[val_idx].contiguous(), real_dataset.labels[val_idx].contiguous())
        self.train_set = train_set
        self.val_set = val_set
        print(f"✂️ Split data: train={len(train_set)} | validasi={len(val_set)} (seed={split_seed})", flush=True)

        # Hitung skor keseimbangan kelas lokal untuk CBR (FedBB Subbab 2.4.2)
        counts = self.train_set.labels.sum(dim=0)
        self.balance_score = float(1.0 / (1.0 + counts.std() / (counts.mean() + 1e-8)))
        print(f"⚖️ Skor keseimbangan kelas ({self.dataset_name}): {self.balance_score:.4f}", flush=True)

        self.train_loader = FastTensorLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.eval_loader = FastTensorLoader(self.val_set, batch_size=max(self.batch_size, 128), shuffle=False)

    def get_parameters(self, config):
        """Ambil parameter model untuk dikirim ke server"""
        if self.strategy == "fedbn":
            # FedBN: jangan kirim parameter BatchNorm1d
            return [val.cpu().numpy() for name, val in self.model.state_dict().items() if not is_batchnorm_param(name)]
        if self.strategy == "scaffold" and getattr(self, "delta_c", None) is not None:
            # SCAFFOLD: kirim [W_local, Delta_c_local]
            w_arrays = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
            dc_arrays = [dc.cpu().numpy() for dc in self.delta_c]
            return w_arrays + dc_arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Pasang parameter global dari server ke model lokal"""
        state_dict = self.model.state_dict()
        non_bn_keys = [k for k in state_dict.keys() if not is_batchnorm_param(k)]
        num_keys = len(state_dict.keys())
        if len(parameters) == len(non_bn_keys):
            # Server mengirim parameter parsial tanpa BN (Skema FedBN)
            for k, param in zip(non_bn_keys, parameters):
                state_dict[k] = torch.from_numpy(np.ascontiguousarray(param)).to(self.device)
        else:
            # Server mengirim parameter lengkap (FedAvg / FedBB / FedSOL / FedProx / SCAFFOLD)
            for k, param in zip(state_dict.keys(), parameters[:num_keys]):
                state_dict[k] = torch.from_numpy(np.ascontiguousarray(param)).to(self.device)

            # Jika SCAFFOLD, parameter ke-2M berisi server control variate c
            if len(parameters) == 2 * num_keys:
                self.server_c = [
                    torch.from_numpy(np.ascontiguousarray(c)).to(self.device)
                    for c in parameters[num_keys:]
                ]
        self.model.load_state_dict(state_dict)

    def _grad_at_global(self):
        """Menghitung gradien rata-rata pada model global sebelum pelatihan lokal (SCAFFOLD Option I).
        Option I tidak bergantung pada jenis optimizer lokal (sepenuhnya kompatibel dengan Adam)."""
        self.model.eval()  # Jangan ubah statistik running BatchNorm
        g = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        nb = 0
        for x, y in self.train_loader:
            self.model.zero_grad()
            logits = self.model(x.to(self.device))
            loss = self.criterion(logits, y.to(self.device))
            loss.backward()
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        g[n] += p.grad
            nb += 1
        self.model.train()
        nb = max(1, nb)
        return {n: v / nb for n, v in g.items()}

    def fit(self, parameters, config):
        # Otomatis baca instruksi strategi dan hyperparameter dari Server Pusat
        server_strategy = config.get("strategy", self.strategy).lower()
        server_round = config.get("current_round", 1)
        epochs = config.get("local_epochs", self.epochs)

        # Jika ronde 1, bersihkan riwayat log dan durasi dari eksperimen sebelumnya
        if server_round == 1:
            self.log_history.clear()
            self.last_round_duration = 0.0
            self.log_history.append(f"=== [MEMULAI SESI BARU: RONDE 1] Strategi: {server_strategy.upper()} ===")
            print(f"\n=== [MEMULAI SESI BARU: RONDE 1] Strategi: {server_strategy.upper()} ===", flush=True)

        t_round_start = time.time()
        self.current_round_start_time = t_round_start
        self.strategy = server_strategy
        self.current_round = server_round
        self.total_epochs = epochs
        self.current_lr = float(config.get("lr", self.lr))

        # Sinkronkan batch size jika server meminta ukuran berbeda
        server_batch_size = int(config.get("batch_size", self.batch_size))
        if server_batch_size != self.batch_size and hasattr(self, "train_set"):
            print(f"🔄 Menyesuaikan batch size klien: {self.batch_size} -> {server_batch_size}", flush=True)
            self.batch_size = server_batch_size
            self.train_loader = FastTensorLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
            self.eval_loader = FastTensorLoader(self.val_set, batch_size=max(self.batch_size, 128), shuffle=False)

        # Sinkronkan lr ke Adam optimizer
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.current_lr

        self.total_batches = len(self.train_loader)
        self.current_epoch = 1
        self.current_batch = 0
        self.current_status = "training"
        self.current_detail = f"Ronde {server_round} (Memulai)"
        print(f"\n📢 [ROUND {server_round}] Instruksi dari Server: Menjalankan Strategi '{self.strategy.upper()}' ({epochs} epoch, lr={self.current_lr}, batch_size={self.batch_size})")
        self.log_history.append(f"📢 [ROUND {server_round}] Strategi '{self.strategy.upper()}' ({epochs} epoch, lr={self.current_lr})")

        # Sesuaikan Loss Function & Optimizer sesuai perintah Server
        if self.strategy == "fedbb":
            pos_counts = self.train_set.labels.sum(dim=0).to(self.device)
            total_train = len(self.train_set.labels)
            neg_counts = total_train - pos_counts
            pos_weight = torch.clamp(neg_counts / (pos_counts + 1e-5), min=0.1, max=50.0)
            self.criterion = PNBLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.set_parameters(parameters)
        if self.strategy in ("fedsol", "fedprox", "scaffold"):
            self.global_model.load_state_dict(self.model.state_dict())
            if self.strategy == "fedsol":
                mu = float(config.get("fedsol_mu", 0.01))
                if not hasattr(self, "fedsol_opt"):
                    self.fedsol_opt = FedSOLOptimizer(self.model, self.global_model, lr=self.current_lr, mu=mu)
                else:
                    self.fedsol_opt.mu = mu
                    for param_group in self.fedsol_opt.base_optimizer.param_groups:
                        param_group["lr"] = self.current_lr

        # Setup SCAFFOLD control variates dictionary & hitung Option I c_plus
        c_plus = None
        if self.strategy == "scaffold":
            all_keys = list(self.model.state_dict().keys())
            if not getattr(self, "client_c_dict", None):
                self.client_c_dict = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
            if not getattr(self, "server_c", None):
                self.server_c = [torch.zeros_like(v) for v in self.model.state_dict().values()]
            c_s_dict = {k: c for k, c in zip(all_keys, self.server_c)}
            c_i_dict = self.client_c_dict
            # SCAFFOLD Option I: hitung gradien pada model global sebelum epoch loop
            c_plus = self._grad_at_global()

        self.model.train()
        total_loss = 0.0
        total_batches = len(self.train_loader)
        loss_acc = torch.zeros((), device=self.device)

        for epoch in range(epochs):
            epoch_acc = torch.zeros((), device=self.device)
            for batch_idx, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)

                if self.strategy == "fedsol":
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                    self.fedsol_opt.step(loss)
                else:
                    self.optimizer.zero_grad()
                    logits = self.model(x)
                    loss = self.criterion(logits, y)

                    # Penalti proksimal untuk FedProx: (mu/2) * ||w - w^t||^2
                    if self.strategy == "fedprox":
                        prox_mu = float(config.get("proximal_mu", 0.01))
                        prox_loss = torch.zeros((), device=self.device)
                        for p, p_g in zip(self.model.parameters(), self.global_model.parameters()):
                            prox_loss = prox_loss + ((p - p_g.detach()) ** 2).sum()
                        loss = loss + (prox_mu / 2.0) * prox_loss

                    loss.backward()

                    # Koreksi gradien untuk SCAFFOLD: g_i <- g_i - c_i + c
                    if self.strategy == "scaffold":
                        with torch.no_grad():
                            for name, p in self.model.named_parameters():
                                if p.grad is not None and name in c_s_dict and name in c_i_dict:
                                    p.grad.add_(c_s_dict[name] - c_i_dict[name])

                    self.optimizer.step()

                # Akumulasi di device (hindari sinkronisasi .item() tiap batch)
                loss_acc += loss.detach()
                epoch_acc += loss.detach()

                # Cetak progress setiap 25 batch agar terlihat aktif di terminal
                if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches:
                    avg_b_loss = epoch_acc.item() / (batch_idx + 1)
                    pct = ((batch_idx + 1) / total_batches) * 100
                    elapsed_so_far = round(time.time() - t_round_start, 1)
                    self.current_epoch = epoch + 1
                    self.current_batch = batch_idx + 1
                    self.current_detail = f"Ronde {server_round} (Ep {epoch+1}/{epochs} - {pct:.0f}% | {elapsed_so_far}s)"
                    msg = f"   [Epoch {epoch+1}/{epochs}] Batch {batch_idx+1}/{total_batches} ({pct:.1f}%) | Batch Loss: {loss.item():.4f} (Avg: {avg_b_loss:.4f}) | Waktu: {elapsed_so_far}s"
                    self.log_history.append(msg)
                    print(msg, flush=True)

        # Update control variates SCAFFOLD: Delta c_i = c_i^+ - c_i (Option I)
        if self.strategy == "scaffold" and c_plus is not None:
            new_c_dict = {}
            delta_c_list = []
            with torch.no_grad():
                for name, v in self.model.state_dict().items():
                    c_i = c_i_dict[name]
                    if name in c_plus:  # hanya parameter teroptimasi, bukan buffer BN
                        new_c_dict[name] = c_plus[name]
                        delta_c_list.append(c_plus[name] - c_i)
                    else:
                        new_c_dict[name] = c_i
                        delta_c_list.append(torch.zeros_like(v))
            self.client_c_dict = new_c_dict
            self.delta_c = delta_c_list

        num_examples = len(self.train_loader.dataset)
        total_loss = loss_acc.item()
        avg_loss = total_loss / (epochs * len(self.train_loader))
        round_duration = round(time.time() - t_round_start, 2)
        self.last_round_duration = round_duration
        self.current_status = "online"
        self.current_detail = f"Selesai Ronde {server_round} ({round_duration}s) - Standby"
        print(f"⏱️ [ROUND {server_round}] Pelatihan lokal selesai dalam {round_duration} detik ({round_duration/60:.2f} menit). Siap kirim ke server.", flush=True)

        metrics = {
            "loss": float(avg_loss),
            "client_id": self.client_id,
            "dataset": self.dataset_name,
            "strategy": self.strategy,
            "balance_score": float(getattr(self, "balance_score", 1.0)),
            "round_duration": float(round_duration)
        }

        params = self.get_parameters(config={})
        ckks = get_ckks_client()
        if ckks is not None and ckks.is_available():
            flat = np.concatenate([p.astype(np.float64).ravel() for p in params])
            chunks_bytes, enc_metrics = ckks.encrypt(flat)
            metrics.update(enc_metrics)
            metrics["ckks_encrypted"] = True
            metrics["ckks_num_params"] = int(len(flat))
            chunk_sizes = [len(c) for c in chunks_bytes]
            metrics["ckks_chunk_sizes"] = json.dumps(chunk_sizes)
            metrics["ckks_shapes"] = json.dumps(
                [(list(p.shape), str(p.dtype)) for p in params]
            )
            padded = []
            for c in chunks_bytes:
                pad = (8 - len(c) % 8) % 8
                padded.append(np.frombuffer(c + b'\x00' * pad, dtype=np.float64).copy())
            print(f"🔐 [CKKS] Encrypted {len(flat)} params -> {len(padded)} chunks "
                  f"({enc_metrics.get('ckks_encrypt_ms', 0):.0f}ms, "
                  f"{enc_metrics.get('ckks_bytes_total', 0)/1e6:.1f}MB)", flush=True)
            return padded, num_examples, metrics

        return params, num_examples, metrics

    def evaluate(self, parameters, config):
        """Evaluasi model global pada data VALIDASI lokal (tidak dipakai training).
        Loss evaluasi selalu BCE standar agar sebanding antar strategi (FedBB melatih dengan PNB)."""
        from src.preprocessing.dataset import DIAGNOSTIC_CLASSES_27
        from src.metrics import multilabel_metrics

        self.current_status = "online"
        self.current_detail = "Evaluasi Model Global"
        self.set_parameters(parameters)
        self.model.eval()
        eval_criterion = nn.BCEWithLogitsLoss(reduction="sum")
        total_loss, n_elems = 0.0, 0
        all_prob, all_true = [], []
        with torch.inference_mode():
            for x, y in self.eval_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                total_loss += eval_criterion(logits, y).item()
                n_elems += y.numel()
                all_prob.append(torch.sigmoid(logits).cpu())
                all_true.append(y.cpu())

        avg_loss = total_loss / max(1, n_elems)
        m = multilabel_metrics(torch.cat(all_true).numpy(), torch.cat(all_prob).numpy(), DIAGNOSTIC_CLASSES_27)
        m.update({"val_loss": float(avg_loss), "dataset": self.dataset_name, "client_id": self.client_id})

        msg = f"📊 [EVAL ronde {config.get('current_round', self.current_round)}] val_loss={avg_loss:.4f} | F1-macro={m['f1_macro']:.4f} | AUROC={m['auroc']:.4f}"
        self.log_history.append(msg)
        print(msg, flush=True)
        self.current_status = "online"
        self.current_detail = "Online (Standby)"
        return float(avg_loss), len(self.eval_loader.dataset), m


def main():
    set_seed(int(os.environ.get("SPLIT_SEED", 42)))

    parser = argparse.ArgumentParser(description="Flower ECG FL Client")
    parser.add_argument("--server", type=str, default=os.environ.get("SERVER_ADDRESS", "10.34.100.114:8080"))
    parser.add_argument("--client_id", type=str, default=os.environ.get("CLIENT_ID", "1"))
    parser.add_argument("--dataset", type=str, default=os.environ.get("DATASET_NAME", "ptbxl"))
    parser.add_argument("--strategy", type=str, default=os.environ.get("STRATEGY", "auto"), help="Strategi FL (otomatis dikontrol oleh server jika 'auto')")
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", 32)))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("LOCAL_EPOCHS", 3)))
    args = parser.parse_args()

    print(f"\n=======================================================", flush=True)
    print(f"🏥 Starting ECG Client: {args.dataset.upper()} (ID: {args.client_id})", flush=True)
    print(f"🔗 Target Server: {args.server} | Strategy: {args.strategy.upper()}", flush=True)
    print(f"=======================================================\n", flush=True)

    client = ECGFlowerClient(
        client_id=args.client_id,
        dataset_name=args.dataset,
        strategy=args.strategy,
        batch_size=args.batch_size,
        epochs=args.epochs
    )

    # Mulai thread heartbeat background ke Dashboard Web UI (port 8501)
    heartbeat = HeartbeatThread(client, args.server)
    heartbeat.start()

    # Auto-reconnect loop: Klien tetap standby jika server menjalankan eksperimen berurutan (FedBN -> FedBB -> FedSOL)
    import time
    while True:
        try:
            print("⏳ Menghubungkan ke Flower Server...", flush=True)
            fl.client.start_numpy_client(server_address=args.server, client=client)
            client.reset_state()
            print("🏁 Sesi eksperimen selesai. Model lokal di-reset ke 'auto'. Standby menunggu ronde/strategi berikutnya...", flush=True)
            time.sleep(3)
        except KeyboardInterrupt:
            print("\n🛑 Klien dihentikan oleh pengguna.")
            break
        except Exception as e:
            client.reset_state()
            print(f"⚠️ Terputus dari server ({e}). Mencoba menyambung kembali dalam 5 detik...", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main()
