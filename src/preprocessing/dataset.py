"""
PyTorch Dataset Loader for 12-Lead ECG Datasets (PTB-XL, Chapman-Shaoxing, G12EC)
Integrates WFDB record reading, ECGPreprocessor (Filtering & 5000 Samples Standardization),
and 27-Class Multi-Label Harmonization.
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb

from src.preprocessing.filter import ECGPreprocessor

# 27 Target Multi-Label Arrhythmia Diagnostic Classes (PhysioNet/CinC 2020 Standard)
DIAGNOSTIC_CLASSES_27 = [
    "IAVB", "AF", "AFL", "Brady", "CRBBB", "IRBBB", "LAnFB", "LAD", "LBBB",
    "LQRSV", "NSIVCB", "PR", "PAC", "PVC", "LPR", "LQT", "QAb", "RAD",
    "RBBB", "SA", "SB", "NSR", "STach", "SVPB", "TAb", "TInv", "VPB"
]

# SNOMED-CT code -> kelas (standar PhysioNet/CinC 2020; dipakai Chapman & G12EC pada baris '#Dx:')
SNOMED_TO_CLASS = {
    "270492004": "IAVB", "164889003": "AF", "164890007": "AFL", "426627000": "Brady",
    "713427006": "CRBBB", "713426002": "IRBBB", "445211001": "LAnFB", "39732003": "LAD",
    "164909002": "LBBB", "251146004": "LQRSV", "698252002": "NSIVCB", "10370003": "PR",
    "284470004": "PAC", "427172004": "PVC", "164947007": "LPR", "111975006": "LQT",
    "164917005": "QAb", "47665007": "RAD", "59118001": "RBBB", "427393009": "SA",
    "426177001": "SB", "426783006": "NSR", "427084000": "STach", "63593006": "SVPB",
    "164934002": "TAb", "59931005": "TInv", "164884008": "VPB",
}

# PTB-XL SCP code -> kelas 27 (pemetaan aproksimasi untuk kode yang punya padanan)
SCP_TO_CLASS = {
    "NORM": "NSR", "AFIB": "AF", "AFLT": "AFL", "SBRAD": "SB", "STACH": "STach",
    "SARRH": "SA", "CRBBB": "CRBBB", "IRBBB": "IRBBB", "CLBBB": "LBBB", "ILBBB": "LBBB",
    "LAFB": "LAnFB", "1AVB": "IAVB", "PVC": "PVC", "PAC": "PAC", "LNGQT": "LQT",
    "INVT": "TInv", "ABQRS": "QAb", "NDT": "TAb", "NST_": "TAb", "LOWT": "TAb",
    "IVCD": "NSIVCB",
}


def parse_dx_labels(comments, classes=None):
    """Ekstrak vektor label 27-kelas dari komentar header WFDB (baris 'Dx: kode,kode')."""
    classes = classes or DIAGNOSTIC_CLASSES_27
    idx = {c: i for i, c in enumerate(classes)}
    vec = np.zeros(len(classes), dtype=np.float32)
    for comment in comments or []:
        line = comment.strip().lstrip("#").strip()
        if not line.lower().startswith("dx"):
            continue
        _, _, rhs = line.partition(":")
        for tok in rhs.replace(";", ",").split(","):
            tok = tok.strip()
            cls = SNOMED_TO_CLASS.get(tok) or (tok if tok in idx else None)
            if cls is not None:
                vec[idx[cls]] = 1.0
    return vec


def build_ptbxl_label_map(ptbxl_dir):
    """Baca ptbxl_database.csv -> {path rekaman tanpa ekstensi: vektor label 27 kelas}."""
    import ast
    csv_path = None
    for root, _, files in os.walk(ptbxl_dir):
        if "ptbxl_database.csv" in files:
            csv_path = os.path.join(root, "ptbxl_database.csv")
            break
    if csv_path is None:
        return {}
    base = os.path.dirname(csv_path)
    idx = {c: i for i, c in enumerate(DIAGNOSTIC_CLASSES_27)}
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        vec = np.zeros(len(DIAGNOSTIC_CLASSES_27), dtype=np.float32)
        try:
            codes = ast.literal_eval(row["scp_codes"])
        except Exception:
            codes = {}
        for code in codes:
            cls = SCP_TO_CLASS.get(code)
            if cls:
                vec[idx[cls]] = 1.0
        for col in ("filename_hr", "filename_lr"):
            if col in df.columns and isinstance(row[col], str):
                mapping[os.path.normpath(os.path.join(base, row[col]))] = vec
    return mapping


class ECGDataset(Dataset):
    def __init__(self, data_dir: str, dataset_name: str, max_samples: int = None, cache_in_ram: bool = True):
        self.data_dir = data_dir
        self.dataset_name = dataset_name.lower()
        self.cache_in_ram = cache_in_ram
        self.cache = {}
        self.preprocessor = ECGPreprocessor(
            orig_rate=500,
            target_rate=100,
            target_duration=10,
            lowcut=0.5,
            highcut=45.0,
            norm_method="zscore"
        )
        self.classes = DIAGNOSTIC_CLASSES_27
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Cek apakah ada file tensor bersih yang sudah dipra-proses offline
        preprocessed_candidates = [
            os.path.join(self.data_dir, f"{self.dataset_name}_preprocessed.pt"),
            os.path.join(self.data_dir, "preprocessed.pt"),
            os.path.join(os.getcwd(), "data", f"{self.dataset_name}_preprocessed.pt"),
            os.path.join(os.path.expanduser("~"), "ecg-fl-project", "data", f"{self.dataset_name}_preprocessed.pt"),
            f"/data/{self.dataset_name}/{self.dataset_name}_preprocessed.pt",
            f"/home/jes/ecg-fl-project/data/{self.dataset_name}_preprocessed.pt",
            os.path.join(str(Path(self.data_dir).parent), f"{self.dataset_name}_preprocessed.pt")
        ]
        self.preprocessed_file = None
        for p in preprocessed_candidates:
            if os.path.exists(p):
                self.preprocessed_file = p
                break

        if self.preprocessed_file:
            print(f"⚡ [OFFLINE PREPROCESSED] Memuat dataset bersih langsung dari file: {self.preprocessed_file}", flush=True)
            data = torch.load(self.preprocessed_file, weights_only=False)
            self.signals = data["signals"]
            self.labels = data["labels"]
            if float(self.labels.sum()) == 0.0:
                raise ValueError(
                    f"❌ Label di '{self.preprocessed_file}' SEMUA NOL (model tidak bisa belajar).\n"
                    f"   Hapus file tersebut lalu jalankan ulang: python3 preprocess_data.py --dataset {self.dataset_name}"
                )
            if max_samples and max_samples < len(self.signals):
                self.signals = self.signals[:max_samples]
                self.labels = self.labels[:max_samples]
            self.use_preprocessed = True
            print(f"✅ Berhasil memuat {len(self.signals)} rekaman pra-proses (tanpa membaca .hea/.dat saat training).", flush=True)
        else:
            # Training TIDAK membaca file mentah .hea/.dat; wajib lewat preprocess_data.py
            self.use_preprocessed = False
            self.record_files = []

    def _discover_records(self):
        """Mencari semua rekaman .hea di folder dataset (mengabaikan file macOS metadata ._*)"""
        if not os.path.exists(self.data_dir):
            return []
        
        # Cari file .hea
        records = glob.glob(os.path.join(self.data_dir, "**", "*.hea"), recursive=True)
        # Saring: abaikan file tersembunyi macOS (misal ._E06678.hea)
        valid_records = []
        for r in records:
            base_filename = os.path.basename(r)
            if not base_filename.startswith("._") and not base_filename.startswith("."):
                valid_records.append(os.path.splitext(r)[0])
        return sorted(valid_records)

    def preload_to_ram(self, num_workers: int = 0, batch_size: int = 256):
        """Baca & praproses seluruh rekaman SEKALI (paralel), simpan sebagai tensor kontigu di RAM.
        Setelah ini dataset bekerja seperti data pra-proses (tanpa I/O disk per epoch)."""
        if self.use_preprocessed:
            return
        from torch.utils.data import DataLoader
        loader = DataLoader(self, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, persistent_workers=False)
        xs, ys = [], []
        total = len(self)
        done = 0
        for x, y in loader:
            xs.append(x)
            ys.append(y)
            done += len(x)
            print(f"   📥 Preload ke RAM: {done}/{total}", flush=True)
        self.signals = torch.cat(xs).contiguous()
        self.labels = torch.cat(ys).contiguous()
        self.cache = {}
        self.use_preprocessed = True

    def __len__(self):
        if self.use_preprocessed:
            return len(self.signals)
        return len(self.record_files)

    def __getitem__(self, idx):
        if self.use_preprocessed:
            return self.signals[idx], self.labels[idx]

        if self.cache_in_ram and idx in self.cache:
            return self.cache[idx]

        # Gunakan loop iteratif (bukan rekursi) untuk mencegah RecursionError jika ada file bermasalah
        for attempt in range(min(10, len(self.record_files))):
            current_idx = (idx + attempt) % len(self.record_files)
            record_path = self.record_files[current_idx]
            try:
                signal_data = None
                comments = []

                # Cara 1: Coba via wfdb.rdrecord
                try:
                    record = wfdb.rdrecord(record_path)
                    if record.p_signal is not None:
                        signal_data = record.p_signal.T
                    elif hasattr(record, "d_signal") and record.d_signal is not None:
                        signal_data = record.d_signal.T.astype(np.float32)
                    if hasattr(record, "comments") and record.comments:
                        comments = record.comments
                except Exception:
                    pass

                # Cara 2: Jika wfdb gagal, baca langsung file .mat (format standar CinC 2020 / G12EC / Chapman)
                if signal_data is None:
                    mat_path = record_path + ".mat"
                    if os.path.exists(mat_path):
                        from scipy.io import loadmat
                        mat_dict = loadmat(mat_path)
                        if "val" in mat_dict:
                            signal_data = mat_dict["val"].astype(np.float32)

                    # Baca comments dari header .hea
                    hea_path = record_path + ".hea"
                    if os.path.exists(hea_path):
                        with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
                            comments = [line.strip() for line in f if line.startswith("#")]

                if signal_data is None:
                    continue

                # Bersihkan NaN dan Inf dari sensor
                signal_data = np.nan_to_num(signal_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                # Pastikan bentuk 2D (12, N)
                if signal_data.ndim == 1:
                    signal_data = signal_data.reshape(1, -1)
                if signal_data.shape[0] < 12:
                    padded_leads = np.zeros((12, signal_data.shape[1]), dtype=np.float32)
                    padded_leads[:signal_data.shape[0], :] = signal_data
                    signal_data = padded_leads
                elif signal_data.shape[0] > 12:
                    signal_data = signal_data[:12, :]

                # Praproses sinyal (Filtering 0.5-45Hz -> Resample 100Hz -> Z-score)
                record_fs = 500
                if "record" in locals() and hasattr(record, "fs") and record.fs:
                    record_fs = int(record.fs)

                processed_signal = self.preprocessor.process_record(signal_data, fs=record_fs)

                # Ekstraksi label diagnosis
                label_vec = np.zeros(len(self.classes), dtype=np.float32)
                for comment in comments:
                    for cls_name, cls_idx in self.class_to_idx.items():
                        if cls_name in comment:
                            label_vec[cls_idx] = 1.0

                x_tensor = torch.tensor(processed_signal, dtype=torch.float32)
                y_tensor = torch.tensor(label_vec, dtype=torch.float32)
                if self.cache_in_ram:
                    self.cache[idx] = (x_tensor, y_tensor)
                return x_tensor, y_tensor

            except Exception:
                continue

        # Fallback aman jika seluruh sampel dalam batch gagal dibaca
        return torch.zeros((12, 1000), dtype=torch.float32), torch.zeros(len(self.classes), dtype=torch.float32)
