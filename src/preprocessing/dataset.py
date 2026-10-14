"""
PyTorch Dataset Loader for 12-Lead ECG Datasets (PTB-XL, Chapman-Shaoxing, G12EC)
Integrates WFDB record reading, ECGPreprocessor (Filtering & 5000 Samples Standardization),
and 27-Class Multi-Label Harmonization.
"""

import os
import glob
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


class ECGDataset(Dataset):
    def __init__(self, data_dir: str, dataset_name: str, max_samples: int = None):
        self.data_dir = data_dir
        self.dataset_name = dataset_name.lower()
        self.preprocessor = ECGPreprocessor(target_rate=500, target_duration=10)
        self.classes = DIAGNOSTIC_CLASSES_27
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.record_files = self._discover_records()
        if max_samples and max_samples < len(self.record_files):
            self.record_files = self.record_files[:max_samples]

    def _discover_records(self):
        """Mencari semua rekaman .hea di folder dataset"""
        if not os.path.exists(self.data_dir):
            return []
        
        # Cari file .hea
        records = glob.glob(os.path.join(self.data_dir, "**", "*.hea"), recursive=True)
        # Hapus ekstensi .hea untuk pembacaan via wfdb
        base_records = [os.path.splitext(r)[0] for r in records]
        return base_records

    def __len__(self):
        return len(self.record_files)

    def __getitem__(self, idx):
        record_path = self.record_files[idx]
        try:
            # Baca sinyal mentah via wfdb
            record = wfdb.rdrecord(record_path)
            # Sinyal mentah shape: (samples, leads) -> transpose ke (leads, samples)
            signal_data = record.p_signal.T

            # Jika jumlah lead kurang dari 12, pad channel
            if signal_data.shape[0] < 12:
                padded_leads = np.zeros((12, signal_data.shape[1]), dtype=np.float32)
                padded_leads[:signal_data.shape[0], :] = signal_data
                signal_data = padded_leads
            elif signal_data.shape[0] > 12:
                signal_data = signal_data[:12, :]

            # Praproses sinyal (Filtering 0.05-150Hz + Notch 50Hz + Length to 5000 + Z-score)
            processed_signal = self.preprocessor.process_record(signal_data)

            # Ekstraksi label diagnosis dari header
            label_vec = np.zeros(len(self.classes), dtype=np.float32)
            if hasattr(record, "comments"):
                for comment in record.comments:
                    for cls_name, cls_idx in self.class_to_idx.items():
                        if cls_name in comment:
                            label_vec[cls_idx] = 1.0

            return torch.tensor(processed_signal, dtype=torch.float32), torch.tensor(label_vec, dtype=torch.float32)

        except Exception as e:
            # Fallback jika terjadi corrupt pada satu file
            dummy_signal = np.random.randn(12, 5000).astype(np.float32)
            dummy_label = np.zeros(len(self.classes), dtype=np.float32)
            return torch.tensor(dummy_signal), torch.tensor(dummy_label)
