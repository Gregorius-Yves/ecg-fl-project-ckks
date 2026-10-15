#!/usr/bin/env python3
"""
Offline Standalone ECG Preprocessing Script
Membersihkan seluruh sinyal EKG secara terpisah sebelum training dimulai:
1. Filtering: Bandpass filter 0.5 - 45 Hz (Butterworth orde 3)
2. Resampling: Downsample ke 100 Hz (10 detik = tepat 1000 sampel per lead)
3. Normalisasi: Z-score normalization per lead
4. Harmonization: 27 Multi-Label Arrhythmia Classes

Hasil pembersihan disimpan ke file tensor persisten:
  data/<dataset>_preprocessed.pt
Sehingga saat eksperimen dimulai atau di-restart, tidak ada waktu yang terbuang
untuk melakukan pra-pemrosesan ulang (0 detik loading).
"""

import os
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
import wfdb

# Tambahkan root repo ke Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.preprocessing.filter import ECGPreprocessor
from src.preprocessing.dataset import DIAGNOSTIC_CLASSES_27, parse_dx_labels, build_ptbxl_label_map

DATASETS = {
    "ptbxl": "data/ptbxl",
    "chapman": "data/chapman",
    "g12ec": "data/g12ec"
}


def process_single_record(record_path: str, preprocessor, class_to_idx, label_override=None):
    """Memproses 1 rekaman EKG mentah menjadi tensor bersih (12, 1000) dan label (27,)"""
    try:
        signal_data = None
        comments = []

        # Coba via wfdb
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

        # Coba via .mat (standard CinC 2020 / Chapman / G12EC)
        if signal_data is None:
            mat_path = record_path + ".mat"
            if os.path.exists(mat_path):
                from scipy.io import loadmat
                mat_dict = loadmat(mat_path)
                if "val" in mat_dict:
                    signal_data = mat_dict["val"].astype(np.float32)

            hea_path = record_path + ".hea"
            if os.path.exists(hea_path):
                with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
                    comments = [line.strip() for line in f if line.startswith("#")]

        if signal_data is None:
            return None, None

        # Clean NaN/Inf
        signal_data = np.nan_to_num(signal_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Ensure (12, N)
        if signal_data.ndim == 1:
            signal_data = signal_data.reshape(1, -1)
        if signal_data.shape[0] < 12:
            padded = np.zeros((12, signal_data.shape[1]), dtype=np.float32)
            padded[:signal_data.shape[0], :] = signal_data
            signal_data = padded
        elif signal_data.shape[0] > 12:
            signal_data = signal_data[:12, :]

        record_fs = 500
        if "record" in locals() and hasattr(record, "fs") and record.fs:
            record_fs = int(record.fs)

        # 3-Stage Pipeline: Bandpass 0.5-45Hz -> Resample 100Hz (1000 samples) -> Z-score
        processed_signal = preprocessor.process_record(signal_data, fs=record_fs)

        # Label vector (27 classes): PTB-XL dari CSV (override), lainnya dari baris '#Dx:' (kode SNOMED)
        if label_override is not None:
            label_vec = np.array(label_override, dtype=np.float32)
        else:
            label_vec = parse_dx_labels(comments, list(class_to_idx.keys()))

        return processed_signal, label_vec

    except Exception:
        return None, None


def discover_records(data_dir: str):
    """Mencari semua record .hea valid dalam direktori"""
    import glob
    records = glob.glob(os.path.join(data_dir, "**", "*.hea"), recursive=True)
    valid_records = []
    for r in records:
        base = os.path.basename(r)
        if not base.startswith("._") and not base.startswith("."):
            valid_records.append(os.path.splitext(r)[0])
    return sorted(valid_records)


def preprocess_dataset(dataset_name: str, data_dir: str, output_file: str, num_workers: int = None):
    print(f"\n==================================================================")
    print(f"🧹 Memulai Pra-pemrosesan Terpisah: {dataset_name.upper()}")
    print(f"📂 Folder Input Utama : {data_dir}")
    print(f"💾 File Output        : {output_file}")
    print(f"==================================================================")

    # Deteksi otomatis berbagai kemungkinan path direktori data EKG
    candidate_dirs = [
        data_dir,
        os.path.join(os.getcwd(), "data", dataset_name),
        os.path.join(os.path.expanduser("~"), "ecg-fl-project", "data", dataset_name),
        f"/home/jes/ecg-fl-project/data/{dataset_name}",
        f"data/{dataset_name}",
        f"/data/{dataset_name}",
    ]
    
    actual_data_dir = None
    records = []
    for c in candidate_dirs:
        if c and os.path.exists(c):
            found = discover_records(c)
            if len(found) > 0:
                actual_data_dir = c
                records = found
                print(f"🎯 Ditemukan {len(records)} rekaman EKG asli di: '{actual_data_dir}'")
                break

    if not actual_data_dir or len(records) == 0:
        print(f"❌ Tidak ditemukan file rekaman (.hea) di '{data_dir}' maupun di kandidat lokasi: {candidate_dirs}")
        return False

    # PTB-XL: label dari ptbxl_database.csv; pakai 500 Hz saja agar tidak duplikat dengan 100 Hz
    label_map = {}
    if dataset_name == "ptbxl":
        label_map = build_ptbxl_label_map(actual_data_dir)
        if not label_map:
            print("❌ ptbxl_database.csv tidak ditemukan; label PTB-XL tidak dapat dibuat.")
            return False
        hr = [r for r in records if f"records500" in r]
        if hr:
            records = hr
        # Buang rekaman tanpa padanan kelas 27
        records = [r for r in records if label_map.get(os.path.normpath(r)) is not None
                   and label_map[os.path.normpath(r)].sum() > 0]

    total_records = len(records)
    print(f"🔍 Memproses {total_records} rekaman EKG secara paralel...")

    preprocessor = ECGPreprocessor(
        orig_rate=500,
        target_rate=100,
        target_duration=10,
        lowcut=0.5,
        highcut=45.0,
        norm_method="zscore"
    )
    class_to_idx = {cls: idx for idx, cls in enumerate(DIAGNOSTIC_CLASSES_27)}

    num_workers = num_workers or min(os.cpu_count() or 4, 8)
    print(f"⚡ Menggunakan {num_workers} worker CPU paralel.")

    t_start = time.time()
    clean_signals = []
    clean_labels = []

    # Eksekusi batch paralel menggunakan ProcessPoolExecutor
    processed_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_record, r, preprocessor, class_to_idx,
                                   label_map.get(os.path.normpath(r))): r for r in records}

        for future in as_completed(futures):
            processed_count += 1
            sig, lbl = future.result()
            if sig is not None and lbl is not None and lbl.sum() > 0:
                clean_signals.append(sig)
                clean_labels.append(lbl)

            if processed_count % 500 == 0 or processed_count == total_records:
                pct = (processed_count / total_records) * 100
                elapsed = time.time() - t_start
                rate = processed_count / max(1e-5, elapsed)
                eta = (total_records - processed_count) / max(1e-5, rate)
                print(f"   [{dataset_name.upper()}] Progress: {processed_count}/{total_records} ({pct:.1f}%) | "
                      f"Kecepatan: {rate:.1f} rec/dtk | Sisa waktu: {eta:.0f}s", flush=True)

    elapsed_total = time.time() - t_start
    valid_count = len(clean_signals)
    print(f"\n✅ Pembersihan selesai dalam {elapsed_total:.1f} detik ({elapsed_total/60:.2f} menit).")
    print(f"   Rekaman valid berhasil diproses: {valid_count} dari {total_records}")

    if valid_count == 0:
        print("❌ Tidak ada rekaman dengan label valid. Periksa format header (#Dx) / CSV label.")
        return False

    # Konversi ke PyTorch Tensor
    print(f"📦 Mengompilasi menjadi Tensor PyTorch (N, 12, 1000)...")
    signals_tensor = torch.tensor(np.array(clean_signals), dtype=torch.float32)
    labels_tensor = torch.tensor(np.array(clean_labels), dtype=torch.float32)
    print("🏷️  Distribusi label per kelas:")
    for i, c in enumerate(DIAGNOSTIC_CLASSES_27):
        print(f"   {c:7s}: {int(labels_tensor[:, i].sum())}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "signals": signals_tensor,
        "labels": labels_tensor,
        "classes": DIAGNOSTIC_CLASSES_27,
        "created_at": time.time(),
        "shape": list(signals_tensor.shape)
    }

    print(f"💾 Menyimpan tensor bersih ke disk: {output_file} ...")
    torch.save(payload, output_file)
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"🎉 SUKSES! File tersimpan: {output_file} ({file_size_mb:.1f} MB)")

    # Pastikan file tensor juga tersedia di ~/ecg-fl-project/data/
    fallback_output = os.path.expanduser(f"~/ecg-fl-project/data/{dataset_name}_preprocessed.pt")
    if os.path.abspath(output_file) != os.path.abspath(fallback_output):
        try:
            os.makedirs(os.path.dirname(fallback_output), exist_ok=True)
            if os.path.exists(fallback_output) or os.path.islink(fallback_output):
                os.remove(fallback_output)
            os.symlink(output_file, fallback_output)
        except Exception:
            pass

    print(f"   Kini saat training dimulai / di-restart, data langsung dimuat dalam 1 detik tanpa preprocessing ulang!\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Standalone ECG Preprocessing Pipeline (Offline Generator)")
    parser.add_argument("--dataset", type=str, default="all", choices=["ptbxl", "chapman", "g12ec", "all"],
                        help="Dataset yang akan dipra-proses (ptbxl, chapman, g12ec, atau all)")
    parser.add_argument("--data_dir", type=str, default=None, help="Lokasi folder data (opsional)")
    parser.add_argument("--output", type=str, default=None, help="File output .pt (opsional)")
    parser.add_argument("--workers", type=int, default=None, help="Jumlah worker CPU paralel")
    args = parser.parse_args()

    target_datasets = ["ptbxl", "chapman", "g12ec"] if args.dataset == "all" else [args.dataset]

    for ds in target_datasets:
        data_dir = args.data_dir or DATASETS.get(ds, f"data/{ds}")
        output_file = args.output or f"data/{ds}_preprocessed.pt"
        ok = preprocess_dataset(ds, data_dir, output_file, num_workers=args.workers)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
