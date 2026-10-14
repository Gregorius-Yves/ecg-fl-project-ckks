#!/usr/bin/env python3
"""
Resumable ECG Dataset Downloader (Zero-Dependency & Zero Fabrication)
Features:
  - Real-time periodic console updates (Prints progress line every 2-3 seconds with flush)
  - HTTP Range Resume from exact last byte
  - Automatic Extraction to data/<dataset>/
"""

import os
import sys
import time
import argparse
import urllib.request
import zipfile
import tarfile
from datetime import datetime

DATASETS = {
    "ptbxl": {
        "name": "PTB-XL (~21,799 ECG Records)",
        "url": "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip",
        "archive": "data/ptbxl.zip",
        "target_dir": "data/ptbxl"
    },
    "chapman": {
        "name": "Chapman-Shaoxing (~10,646 ECG Records)",
        "url": "https://physionet.org/static/published-projects/ecg-arrhythmia/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip",
        "archive": "data/chapman.zip",
        "target_dir": "data/chapman"
    },
    "g12ec": {
        "name": "G12EC Georgia 12-Lead ECG (~10,344 Records)",
        "url": "https://physionet.org/content/challenge-2020/get-zip/1.0.2/",
        "archive": "data/challenge2020_georgia.zip",
        "target_dir": "data/g12ec"
    }
}


def format_bytes(bytes_num: int) -> str:
    """Format bytes ke MB atau GB yang mudah dibaca."""
    if bytes_num >= 1024 * 1024 * 1024:
        return f"{bytes_num / (1024 * 1024 * 1024):.2f} GB"
    return f"{bytes_num / (1024 * 1024):.1f} MB"


def download_resumable(url: str, dest_path: str, desc: str, dataset_key: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/download_{dataset_key}.log"
    part_path = dest_path + ".part"

    existing_size = os.path.getsize(part_path) if os.path.exists(part_path) else 0

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) CardioPulse/1.0")

    if existing_size > 0:
        req.add_header("Range", f"bytes={existing_size}-")
        print(f"🔄 Menemukan unduhan sebelumnya: {format_bytes(existing_size)}. Melanjutkan...", flush=True)

    try:
        response = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            print("✅ File sudah terunduh komplit (HTTP 416).", flush=True)
            if os.path.exists(part_path):
                os.rename(part_path, dest_path)
            return
        raise e

    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        total_size = int(content_length) + existing_size
    else:
        total_size = None

    is_resume = (existing_size > 0 and response.status in (206, 200))
    mode = "ab" if is_resume else "wb"
    downloaded = existing_size if is_resume else 0

    print(f"\n🚀 Memulai unduh [{desc}] -> Total Target: {format_bytes(total_size) if total_size else 'Unknown'}", flush=True)
    print(f"📡 Progress akan ditampilkan setiap 3 detik di bawah ini:\n", flush=True)

    # 256 KB chunk size untuk update yang responsif
    chunk_size = 256 * 1024
    start_time = time.time()
    last_print_time = start_time
    last_print_bytes = downloaded

    with open(part_path, mode) as f, open(log_file, "a", encoding="utf-8") as flog:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            elapsed_interval = now - last_print_time

            # Update ke layar terminal setiap 3 detik sekali
            if elapsed_interval >= 3.0:
                speed_mb = ((downloaded - last_print_bytes) / (1024 * 1024)) / elapsed_interval
                timestamp = datetime.now().strftime("%H:%M:%S")

                if total_size and total_size > 0:
                    pct = (downloaded / total_size) * 100
                    rem_bytes = max(total_size - downloaded, 0)
                    eta_sec = int(rem_bytes / (speed_mb * 1024 * 1024)) if speed_mb > 0 else 0
                    eta_str = f"{eta_sec // 60}m {eta_sec % 60}s"

                    status_line = (
                        f"[{timestamp}] 📥 {pct:5.1f}% | "
                        f"{format_bytes(downloaded)} / {format_bytes(total_size)} | "
                        f"Kecepatan: {speed_mb:5.1f} MB/s | "
                        f"ETA: {eta_str} [BERJALAN]"
                    )
                else:
                    status_line = (
                        f"[{timestamp}] 📥 Terunduh: {format_bytes(downloaded)} | "
                        f"Kecepatan: {speed_mb:5.1f} MB/s [BERJALAN]"
                    )

                # Cetak baris baru secara eksplisit agar pasti tampil di terminal SSH
                print(status_line, flush=True)
                flog.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {status_line}\n")
                flog.flush()

                last_print_time = now
                last_print_bytes = downloaded

    # Final line
    final_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{final_time}] ✅ 100.0% | Unduhan tuntas ({format_bytes(downloaded)})! Memindahkan file...", flush=True)

    if os.path.exists(part_path):
        os.rename(part_path, dest_path)
    print(f"📁 Tersimpan permanen di: {dest_path}\n", flush=True)


def extract_archive(archive_path: str, target_dir: str):
    print(f"📦 Mengekstrak berkas ke: {target_dir}...", flush=True)
    os.makedirs(target_dir, exist_ok=True)

    start_time = time.time()
    last_print = start_time

    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            members = z.namelist()
            total_files = len(members)
            for idx, member in enumerate(members):
                z.extract(member, target_dir)
                now = time.time()
                if now - last_print >= 4.0 or idx == total_files - 1:
                    pct = ((idx + 1) / total_files) * 100
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📦 Ekstraksi: {idx + 1:,} / {total_files:,} file ({pct:.1f}%)...", flush=True)
                    last_print = now
    elif archive_path.endswith((".tar.gz", ".tar")):
        with tarfile.open(archive_path, "r:*") as t:
            t.extractall(target_dir)

    print(f"🎉 Selesai! Seluruh data asli terpasang di '{target_dir}'. Siap untuk training!", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Resumable ECG Dataset Downloader")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ptbxl", "chapman", "g12ec"],
        help="Pilih dataset: ptbxl (VM 2), chapman (VM 3), g12ec (VM 4)"
    )
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    print("=" * 65, flush=True)
    print(f"🏥 PhysioNet Dataset Downloader: {ds['name']}", flush=True)
    print(f"🔗 Sumber URL: {ds['url']}", flush=True)
    print("=" * 65, flush=True)

    # Cek apakah sudah diekstrak sebelumnya
    if os.path.exists(ds["target_dir"]):
        hea_files = [f for root, _, files in os.walk(ds["target_dir"]) for f in files if f.endswith(".hea")]
        if len(hea_files) > 0:
            print(f"✅ Dataset asli sudah terpasang ({len(hea_files):,} rekaman .hea di '{ds['target_dir']}').", flush=True)
            print("Tidak perlu mengunduh ulang.", flush=True)
            return

    # Unduh file jika arsip belum ada
    if not os.path.exists(ds["archive"]):
        download_resumable(ds["url"], ds["archive"], ds["name"], args.dataset)
    else:
        print(f"📁 File arsip '{ds['archive']}' sudah lengkap. Langsung mengekstrak...", flush=True)

    # Ekstrak arsip
    extract_archive(ds["archive"], ds["target_dir"])


if __name__ == "__main__":
    main()
