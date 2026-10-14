#!/usr/bin/env python3
"""
High-Speed Resumable ECG Dataset Downloader (Multi-Connection Accelerated)
Supports:
  - Multi-stream parallel acceleration (via axel / aria2c if available, 10x-50x speed boost)
  - Native Python HTTP Range Resume fallback
  - Automatic extraction to data/<dataset>/
"""

import os
import sys
import shutil
import subprocess
import argparse
import urllib.request
import zipfile
import tarfile
import time
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
    if bytes_num >= 1024 * 1024 * 1024:
        return f"{bytes_num / (1024 * 1024 * 1024):.2f} GB"
    return f"{bytes_num / (1024 * 1024):.1f} MB"


def try_accelerated_download(url: str, dest_path: str) -> bool:
    """
    Mencoba mengunduh menggunakan multi-stream accelerator (axel atau aria2c)
    yang bisa menembus throttling PhysioNet hingga puluhan MB/s.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # 1. Coba axel (16 koneksi paralel)
    if shutil.which("axel"):
        print("⚡ Terdeteksi 'axel'! Mengunduh dengan 16 koneksi paralel simultan...", flush=True)
        cmd = ["axel", "-n", "16", "-a", "-o", dest_path, url]
        ret = subprocess.call(cmd)
        return ret == 0

    # 2. Coba aria2c (16 koneksi paralel)
    if shutil.which("aria2c"):
        print("⚡ Terdeteksi 'aria2c'! Mengunduh dengan 16 koneksi paralel simultan...", flush=True)
        cmd = [
            "aria2c",
            "-x", "16",
            "-s", "16",
            "-k", "1M",
            "-c",
            "--dir", os.path.dirname(dest_path),
            "--out", os.path.basename(dest_path),
            url
        ]
        ret = subprocess.call(cmd)
        return ret == 0

    return False


def download_native_resumable(url: str, dest_path: str, desc: str):
    """Fallback unduh bawaan Python dengan HTTP Range Resume."""
    part_path = dest_path + ".part"
    existing_size = os.path.getsize(part_path) if os.path.exists(part_path) else 0

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

    if existing_size > 0:
        req.add_header("Range", f"bytes={existing_size}-")
        print(f"🔄 Melanjutkan dari byte sebelumnya: {format_bytes(existing_size)}...", flush=True)

    try:
        response = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            print("✅ File sudah terunduh komplit.", flush=True)
            if os.path.exists(part_path):
                os.rename(part_path, dest_path)
            return
        raise e

    content_length = response.headers.get("Content-Length")
    total_size = int(content_length) + existing_size if content_length is not None else None

    is_resume = (existing_size > 0 and response.status in (206, 200))
    mode = "ab" if is_resume else "wb"
    downloaded = existing_size if is_resume else 0

    print(f"\n📥 Mengunduh [{desc}] -> Target: {format_bytes(total_size) if total_size else 'Unknown'}", flush=True)
    print("💡 Tips: Jalankan 'sudo apt install -y axel' agar unduhan dipercepat hingga 16x lipat!", flush=True)

    chunk_size = 256 * 1024
    start_time = time.time()
    last_print = start_time
    last_bytes = downloaded

    with open(part_path, mode) as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            elapsed = now - last_print
            if elapsed >= 3.0:
                speed_mb = ((downloaded - last_bytes) / (1024 * 1024)) / elapsed
                ts = datetime.now().strftime("%H:%M:%S")
                if total_size and total_size > 0:
                    pct = (downloaded / total_size) * 100
                    rem_mb = (total_size - downloaded) / (1024 * 1024)
                    eta_sec = int(rem_mb / speed_mb) if speed_mb > 0 else 0
                    print(f"[{ts}] 📥 {pct:5.1f}% | {format_bytes(downloaded)} / {format_bytes(total_size)} | {speed_mb:5.1f} MB/s | ETA: {eta_sec // 60}m {eta_sec % 60}s", flush=True)
                else:
                    print(f"[{ts}] 📥 {format_bytes(downloaded)} | {speed_mb:5.1f} MB/s", flush=True)
                last_print = now
                last_bytes = downloaded

    if os.path.exists(part_path):
        os.rename(part_path, dest_path)
    print(f"✅ Unduhan selesai dan tersimpan di: {dest_path}", flush=True)


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
    parser = argparse.ArgumentParser(description="Accelerated Resumable ECG Dataset Downloader")
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

    # Cek apakah sudah diekstrak
    if os.path.exists(ds["target_dir"]):
        hea_files = [f for root, _, files in os.walk(ds["target_dir"]) for f in files if f.endswith(".hea")]
        if len(hea_files) > 0:
            print(f"✅ Dataset asli sudah terpasang ({len(hea_files):,} rekaman .hea di '{ds['target_dir']}').", flush=True)
            print("Tidak perlu mengunduh ulang.", flush=True)
            return

    # Unduh file jika arsip belum ada
    if not os.path.exists(ds["archive"]):
        # 1. Coba multi-stream accelerator (axel/aria2c) jika terpasang
        success = try_accelerated_download(ds["url"], ds["archive"])
        # 2. Jika axel/aria2c tidak ada atau gagal, fallback ke native resumable
        if not success:
            download_native_resumable(ds["url"], ds["archive"], ds["name"])
    else:
        print(f"📁 File arsip '{ds['archive']}' sudah lengkap. Langsung mengekstrak...", flush=True)

    # Ekstrak arsip
    extract_archive(ds["archive"], ds["target_dir"])


if __name__ == "__main__":
    main()
