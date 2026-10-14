#!/usr/bin/env python3
"""
Resumable ECG Dataset Downloader (Zero-Dependency & Zero Fabrication)
Features:
  - HTTP Range Resume from exact last byte
  - Dual Progress Visualization: Interactive CLI Progress Bar + Real-time File Logging (logs/download_<dataset>.log)
  - Zero external packages required (Pure standard library)
"""

import os
import sys
import time
import argparse
import urllib.request
import zipfile
import tarfile
from datetime import datetime

# Optional beauty progress bar, fallback to native text bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

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


class ProgressLogger:
    def __init__(self, dataset_key: str, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.dataset_key = dataset_key
        self.log_file = os.path.join(log_dir, f"download_{dataset_key}.log")
        self.last_log_time = 0.0
        self.start_time = time.time()
        self.last_bytes = 0

    def log(self, message: str, stdout: bool = False):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{self.dataset_key.upper()}] {message}\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(formatted)
        if stdout:
            print(formatted.strip())

    def update_progress(self, current: int, total: int, force: bool = False):
        now = time.time()
        # Log to file every 3 seconds to avoid massive log files
        if force or (now - self.last_log_time >= 3.0):
            elapsed_interval = max(now - self.last_log_time, 0.001)
            bytes_interval = current - self.last_bytes
            speed_mb = (bytes_interval / (1024 * 1024)) / elapsed_interval if self.last_log_time > 0 else 0.0

            cur_mb = current / (1024 * 1024)
            if total and total > 0:
                pct = (current / total) * 100
                tot_mb = total / (1024 * 1024)
                msg = f"Progress: {pct:5.1f}% | {cur_mb:7.1f} / {tot_mb:7.1f} MB | Speed: {speed_mb:5.1f} MB/s | Status: DOWNLOADING"
            else:
                msg = f"Downloaded: {cur_mb:7.1f} MB | Speed: {speed_mb:5.1f} MB/s | Status: DOWNLOADING"

            self.log(msg)
            self.last_log_time = now
            self.last_bytes = current


def print_progress_bar(current: int, total: int, prefix=""):
    """Progress bar terminal bawaan."""
    if total and total > 0:
        percent = float(current) / total
        bar_len = 25
        filled_len = int(round(bar_len * percent))
        bar = "█" * filled_len + "-" * (bar_len - filled_len)
        cur_mb = current / (1024 * 1024)
        tot_mb = total / (1024 * 1024)
        sys.stdout.write(f"\r{prefix} |{bar}| {percent*100:.1f}% ({cur_mb:.1f}/{tot_mb:.1f} MB)")
    else:
        cur_mb = current / (1024 * 1024)
        sys.stdout.write(f"\r{prefix} {cur_mb:.1f} MB...")
    sys.stdout.flush()


def download_resumable(url: str, dest_path: str, desc: str, logger: ProgressLogger):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    part_path = dest_path + ".part"

    existing_size = os.path.getsize(part_path) if os.path.exists(part_path) else 0

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (CardioPulse-FL-Downloader)")

    if existing_size > 0:
        req.add_header("Range", f"bytes={existing_size}-")
        msg = f"Resume detected! Resuming from byte {existing_size} ({existing_size / (1024*1024):.2f} MB)"
        print(f"🔄 {msg}")
        logger.log(msg)

    try:
        response = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            logger.log("Archive already fully downloaded (HTTP 416).", stdout=True)
            if os.path.exists(part_path):
                os.rename(part_path, dest_path)
            return
        logger.log(f"HTTP Error {e.code}: {e.reason}", stdout=True)
        raise e

    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        total_size = int(content_length) + existing_size
    else:
        total_size = None

    is_resume = (existing_size > 0 and response.status in (206, 200))
    mode = "ab" if is_resume else "wb"
    downloaded = existing_size if is_resume else 0

    chunk_size = 1024 * 1024  # 1 MB chunk

    logger.log(f"Download started -> Target Size: {total_size / (1024*1024) if total_size else 'Unknown':.1f} MB", stdout=True)

    if HAS_TQDM:
        with open(part_path, mode) as f, tqdm(
            total=total_size,
            initial=downloaded,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc
        ) as bar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                bar.update(len(chunk))
                logger.update_progress(downloaded, total_size)
    else:
        with open(part_path, mode) as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                print_progress_bar(downloaded, total_size, prefix=desc)
                logger.update_progress(downloaded, total_size)
        print()

    logger.update_progress(downloaded, total_size, force=True)

    # Rename file .part ke file akhir setelah selesai
    os.rename(part_path, dest_path)
    logger.log(f"Download finished successfully -> Saved to: {dest_path}", stdout=True)


def extract_archive(archive_path: str, target_dir: str, logger: ProgressLogger):
    logger.log(f"Starting extraction to: {target_dir}...", stdout=True)
    os.makedirs(target_dir, exist_ok=True)

    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            members = z.namelist()
            total_files = len(members)
            for idx, member in enumerate(members):
                z.extract(member, target_dir)
                if idx % 500 == 0 or idx == total_files - 1:
                    sys.stdout.write(f"\rEkstraksi: {idx + 1}/{total_files} file...")
                    sys.stdout.flush()
            print()
    elif archive_path.endswith((".tar.gz", ".tar")):
        with tarfile.open(archive_path, "r:*") as t:
            t.extractall(target_dir)

    logger.log(f"Extraction completed! Dataset ready at '{target_dir}'.", stdout=True)


def main():
    parser = argparse.ArgumentParser(description="Resumable ECG Dataset Downloader with Progress Logging")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ptbxl", "chapman", "g12ec"],
        help="Pilih dataset: ptbxl (VM 2), chapman (VM 3), g12ec (VM 4)"
    )
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    logger = ProgressLogger(dataset_key=args.dataset)

    logger.log("=" * 60)
    logger.log(f"Target Dataset: {ds['name']}")
    logger.log(f"Source URL: {ds['url']}")
    logger.log(f"Log destination: {logger.log_file}")
    logger.log("=" * 60)

    # Cek apakah dataset sudah ada dan berisi rekaman
    if os.path.exists(ds["target_dir"]):
        hea_files = [f for root, _, files in os.walk(ds["target_dir"]) for f in files if f.endswith(".hea")]
        if len(hea_files) > 0:
            logger.log(f"Dataset already installed ({len(hea_files)} .hea records found in '{ds['target_dir']}'). Skipping download.", stdout=True)
            return

    # Unduh file jika arsip belum ada
    if not os.path.exists(ds["archive"]):
        download_resumable(ds["url"], ds["archive"], ds["name"], logger)
    else:
        logger.log(f"Archive file '{ds['archive']}' already exists. Skipping download.", stdout=True)

    # Ekstrak arsip
    extract_archive(ds["archive"], ds["target_dir"], logger)


if __name__ == "__main__":
    main()
