#!/usr/bin/env python3
"""
Resumable ECG Dataset Downloader (Zero-Dependency & Zero Fabrication)
Supports HTTP Range Resume from exact last byte.
Runs directly on clean Python environments without requiring external packages.
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile

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


def print_progress_bar(current: int, total: int, prefix=""):
    """Progress bar teks bawaan Python standard library."""
    if total and total > 0:
        percent = float(current) / total
        bar_len = 30
        filled_len = int(round(bar_len * percent))
        bar = "█" * filled_len + "-" * (bar_len - filled_len)
        cur_mb = current / (1024 * 1024)
        tot_mb = total / (1024 * 1024)
        sys.stdout.write(f"\r{prefix} |{bar}| {percent*100:.1f}% ({cur_mb:.1f}/{tot_mb:.1f} MB)")
    else:
        cur_mb = current / (1024 * 1024)
        sys.stdout.write(f"\r{prefix} {cur_mb:.1f} MB terunduh...")
    sys.stdout.flush()


def download_resumable(url: str, dest_path: str, desc: str):
    """
    Mengunduh file dengan dukungan HTTP Range (Resume download jika terputus).
    Menggunakan urllib bawaan Python agar tidak butuh dependensi tambahan.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    part_path = dest_path + ".part"

    existing_size = os.path.getsize(part_path) if os.path.exists(part_path) else 0

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (CardioPulse-FL-Downloader)")

    if existing_size > 0:
        req.add_header("Range", f"bytes={existing_size}-")
        print(f"🔄 Menemukan unduhan parsial ({existing_size / (1024*1024):.2f} MB).")
        print(f"⏩ Melanjutkan dari byte terakhir...")

    try:
        response = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            # Range not satisfiable (file sudah lengkap)
            print("✅ Berkas sudah lengkap.")
            os.rename(part_path, dest_path)
            return
        raise e

    # Hitung total ukuran
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        total_size = int(content_length) + existing_size
    else:
        total_size = None

    is_resume = (existing_size > 0 and response.status in (206, 200))
    mode = "ab" if is_resume else "wb"
    downloaded = existing_size if is_resume else 0

    chunk_size = 1024 * 1024  # 1 MB chunk

    print(f"📥 Memulai pengunduhan {desc}...")

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
                bar.update(len(chunk))
    else:
        with open(part_path, mode) as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                print_progress_bar(downloaded, total_size, prefix=desc)
        print()

    # Rename file .part ke file akhir setelah selesai
    os.rename(part_path, dest_path)
    print(f"✅ Unduhan selesai dan tersimpan di: {dest_path}")


def extract_archive(archive_path: str, target_dir: str):
    """Mengekstrak arsip zip/tar ke direktori target."""
    print(f"📦 Mengekstrak berkas ke: {target_dir}...")
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

    print(f"🎉 Ekstraksi selesai! Data asli siap digunakan di '{target_dir}'.")


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
    print(f"\n=======================================================")
    print(f"📥 Target Dataset: {ds['name']}")
    print(f"   URL: {ds['url']}")
    print(f"=======================================================")

    # Cek apakah dataset sudah ada dan berisi rekaman
    if os.path.exists(ds["target_dir"]):
        hea_files = [f for root, _, files in os.walk(ds["target_dir"]) for f in files if f.endswith(".hea")]
        if len(hea_files) > 0:
            print(f"✅ Dataset asli sudah terpasang ({len(hea_files)} rekaman .hea ditemukan di {ds['target_dir']}).")
            print("Tidak perlu mengunduh ulang.")
            return

    # Unduh file jika arsip belum ada
    if not os.path.exists(ds["archive"]):
        download_resumable(ds["url"], ds["archive"], ds["name"])
    else:
        print(f"📁 Arsip {ds['archive']} sudah ada. Melewati proses unduh.")

    # Ekstrak arsip
    extract_archive(ds["archive"], ds["target_dir"])


if __name__ == "__main__":
    main()
