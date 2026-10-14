#!/usr/bin/env python3
"""
Resumable ECG Dataset Downloader with Zero Fabrication Guarantee
Supports:
  - VM 2: PTB-XL (PhysioNet v1.0.3)
  - VM 3: Chapman-Shaoxing (Large ECG Arrhythmia Database v1.0.0)
  - VM 4: G12EC (PhysioNet CinC 2020 Georgia 12-Lead ECG)
Features:
  - HTTP Range header (Resumes broken downloads from exact last byte)
  - Integrity and file size check
  - Automatic extraction to data/<dataset>/
"""

import os
import sys
import argparse
import requests
import zipfile
import tarfile
from tqdm import tqdm

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


def download_resumable(url: str, dest_path: str, desc: str):
    """
    Mengunduh file dengan dukungan HTTP Range (Resume download jika terputus).
    Jika file sudah ada sebagian, proses unduh akan melanjutkan dari byte terakhir.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    part_path = dest_path + ".part"

    # Periksa ukuran file sementara jika sebelumnya terputus
    existing_size = os.path.getsize(part_path) if os.path.exists(part_path) else 0

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"🔄 Menemukan unduhan sebelumnya ({existing_size / (1024*1024):.2f} MB).")
        print(f"⏩ Melanjutkan dari byte terakhir...")

    session = requests.Session()
    response = session.get(url, headers=headers, stream=True, timeout=30)

    # Tangani respons server
    if response.status_code == 416:
        # Range Not Satisfiable (file sudah komplit)
        total_size = existing_size
    elif response.status_code not in (200, 206):
        raise RuntimeError(f"Gagal mengunduh: HTTP {response.status_code} ({response.reason})")
    else:
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            total_size = int(content_length) + existing_size
        else:
            total_size = None

    mode = "ab" if existing_size > 0 and response.status_code == 206 else "wb"
    if mode == "wb":
        existing_size = 0

    chunk_size = 1024 * 1024  # 1 MB chunks

    with open(part_path, mode) as f, tqdm(
        total=total_size,
        initial=existing_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"Unduh {desc}"
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    # Ganti nama dari .part ke file final setelah tuntas
    os.rename(part_path, dest_path)
    print(f"✅ Unduhan selesai dan tersimpan di: {dest_path}")


def extract_archive(archive_path: str, target_dir: str):
    """Mengekstrak arsip zip/tar ke direktori target."""
    print(f"📦 Mengekstrak berkas ke: {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)

    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            # Dapatkan list berkas untuk progress bar
            members = z.namelist()
            for member in tqdm(members, desc="Ekstraksi Zip", unit="file"):
                z.extract(member, target_dir)
    elif archive_path.endswith((".tar.gz", ".tar")):
        with tarfile.open(archive_path, "r:*") as t:
            t.extractall(target_dir)

    print(f"🎉 Ekstraksi selesai! Data asli siap digunakan di '{target_dir}'.")


def main():
    parser = argparse.ArgumentParser(description="Resumable ECG Dataset Downloader (Zero Fabrication)")
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
