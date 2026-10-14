#!/usr/bin/env python3
"""
PhysioNet ECG Dataset Downloader for Federated Learning Clients
Supports:
  - VM 2: PTB-XL (PhysioNet v1.0.3, ~3.1 GB zip)
  - VM 3: Chapman-Shaoxing (Large ECG Arrhythmia Database, ~1.2 GB tar.gz)
  - VM 4: G12EC (CinC Challenge 2020 Georgia 12-Lead ECG, ~1.5 GB tar.gz)
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm

DATA_URLS = {
    "ptbxl": {
        "name": "PTB-XL (PhysioNet v1.0.3)",
        "url": "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip",
        "archive_name": "ptbxl.zip",
        "target_dir": "data/ptbxl"
    },
    "chapman": {
        "name": "Chapman-Shaoxing Arrhythmia Database",
        "url": "https://physionet.org/static/published-projects/ecg-arrhythmia/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip",
        "archive_name": "chapman.zip",
        "target_dir": "data/chapman"
    },
    "g12ec": {
        "name": "G12EC (CinC Challenge 2020 Georgia)",
        "url": "https://cloudstor.aarnet.edu.au/plus/s/5xkmmRPWNU7pZan/download", # PhysioNet Challenge mirror
        "archive_name": "g12ec.tar.gz",
        "target_dir": "data/g12ec"
    }
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str, desc: str):
    print(f"📥 Mengunduh {desc}...")
    print(f"   URL: {url}")
    print(f"   Tujuan: {output_path}")

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    print("✅ Unduhan selesai!")


def extract_archive(archive_path: str, target_dir: str):
    print(f"📦 Mengekstrak berkas ke {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)

    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
    elif archive_path.endswith(('.tar.gz', '.tgz', '.tar')):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(target_dir)
    print("✅ Ekstraksi selesai!")


def main():
    parser = argparse.ArgumentParser(description="Download ECG Datasets from PhysioNet")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ptbxl", "chapman", "g12ec", "all"],
        help="Pilih dataset: ptbxl (VM 2), chapman (VM 3), g12ec (VM 4), atau all"
    )
    args = parser.parse_args()

    targets = list(DATA_URLS.keys()) if args.dataset == "all" else [args.dataset]

    os.makedirs("data", exist_ok=True)

    for ds_key in targets:
        ds_info = DATA_URLS[ds_key]
        archive_path = os.path.join("data", ds_info["archive_name"])
        target_dir = ds_info["target_dir"]

        if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
            print(f"⚠️ Direktori {target_dir} sudah memiliki data. Lewati unduhan.")
            continue

        try:
            download_file(ds_info["url"], archive_path, ds_info["name"])
            extract_archive(archive_path, target_dir)
            # Hapus archive untuk menghemat disk space
            if os.path.exists(archive_path):
                os.remove(archive_path)
        except Exception as e:
            print(f"❌ Gagal mengunduh {ds_info['name']}: {e}")
            print("💡 Anda dapat mendownload manual via wget atau rsync ke folder data/")


if __name__ == "__main__":
    main()
