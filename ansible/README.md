# 🚀 Panduan Orkestrasi Ansible: Benchmark Federated Learning EKG Multi-Dataset

Direktori ini berisi seluruh kumpulan playbook, inventarisasi host, dan variabel konfigurasi Ansible untuk mengotomasi deployment dan eksekusi eksperimen Federated Learning (FL) pada 4 Virtual Machine (VM) Ubuntu 24.04 (Noble Numbat).

---

## 🗺️ Topologi Cluster & Alokasi Node

| Node / Host | IP Address | Peran / Kontainer | Dataset / Fungsi | Karakteristik Heterogenitas |
| :--- | :--- | :--- | :--- | :--- |
| **VM 1** | `10.34.100.114` | `fl-server` (Port 8080)<br>`fl-dashboard` (Port 8501) | Flower Central Server & Streamlit Dashboard | Agregasi model global & visualisasi real-time |
| **VM 2** | `10.34.100.122` | `fl-client-ptbxl` | PTB-XL (~21.799 rekaman) | Feature shift & skema diagnosis SCP-ECG |
| **VM 3** | `10.34.100.123` | `fl-client-chapman` | Chapman-Shaoxing (~44.180 rekaman berlabel) | Label distribution skew & sinyal perangkat GE MUSE |
| **VM 4** | `10.34.100.125` | `fl-client-g12ec` | G12EC CinC 2020 (~10.344 rekaman) | SNOMED-CT labels & durasi sinyal bervariasi |

---

## 📂 Struktur File Ansible

```text
ansible/
├── ansible.cfg          # Konfigurasi SSH, privilege escalation, timeout
├── inventory.ini        # Definisi IP VM 1, 2, 3, 4 dan Google Drive File ID dataset
├── group_vars/
│   └── all.yml          # Variabel global (port, strategi, hyperparameter FL)
├── setup_docker.yml     # Instalasi Docker Engine & Compose plugin (Ubuntu 24.04)
├── distribute_data.yml  # Unduh & ekstrak dataset EKG via Google Drive (gdown)
├── deploy_fl.yml        # Build & initial deployment container (Server, Web UI, Clients)
├── run_experiment.yml   # Eksekusi eksperimen per-strategi (FedBN / FedBB / FedSOL)
├── manage_fl.yml        # Utilitas monitoring status, stop container, & fetch results
└── README.md            # Dokumentasi panduan eksekusi
```

---

## ⚙️ Persiapan Awal (Prasyarat)

1. **SSH Key Access:** Pastikan laptop/mesin controller memiliki akses SSH key-based (tanpa password) ke semua VM:
   ```bash
   ssh-copy-id ubuntu@10.34.100.114
   ssh-copy-id ubuntu@10.34.100.122
   ssh-copy-id ubuntu@10.34.100.123
   ssh-copy-id ubuntu@10.34.100.125
   ```

2. **Verifikasi Konektivitas Ansible:**
   ```bash
   cd ansible
   ansible -i inventory.ini fl_cluster -m ping
   ```

3. **Koleksi Ansible Tambahan (Docker & Posix):**
   ```bash
   ansible-galaxy collection install community.docker ansible.posix
   ```

---

## 🛠️ Alur Eksekusi Sesuai Tahapan Blueprint

### 1. Tahap Instalasi Docker Engine (Satu kali di awal)
Pasang Docker Engine dan Docker Compose plugin resmi untuk Ubuntu 24.04 Noble di ke-4 VM:
```bash
ansible-playbook -i inventory.ini setup_docker.yml
```

### 2. Tahap Build & Initial Deployment
Menyinkronkan kode, membangun docker image, dan menyiapkan container Flower Server, Dashboard, dan 3 Client:
```bash
# Sinkronkan kode dari direktori lokal ke semua VM dan siapkan container:
ansible-playbook -i inventory.ini deploy_fl.yml
```
> **Catatan:** Jika Anda ingin menggunakan Git repository langsung di VM, set `use_git_clone: true` di [group_vars/all.yml](file:///Users/stelatanuwijaya/Documents/Skripsi/ecg-fl-project/ansible/group_vars/all.yml).

---

### 3. Tahap Eksekusi Eksperimen Komparatif (Tahap 5)
Jalankan masing-masing strategi secara bergantian sebanyak 50 round komunikasi:

#### 🧪 Run 1: FedBN (Local Batch Normalization)
Menguji penanganan feature heterogeneity (perbedaan kalibrasi sinyal alat rekam Schiller vs GE MUSE). Parameter `BatchNorm1d` disimpan secara lokal di masing-masing client dan tidak diikutsertakan dalam agregasi server:
```bash
ansible-playbook -i inventory.ini run_experiment.yml -e "strategy=fedbn rounds=50"
```

#### 🧪 Run 2: FedBB (Balanced Loss & Client Balanced Reweighting)
Menguji penanganan class & client imbalance menggunakan Positive Negative Balanced (PNB) Loss di sisi client dan Client Balanced Reweighting (CBR) di server:
```bash
ansible-playbook -i inventory.ini run_experiment.yml -e "strategy=fedbb rounds=50"
```

#### 🧪 Run 3: FedSOL (Orthogonal Learning with Proximal Restriction)
Menguji penanganan client drift dengan memproyeksikan gradien lokal ortogonal terhadap gradien penalti proksimal:
```bash
ansible-playbook -i inventory.ini run_experiment.yml -e "strategy=fedsol rounds=50"
```

---

### 4. Tahap Monitoring & Visualisasi (Tahap 6)
Setelah server dan dashboard berjalan, buka browser pada:
👉 **[http://10.34.100.114:8501](http://10.34.100.114:8501)**

Dashboard Streamlit menyediakan visualisasi:
- Kurva konvergensi Training & Validation Loss per round.
- Perbandingan performa Macro-F1, Macro-AUROC, dan Per-Class Recall (terutama aritmia kelas minoritas).
- Komparasi waktu komputasi lokal dan efisiensi muatan parameter (MB).
- Fitur interaktif inferensi sinyal EKG 12-lead (10 detik @ 500 Hz = 5000 sampel).

---

## 🔧 Manajemen & Pemeliharaan Cluster

### Cek Status Container di Seluruh Node:
```bash
ansible-playbook -i inventory.ini manage_fl.yml -e "action=status"
```

### Hentikan Semua Sesi FL:
```bash
ansible-playbook -i inventory.ini manage_fl.yml -e "action=stop"
```

### Unduh Checkpoint Model & Log Eksperimen ke Laptop Lokal:
```bash
ansible-playbook -i inventory.ini manage_fl.yml -e "action=fetch_results"
```
Hasil akan tersimpan secara otomatis di direktori `results/checkpoints/` dan `results/logs/` di laptop dev Anda.
