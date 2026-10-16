# 🐳 Panduan Orkestrasi Docker via Ansible: Benchmark Federated Learning EKG

Seluruh siklus hidup cluster Federated Learning (FL) 12-lead EKG diorkestrasi secara **100% berbasis Docker** pada 4 Virtual Machine (VM).  
Semua dependensi Python, PyTorch, dan Flower Server terisolasi di dalam Docker container `ecg-fl:latest`.

---

## 🗺️ Topologi Cluster & Container Docker

| Node / Host | IP Address | Container Docker | Volume Mount & Fungsi |
| :--- | :--- | :--- | :--- |
| **VM 1 (Server)** | `10.34.100.114` | `fl-dashboard` (Port 8501 & 8080) | Mount: `results/`, `logs/`<br>Orkestrasi Flower Server & Web UI |
| **VM 2 (Client 1)** | `10.34.100.122` | `fl-client-ptbxl` | Mount: `/data/ptbxl`<br>19,532 sampel (Feature Shift) |
| **VM 3 (Client 2)** | `10.34.100.123` | `fl-client-chapman` | Mount: `/data/chapman`<br>44,180 sampel (Label Skew) |
| **VM 4 (Client 3)** | `10.34.100.125` | `fl-client-g12ec` | Mount: `/data/g12ec`<br>9,355 sampel (Duration Skew) |

---

## 📂 Daftar Playbook Ansible

```text
ansible/
├── inventory.ini        # IP Address VM 1, 2, 3, 4 & spesifikasi container
├── group_vars/
│   └── all.yml          # Hyperparameter & konfigurasi Docker image (ecg-fl:latest)
├── setup_docker.yml     # Instalasi Docker Engine & Docker Compose di seluruh VM
├── distribute_data.yml  # Download & ekstrak dataset EKG mentah ke folder /data
├── preprocess_data.yml  # Pra-pemrosesan sinyal EKG di dalam Docker container terisolasi
├── deploy_fl.yml        # Build image Docker & start cluster (Server + Dashboard + Clients)
├── start_clients.yml    # Start seluruh container Docker client (fl-client-*)
├── stop_clients.yml     # Stop & remove container Docker client
├── run_experiment.yml   # Eksekusi eksperimen per-metode dengan parameter modular
├── manage_fl.yml        # Operasi container (status, logs, restart, stop, fetch_results)
└── README.md            # Panduan eksekusi ini
```

---

## 🛠️ Alur Eksekusi Eksperimen Berbasis Docker

### 1. Deployment Awal (Build Image & Start Cluster)
Membangun image `ecg-fl:latest` di semua VM dan menyalakan container `fl-dashboard` (Server) serta 3 container client:
```bash
cd ansible
ansible-playbook -i inventory.ini deploy_fl.yml
```

### 2. Pra-Pemrosesan Data EKG di dalam Docker (Opsional / Sekali Saja)
Menjalankan pembersihan sinyal mentah ke file tensor `.pt` langsung di dalam Docker container:
```bash
ansible-playbook -i inventory.ini preprocess_data.yml
```

### 3. Mengontrol Container Client
```bash
# Menyalakan 3 container Docker client:
ansible-playbook -i inventory.ini start_clients.yml

# Menghentikan 3 container Docker client:
ansible-playbook -i inventory.ini stop_clients.yml
```

---

## 🧪 Menjalankan Eksperimen FL per-Metode

Setiap eksperimen menghasilkan folder run baru di `results/runs/<metode>-<timestamp>/` (append-only, tidak pernah menimpa hasil lama).

```bash
# 1. FedAvg (Baseline Standard)
ansible-playbook -i inventory.ini run_experiment.yml -e "strategy=fedavg rounds=50"

# 2. FedProx (dengan opsi modular proximal_mu)
ansible-playbook -i inventory.ini run_experiment.yml -e "strategy=fedprox rounds=50 proximal_mu=0.01"

# 3. SCAFFOLD (Control Variates Option 2)
ansible-playbook -i inventory.ini run_experiment.yml -e "strategy=scaffold rounds=50"

# 4. FedBN (Local BatchNorm Isolation)
ansible-playbook -i inventory.ini run_experiment.yml -e "strategy=fedbn rounds=50"

# 5. FedBB (dengan opsi modular fedbb_alpha)
ansible-playbook -i inventory.ini run_experiment.yml -e "strategy=fedbb rounds=50 fedbb_alpha=0.5"

# 6. FedSOL (dengan opsi modular fedsol_mu)
ansible-playbook -i inventory.ini run_experiment.yml -e "strategy=fedsol rounds=50 fedsol_mu=0.01"
```

---

## 📊 Monitoring & Operasi Docker Cluster

### Memantau Lewat Web Dashboard:
Buka di browser: **`http://10.34.100.114:8501`**  
*(Menyediakan kontrol GUI, grafik konvergensi, inspeksi riwayat run, dan download model PyTorch `best.pt`)*.

### Cek Status Semua Container Docker:
```bash
ansible-playbook -i inventory.ini manage_fl.yml -e "action=status"
```

### Melihat Log Real-time Container:
```bash
ansible-playbook -i inventory.ini manage_fl.yml -e "action=logs"
```

### Restart Container Tertentu:
```bash
# Restart container dashboard server:
ansible-playbook -i inventory.ini manage_fl.yml -e "action=restart_dashboard"

# Restart container seluruh klien:
ansible-playbook -i inventory.ini manage_fl.yml -e "action=restart_clients"
```

### Hentikan Seluruh Container Cluster:
```bash
ansible-playbook -i inventory.ini manage_fl.yml -e "action=stop"
```

### Unduh Hasil Eksperimen (`results/runs`) ke Laptop Dev:
```bash
ansible-playbook -i inventory.ini manage_fl.yml -e "action=fetch_results"
```
Hasil evaluasi (`metrics.json`, `best.pt`, `final.pt`) akan ditarik ke direktori lokal `results/runs/` tanpa menimpa data yang sudah ada.
