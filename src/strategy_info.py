"""
Metadata strategi FL (ringan, TANPA import torch/flwr) agar Dashboard bisa membacanya.

Untuk menambah metode baru: tambahkan satu entri di sini + satu factory di
src/strategies/registry.py (+ cabang loss/optimizer di src/client.py bila perlu).
Urutan dict = urutan tampil di Dashboard.
"""

STRATEGY_INFO = {
    "fedavg": {
        "name": "FedAvg",
        "full_name": "Federated Averaging (Baseline)",
        "mechanism": "Rata-rata terbobot semua parameter, loss BCE standar",
        "target": "Baseline pembanding tanpa regulasi",
        "note": "Acuan standar tanpa mitigasi heterogenitas",
        "exclude_bn": False,
    },
    "fedprox": {
        "name": "FedProx",
        "full_name": "Federated Optimization with Proximal Term",
        "mechanism": "Penalti proksimal (mu/2 * ||w - w^t||^2) pada loss lokal",
        "target": "System & Statistical Heterogeneity (Kompensasi Client Drift)",
        "note": "Menstabilkan optimasi lokal saat data non-IID",
        "exclude_bn": False,
    },
    "scaffold": {
        "name": "SCAFFOLD",
        "full_name": "Stochastic Controlled Averaging for Federated Learning",
        "mechanism": "Control variates lokal & global (koreksi gradien g_i - c_i + c)",
        "target": "Client Drift Akut pada variasi distribusi sinyal ekstrem",
        "note": "Mengurangi variansi gradien lokal terhadap konsensus global",
        "exclude_bn": False,
    },
    "fedbn": {
        "name": "FedBN",
        "full_name": "Federated Batch Normalization",
        "mechanism": "Local BatchNorm1d (Isolasi mean/var per perangkat)",
        "target": "Feature Shift (Sensitivitas Schiller vs GE MUSE)",
        "note": "Mengatasi perbedaan noise & baseline alat rekam EKG",
        "exclude_bn": True,
    },
    "fedbb": {
        "name": "FedBB",
        "full_name": "Class-Balanced Reweighting + PNB Loss",
        "mechanism": "PNB Loss + Cluster-Balanced Weighting (CBR)",
        "target": "Label Skew (Ketimpangan Diagnosis Kelas Langka)",
        "note": "Deteksi kelas minoritas (IAVB, PAC, AFL, PR)",
        "exclude_bn": False,
    },
    "fedsol": {
        "name": "FedSOL",
        "full_name": "Subspace Orthogonal Optimization",
        "mechanism": "Orthogonal Gradient Projection terhadap Model Global",
        "target": "Client Drift & Duration Skew (Resampling 100Hz)",
        "note": "Mencegah catastrophic forgetting saat agregasi model",
        "exclude_bn": False,
    },
}
