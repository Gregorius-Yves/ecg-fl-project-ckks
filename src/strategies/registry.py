"""
Registry strategi FL (factory server).

Menambah metode baru cukup:
  1. Buat kelas strategi server di src/strategies/<nama>.py
  2. Tambah metadata di src/strategy_info.py dan factory di FACTORIES di bawah
  3. (Jika perlu loss/optimizer khusus di client) tambahkan cabang di src/client.py
Dashboard, tabel benchmark, grafik dan penyimpanan hasil membaca metadata yang sama,
dan run lama TIDAK pernah dihapus.
"""

from typing import Callable, Dict
import os

from src.strategy_info import STRATEGY_INFO
from .fedavg import FedAvgStrategy
from .fedprox import FedProxStrategy
from .scaffold import SCAFFOLDStrategy
from .fedbn import FedBNStrategy
from .fedbb import FedBBStrategy
from .fedsol import FedSOLStrategy

FACTORIES: Dict[str, Callable] = {
    "fedavg": lambda **kw: FedAvgStrategy(**kw),
    "fedprox": lambda **kw: FedProxStrategy(
        proximal_mu=float(kw.get("proximal_mu") if "proximal_mu" in kw else os.environ.get("PROXIMAL_MU", 0.01)),
        **{k: v for k, v in kw.items() if k != "proximal_mu"}
    ),
    "scaffold": lambda **kw: SCAFFOLDStrategy(
        total_clients=int(kw.get("total_clients") if "total_clients" in kw else os.environ.get("MIN_FIT_CLIENTS", 3)),
        **{k: v for k, v in kw.items() if k != "total_clients"}
    ),
    "fedbn": lambda **kw: FedBNStrategy(**kw),
    "fedbb": lambda **kw: FedBBStrategy(
        alpha=float(kw.get("alpha") if "alpha" in kw else os.environ.get("FEDBB_ALPHA", 0.5)),
        **{k: v for k, v in kw.items() if k != "alpha"}
    ),
    "fedsol": lambda **kw: FedSOLStrategy(
        mu=float(kw.get("fedsol_mu") if "fedsol_mu" in kw else os.environ.get("FEDSOL_MU", 0.01)),
        **{k: v for k, v in kw.items() if k != "fedsol_mu"}
    ),
}

assert set(FACTORIES) == set(STRATEGY_INFO), "STRATEGY_INFO dan FACTORIES harus sinkron"


def strategy_names():
    return list(STRATEGY_INFO.keys())


def is_registered(name: str) -> bool:
    return name.lower() in FACTORIES


def build_strategy(name: str, **common_args):
    key = name.lower()
    if key not in FACTORIES:
        raise ValueError(
            f"Strategi '{name}' tidak terdaftar. Pilihan: {', '.join(FACTORIES)}. "
            f"(Tidak ada fallback diam-diam agar hasil tidak salah label.)"
        )
    return FACTORIES[key](**common_args)
