"""
Penyimpanan hasil eksperimen yang PERMANEN dan append-only.

Struktur (volume Docker `results/`):
  results/runs/<strategi>-<YYYYmmdd-HHMMSS>/
      metrics.json   # meta + config + metrik per ronde (global & per-klien), ditulis tiap ronde
      best.pt        # state_dict global dengan macro-F1 validasi terbaik
      final.pt       # state_dict global ronde terakhir

Prinsip:
  * Setiap sesi training = satu folder run baru; tidak ada kode yang menimpa/menghapus run lama.
  * Menambah strategi baru (mis. FedAvg) hanya menambah folder run baru.
  * Semua angka berasal dari evaluasi model sungguhan di client (tidak ada data sintetis).
"""

import json
import math
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.metrics import clean

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", Path.cwd() / "results"))
RUNS_DIR = RESULTS_DIR / "runs"
METRICS_FILE = "metrics.json"


def _atomic_write_json(path: Path, data: Any):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _weighted_mean(values: List[Optional[float]], weights: List[float]) -> Optional[float]:
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None and w > 0]
    if not pairs:
        return None
    tw = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / tw


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


class RunRecorder:
    """Merekam satu sesi training (satu strategi) ke results/runs/<run_id>/."""

    def __init__(self, strategy: str, config: Dict[str, Any], runs_dir: Optional[Path] = None):
        self.runs_dir = Path(runs_dir) if runs_dir else RUNS_DIR
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_id, n = f"{strategy}-{stamp}", 1
        while (self.runs_dir / run_id).exists():  # jangan pernah menimpa run lama
            n += 1
            run_id = f"{strategy}-{stamp}-{n}"
        self.run_id = run_id
        self.dir = self.runs_dir / run_id
        self.dir.mkdir(parents=True)

        self.meta: Dict[str, Any] = {
            "run_id": run_id,
            "strategy": strategy,
            "config": config,
            "status": "running",
            "host": socket.gethostname(),
            "started_at": time.time(),
            "finished_at": None,
            "models": {},
            "rounds": [],
        }
        self._rounds: Dict[int, Dict[str, Any]] = {}
        self._best_f1 = -1.0
        self._flush()

    # ---------- penulisan ----------
    def _flush(self):
        self.meta["rounds"] = [self._rounds[k] for k in sorted(self._rounds)]
        _atomic_write_json(self.dir / METRICS_FILE, self.meta)

    def _round(self, r: int) -> Dict[str, Any]:
        return self._rounds.setdefault(r, {"round": r, "clients": {}})

    def start_round(self, r: int):
        rd = self._round(r)
        rd.setdefault("start", time.time())
        self._flush()

    def record_fit(self, r: int, clients: List[Dict[str, Any]], num_failures: int = 0):
        rd = self._round(r)
        rd["num_fit_failures"] = num_failures
        for c in clients:
            ds = c["dataset"]
            rec = rd["clients"].setdefault(ds, {})
            rec.update({
                "num_train_examples": c.get("num_examples"),
                "train_loss": clean(c.get("train_loss")),
                "train_duration": clean(c.get("duration")),
            })
        rd["train_loss"] = _weighted_mean(
            [clean(c.get("train_loss")) for c in clients], [c.get("num_examples", 0) for c in clients])
        rd["fit_end"] = time.time()
        self._flush()

    def record_eval(self, r: int, clients: List[Dict[str, Any]], num_failures: int = 0) -> Dict[str, Any]:
        rd = self._round(r)
        rd["num_eval_failures"] = num_failures
        weights = [c.get("num_examples", 0) for c in clients]
        for c in clients:
            ds = c["dataset"]
            rec = rd["clients"].setdefault(ds, {})
            rec.update({
                "num_val_examples": c.get("num_examples"),
                "val_loss": clean(c.get("val_loss")),
                "f1_macro": clean(c.get("f1_macro")),
                "f1_micro": clean(c.get("f1_micro")),
                "auroc": clean(c.get("auroc")),
                "per_class_f1": {k: clean(v) for k, v in (c.get("per_class") or {}).items() if clean(v) is not None},
            })
        g = {}
        for key in ("val_loss", "f1_macro", "f1_micro", "auroc"):
            vals = [clean(c.get(key)) for c in clients]
            g[key] = _weighted_mean(vals, weights)          # dibobot jumlah sampel validasi
            g[f"{key}_client_mean"] = _mean(vals)           # rata-rata sederhana antar klien
        rd["global"] = g
        now = time.time()
        rd["end"] = now
        if "start" in rd:
            rd["duration"] = round(now - rd["start"], 2)
        self._flush()
        return g

    def is_best(self, f1: Optional[float]) -> bool:
        if f1 is None:
            return False
        if f1 > self._best_f1:
            self._best_f1 = f1
            return True
        return False

    def save_model(self, tag: str, state_dict: Dict[str, Any], round_num: int, f1: Optional[float], **extra):
        import torch
        tmp = self.dir / f"{tag}.pt.tmp"
        torch.save(state_dict, tmp)
        os.replace(tmp, self.dir / f"{tag}.pt")
        self.meta["models"][tag] = {"file": f"{tag}.pt", "round": round_num, "f1_macro": f1, **extra}
        self._flush()

    def finish(self, status: str):
        self.meta["status"] = status
        self.meta["finished_at"] = time.time()
        self._flush()


# ---------- pembacaan (dipakai Dashboard) ----------
def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path / METRICS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_runs(runs_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    base = Path(runs_dir) if runs_dir else RUNS_DIR
    if not base.exists():
        return []
    runs = [m for m in (_load(p) for p in base.iterdir() if p.is_dir()) if m]
    return sorted(runs, key=lambda m: m.get("started_at", 0))


def load_run(run_id: str, runs_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    base = Path(runs_dir) if runs_dir else RUNS_DIR
    if "/" in run_id or ".." in run_id:
        return None
    return _load(base / run_id)


def latest_run(strategy: str, runs_dir: Optional[Path] = None, require_rounds: bool = True) -> Optional[Dict[str, Any]]:
    cands = [r for r in list_runs(runs_dir) if r.get("strategy") == strategy]
    if require_rounds:
        cands = [r for r in cands if any("global" in rd for rd in r.get("rounds", []))]
    return cands[-1] if cands else None


def model_path(run_id: str, tag: str = "best", runs_dir: Optional[Path] = None) -> Optional[Path]:
    base = Path(runs_dir) if runs_dir else RUNS_DIR
    if "/" in run_id or ".." in run_id or tag not in ("best", "final"):
        return None
    p = base / run_id / f"{tag}.pt"
    return p if p.exists() else None


def mark_stale_running(runs_dir: Optional[Path] = None):
    """Run berstatus 'running' tetapi proses server sudah mati -> tandai 'interrupted' (data TETAP disimpan)."""
    base = Path(runs_dir) if runs_dir else RUNS_DIR
    if not base.exists():
        return
    for p in base.iterdir():
        m = _load(p) if p.is_dir() else None
        if m and m.get("status") == "running":
            m["status"] = "interrupted"
            m["finished_at"] = m.get("finished_at") or time.time()
            _atomic_write_json(p / METRICS_FILE, m)
