#!/usr/bin/env python3
"""
CardioPulse FL - Dedicated Web Dashboard & Orchestration Server
Provides:
  - Experiment Control REST API (/api/experiment/start, /api/experiment/stop)
  - Real-time Subprocess Management for Flower Server
  - Live Log Streaming (/api/logs)
  - Live Metric Tracking (/api/status, /api/metrics)
  - Best Model Weight Download (/api/download-model)
Runs on Port 8501.
"""

import os
import sys
import json
import time
import socket
import signal
import subprocess
import mimetypes
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

PORT = int(os.environ.get("PORT", os.environ.get("STREAMLIT_SERVER_PORT", 8501)))
HOST = os.environ.get("HOST", "0.0.0.0")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)

SERVER_LOG_FILE = os.path.join(LOG_DIR, "server_exec.log")

sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("RESULTS_DIR", str(BASE_DIR / "results"))
from src import results_store as rs  # noqa: E402
from src.strategy_info import STRATEGY_INFO  # noqa: E402


def _strategy_list(value):
    if isinstance(value, list):
        return [str(s).strip().lower() for s in value if str(s).strip()]
    return [s.strip().lower() for s in str(value or "").split(",") if s.strip()]


def _first_strategy(value):
    names = _strategy_list(value)
    if names and names[0] == "all":
        return next(iter(STRATEGY_INFO))
    return names[0] if names else None


def _finished_rounds(run):
    return [rd for rd in run.get("rounds", []) if "global" in rd]


def _run_view(run):
    """Ubah satu run menjadi seri per-ronde untuk grafik (semua nilai dari evaluasi nyata)."""
    rds = _finished_rounds(run)
    view = {
        "run_id": run["run_id"],
        "status": run.get("status"),
        "rounds": [rd["round"] for rd in rds],
        "loss": [rd["global"].get("val_loss") for rd in rds],
        "train_loss": [rd.get("train_loss") for rd in rds],
        "val_loss": [rd["global"].get("val_loss") for rd in rds],
        "f1_macro": [rd["global"].get("f1_macro") for rd in rds],
        "auroc": [rd["global"].get("auroc") for rd in rds],
    }
    datasets = {ds for rd in rds for ds in rd.get("clients", {})}
    datasets |= {c["dataset"] for c in KNOWN_CLIENTS.values()}
    for ds in sorted(datasets):
        cl = [rd.get("clients", {}).get(ds, {}) for rd in rds]
        view[ds] = {
            "loss": [c.get("val_loss") for c in cl],
            "train_loss": [c.get("train_loss") for c in cl],
            "f1_macro": [c.get("f1_macro") for c in cl],
            "auroc": [c.get("auroc") for c in cl],
        }
    return view


def build_metrics_history(active_strat):
    runs = rs.list_runs()
    with_rounds = [r for r in runs if _finished_rounds(r)]
    latest = {}
    for r in with_rounds:  # urut waktu -> yang terakhir menang
        latest[r["strategy"]] = r

    hist = {"rounds": [], "loss": [], "val_loss": [], "f1_macro": [], "auroc": [], "strategies": {}}
    for c in KNOWN_CLIENTS.values():
        hist[c["dataset"]] = {"loss": [], "f1_macro": [], "auroc": []}

    for sk in STRATEGY_INFO:
        if sk in latest:
            hist["strategies"][sk] = _run_view(latest[sk])
    # strategi non-registry (mis. run lama) tetap ditampilkan
    for sk, r in latest.items():
        if sk not in hist["strategies"]:
            hist["strategies"][sk] = _run_view(r)

    focus = latest.get(active_strat) or (with_rounds[-1] if with_rounds else None)
    if focus:
        v = _run_view(focus)
        hist.update({k: v[k] for k in ("rounds", "val_loss", "f1_macro", "auroc")})
        hist["loss"] = v["train_loss"]
        hist["run_id"] = focus["run_id"]
        hist["focus_strategy"] = focus["strategy"]
        for c in KNOWN_CLIENTS.values():
            hist[c["dataset"]] = v[c["dataset"]]
    return hist


def build_benchmark(is_running, active_strat):
    """Tabel benchmark = ringkasan run TERAKHIR tiap strategi (run lama tetap ada di results/runs)."""
    runs = rs.list_runs()
    out = {}
    keys = list(STRATEGY_INFO) + sorted({r["strategy"] for r in runs} - set(STRATEGY_INFO))
    for sk in keys:
        info = STRATEGY_INFO.get(sk, {"name": sk.upper(), "full_name": sk, "mechanism": "-", "target": "-", "note": ""})
        item = {k: v for k, v in info.items() if k != "exclude_bn"}
        item.update({"macro_f1": None, "auroc": None, "final_loss": None, "speed_sec": None,
                     "status": "pending", "rounds_done": 0, "run_id": None,
                     "best_f1": None, "best_round": None,
                     "num_runs": sum(1 for r in runs if r["strategy"] == sk)})
        mine = [r for r in runs if r["strategy"] == sk]
        live = [r for r in mine if r.get("status") == "running"]
        with_rounds = [r for r in mine if _finished_rounds(r)]
        run = (live[-1] if live and is_running and sk == active_strat else None) or (with_rounds[-1] if with_rounds else None)
        if run:
            rds = _finished_rounds(run)
            item["run_id"] = run["run_id"]
            item["rounds_done"] = len(rds)
            status = run.get("status", "pending")
            if status == "running" and not (is_running and sk == active_strat):
                status = "interrupted"
            item["status"] = status
            if rds:
                g = rds[-1]["global"]
                item["macro_f1"], item["auroc"], item["final_loss"] = g.get("f1_macro"), g.get("auroc"), g.get("val_loss")
                durs = [rd["duration"] for rd in rds if rd.get("duration")]
                if durs:
                    item["speed_sec"] = round(sum(durs) / len(durs), 1)
                scored = [(rd["global"].get("f1_macro"), rd["round"]) for rd in rds if rd["global"].get("f1_macro") is not None]
                if scored:
                    item["best_f1"], item["best_round"] = max(scored)
        out[sk] = item
    return out


import time
import socket

KNOWN_CLIENTS = {
    "vm2": {"id": "1", "name": "VM 2 — PTB-XL", "ip": "10.34.100.122", "dataset": "ptbxl"},
    "vm3": {"id": "2", "name": "VM 3 — Chapman-Shaoxing", "ip": "10.34.100.123", "dataset": "chapman"},
    "vm4": {"id": "3", "name": "VM 4 — G12EC (CinC 2020)", "ip": "10.34.100.125", "dataset": "g12ec"},
}

class ClientTracker:
    heartbeats = {}  # key -> dict(last_seen=float, status=str, detail=str, ...)
    client_logs = {"vm2": "", "vm3": "", "vm4": ""}

    @classmethod
    def register_heartbeat(cls, data, client_ip=None):
        cid = str(data.get("client_id", "")).strip()
        dataset = str(data.get("dataset", "")).strip().lower()
        key = None
        for k, v in KNOWN_CLIENTS.items():
            if str(v["id"]) == cid:
                key = k
                break
            if dataset and (v["dataset"] in dataset or dataset in v["dataset"]):
                key = k
                break
            if client_ip and v["ip"] == client_ip:
                key = k
                break
        if key:
            # Hanya terima log dari klien versi baru (log sesi berjalan). Klien lama mengirim isi client.log basi.
            if data.get("log_version", 1) >= 2 and "logs" in data and data["logs"]:
                cls.client_logs[key] = str(data["logs"])
            was_registered = key in cls.heartbeats and (time.time() - cls.heartbeats[key]["last_seen"] < 8.0)
            if not was_registered:
                print(f"💓 [HEARTBEAT] Klien {KNOWN_CLIENTS[key]['name']} (ID: {cid}) TERHUBUNG dari {client_ip}!", flush=True)
            cls.heartbeats[key] = {
                "last_seen": time.time(),
                "status": data.get("status", "online"),
                "detail": data.get("detail", "Online"),
                "current_round": data.get("current_round", 0),
                "epoch": data.get("epoch", 0),
                "total_epochs": data.get("total_epochs", pm.config.get("local_epochs", 3)),
                "batch": data.get("batch", 0),
                "total_batches": data.get("total_batches", 0),
                "learning_rate": data.get("learning_rate", pm.config.get("learning_rate", 0.001)),
                "measured_duration": data.get("measured_duration", 0),
                "elapsed_in_round": data.get("elapsed_in_round", 0)
            }

    @classmethod
    def get_client_log(cls, key):
        """Kembalikan log klien; tandai jika klien sudah offline (log basi)."""
        log = cls.client_logs.get(key, "")
        if not log:
            return ""
        hb = cls.heartbeats.get(key)
        if hb is None or (time.time() - hb["last_seen"]) >= 8.0:
            cls.client_logs[key] = ""  # klien offline: buang log basi
            return ""
        return log

    @classmethod
    def get_established_connections(cls):
        """Mendeteksi IP klien yang aktif terhubung via gRPC Flower Server (port 8080)"""
        active_ips = set()
        try:
            res = subprocess.run(
                ["ss", "-Htn", "sport", "=", ":8080"],
                capture_output=True,
                text=True,
                timeout=0.8
            )
            for line in res.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 4 and parts[0] == "ESTAB":
                    peer = parts[3]
                    ip = peer.rsplit(":", 1)[0].strip("[]")
                    active_ips.add(ip)
        except Exception:
            pass
        return active_ips

    @classmethod
    def get_all_status(cls, is_server_active=True):
        now = time.time()
        active_tcp_ips = cls.get_established_connections()
        result = {}
        connected_count = 0

        for key, info in KNOWN_CLIENTS.items():
            ip = info["ip"]
            hb = cls.heartbeats.get(key)
            is_recent_heartbeat = hb is not None and (now - hb["last_seen"] < 8.0)
            is_active_tcp = ip in active_tcp_ips

            if is_recent_heartbeat:
                raw_status = hb.get("status", "online")
                detail = hb.get("detail", "Online")
                if is_server_active:
                    status = "training" if raw_status == "training" else "online"
                else:
                    status = "online"
                    detail = "Online (Standby)"
                connected_count += 1
            elif is_active_tcp:
                status = "online"
                detail = "Terhubung ke FL Server"
                connected_count += 1
            else:
                status = "offline"
                detail = "Offline"

            result[key] = {
                "id": info["id"],
                "name": info["name"],
                "ip": ip,
                "dataset": info["dataset"],
                "status": status,
                "status_label": detail,
                "current_round": hb.get("current_round", 0) if hb else 0,
                "epoch": hb.get("epoch", 0) if hb else 0,
                "total_epochs": hb.get("total_epochs", pm.config.get("local_epochs", 3)) if hb else pm.config.get("local_epochs", 3),
                "batch": hb.get("batch", 0) if hb else 0,
                "total_batches": hb.get("total_batches", 0) if hb else 0,
                "learning_rate": hb.get("learning_rate", pm.config.get("learning_rate", 0.001)) if hb else pm.config.get("learning_rate", 0.001),
                "last_seen": hb["last_seen"] if hb else None,
                "measured_duration": (hb.get("measured_duration", 0) if (hb and is_server_active) else 0),
                "elapsed_in_round": (hb.get("elapsed_in_round", 0) if (hb and is_server_active) else 0)
            }

        return result, connected_count

# Global Process State Manager
class ProcessManager:
    server_process = None
    start_time = None
    round_timestamps = {}  # {round_num: timestamp}
    config = {
        "strategy": "fedbn",
        "num_rounds": 50,
        "local_epochs": 3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "model_arch": "resnet1d"
    }

pm = ProcessManager()


class CardioPulseHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def _json(self, payload, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_POST(self):
        if self.path == "/api/experiment/start":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            params = json.loads(body.decode("utf-8")) if body else {}

            pm.config.update(params)

            names = _strategy_list(pm.config.get("strategy", "fedbn"))
            bad = [s for s in names if s != "all" and s not in STRATEGY_INFO]
            if bad:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": f"Strategi tidak terdaftar: {bad}. Pilihan: {list(STRATEGY_INFO)}"}).encode("utf-8"))
                return

            # 1. Hentikan SEMUA proses Flower server sebelumnya (baik dari UI maupun terminal CLI)
            try:
                subprocess.run(["pkill", "-9", "-f", "src.server"], timeout=1)
                time.sleep(0.5)
            except Exception:
                pass
            if pm.server_process and pm.server_process.poll() is None:
                try:
                    pm.server_process.kill()
                except Exception:
                    pass

            # 2. Hapus file timing ronde dan bersihkan log
            timing_file = BASE_DIR / "logs" / "round_timing.json"
            if timing_file.exists():
                try:
                    timing_file.unlink()
                except Exception:
                    pass

            with open(SERVER_LOG_FILE, "w", encoding="utf-8") as f:
                f.write(f"[{params.get('strategy', 'fedbn').upper()}] Memulai Flower Server Orchestrator (Ronde 1 - {pm.config['num_rounds']})...\n")

            # 3. Reset data tracking pada UI
            ClientTracker.client_logs = {"vm2": "", "vm3": "", "vm4": ""}
            for k in ClientTracker.heartbeats:
                ClientTracker.heartbeats[k]["current_round"] = 0
                ClientTracker.heartbeats[k]["epoch"] = 0
                ClientTracker.heartbeats[k]["batch"] = 0
                ClientTracker.heartbeats[k]["measured_duration"] = 0
                ClientTracker.heartbeats[k]["elapsed_in_round"] = 0

            log_output = open(SERVER_LOG_FILE, "a", encoding="utf-8")

            # Deteksi otomatis Python interpreter (prioritaskan venv jika ada)
            venv_python = BASE_DIR / "venv" / "bin" / "python3"
            python_bin = str(venv_python) if venv_python.exists() else sys.executable

            # Ambil format strategi (bisa string tunggal, list array urutan, atau csv)
            strat_val = pm.config.get("strategy", "fedbn")
            if isinstance(strat_val, list):
                strat_str = ",".join(strat_val)
            else:
                strat_str = str(strat_val)

            cmd = [
                python_bin, "-m", "src.server",
                "--strategy", strat_str,
                "--rounds", str(pm.config["num_rounds"]),
                "--min_clients", "3"
            ]

            env = os.environ.copy()
            if venv_python.exists():
                env["PATH"] = f"{BASE_DIR}/venv/bin:" + env.get("PATH", "")
                env["VIRTUAL_ENV"] = str(BASE_DIR / "venv")

            env["PYTHONPATH"] = str(BASE_DIR)
            env["RESULTS_DIR"] = str(rs.RESULTS_DIR)
            env["STRATEGY"] = str(pm.config["strategy"])
            env["NUM_ROUNDS"] = str(pm.config["num_rounds"])
            env["LOCAL_EPOCHS"] = str(pm.config["local_epochs"])
            env["LEARNING_RATE"] = str(pm.config["learning_rate"])
            env["BATCH_SIZE"] = str(pm.config["batch_size"])
            env["PROXIMAL_MU"] = str(pm.config.get("proximal_mu", 0.01))
            env["FEDBB_ALPHA"] = str(pm.config.get("fedbb_alpha", 0.5))
            env["FEDSOL_MU"] = str(pm.config.get("fedsol_mu", 0.01))

            pm.start_time = time.time()
            pm.round_timestamps = {}

            # Jalankan Flower Server di background
            pm.server_process = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                env=env,
                stdout=log_output,
                stderr=subprocess.STDOUT
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started", "pid": pm.server_process.pid, "config": pm.config}).encode("utf-8"))
            return

        elif self.path == "/api/experiment/stop":
            try:
                subprocess.run(["pkill", "-9", "-f", "src.server"], timeout=1)
            except Exception:
                pass
            if pm.server_process and pm.server_process.poll() is None:
                try:
                    pm.server_process.terminate()
                    pm.server_process.wait(timeout=2)
                except Exception:
                    pm.server_process.kill()
            pm.server_process = None
            pm.start_time = None
            pm.round_timestamps = {}
            with open(SERVER_LOG_FILE, "a", encoding="utf-8") as f:
                f.write("\n[SERVER] Proses pelatihan dihentikan oleh pengguna dari Web UI.\n")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "stopped"}).encode("utf-8"))
            return

        elif self.path == "/api/experiment/reset":
            # Hentikan semua proses Flower server (baik yang via UI maupun manual CLI)
            if pm.server_process and pm.server_process.poll() is None:
                try:
                    pm.server_process.terminate()
                    pm.server_process.wait(timeout=2)
                except Exception:
                    pm.server_process.kill()
            pm.server_process = None
            pm.start_time = None
            pm.round_timestamps = {}

            try:
                subprocess.run(["pkill", "-9", "-f", "src.server"], timeout=1)
            except Exception:
                pass

            # Bersihkan file log dan timing
            try:
                with open(SERVER_LOG_FILE, "w", encoding="utf-8") as f:
                    f.write("Sesi di-reset. Siap memulai eksperimen baru.\n")
            except Exception:
                pass

            timing_file = BASE_DIR / "logs" / "round_timing.json"
            if timing_file.exists():
                try:
                    timing_file.unlink()
                except Exception:
                    pass

            # Hasil run di results/runs TIDAK pernah dihapus oleh reset (hanya state live & log).

            # Reset status dan log pada ClientTracker
            ClientTracker.client_logs = {"vm2": "", "vm3": "", "vm4": ""}
            for k in ClientTracker.heartbeats:
                ClientTracker.heartbeats[k]["status"] = "online"
                ClientTracker.heartbeats[k]["detail"] = "Online (Standby)"
                ClientTracker.heartbeats[k]["current_round"] = 0
                ClientTracker.heartbeats[k]["epoch"] = 0
                ClientTracker.heartbeats[k]["batch"] = 0
                ClientTracker.heartbeats[k]["measured_duration"] = 0
                ClientTracker.heartbeats[k]["elapsed_in_round"] = 0

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "reset", "message": "Sesi live & log di-reset. Hasil run tersimpan permanen di results/runs"}).encode("utf-8"))
            return

        elif self.path == "/api/client/heartbeat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            if body:
                try:
                    payload = json.loads(body.decode("utf-8"))
                    client_ip = self.client_address[0] if self.client_address else None
                    ClientTracker.register_heartbeat(payload, client_ip=client_ip)
                except Exception:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
            return

        self.send_error(404, "Endpoint not found")

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
            return super().do_GET()

        elif self.path == "/api/status":
            def is_flower_server_active():
                if pm.server_process and pm.server_process.poll() is None:
                    return True
                try:
                    with socket.create_connection(("127.0.0.1", 8080), timeout=0.15):
                        return True
                except Exception:
                    pass
                try:
                    res = subprocess.run(["pgrep", "-f", "(src\\.server|server\\.py|run_benchmark)"], capture_output=True, text=True, timeout=0.5)
                    if res.returncode == 0 and res.stdout.strip():
                        return True
                except Exception:
                    pass
                now = time.time()
                active_hb = any((now - hb.get("last_seen", 0) < 15.0 and hb.get("status") in ["training", "evaluating"]) for hb in ClientTracker.heartbeats.values())
                if active_hb:
                    return True
                return False

            is_running = is_flower_server_active()
            clients_status, connected_count = ClientTracker.get_all_status(is_server_active=is_running)

            # Baca round_timing.json untuk waktu nyata tiap ronde
            timing_file = BASE_DIR / "logs" / "round_timing.json"
            timing_data = {}
            completed_durations = []
            if timing_file.exists():
                try:
                    with open(timing_file, "r") as f:
                        timing_data = json.load(f)
                        for r_k, r_v in timing_data.items():
                            if isinstance(r_v, dict) and "duration" in r_v and r_v["duration"] > 0:
                                completed_durations.append(float(r_v["duration"]))
                except Exception:
                    pass

            # Ekstrak current_round dari berbagai sumber (timing_file, heartbeat klien, results_store, dan server log)
            current_round = 0
            all_strat_keys = list(STRATEGY_INFO.keys())
            completed_strats = set()
            active_strat = None
            strat_rounds = {}
            total_rounds = pm.config.get("num_rounds", 50)

            if timing_data:
                try:
                    valid_rounds = [int(k) for k in timing_data.keys() if str(k).isdigit()]
                    if valid_rounds:
                        current_round = max(current_round, max(valid_rounds))
                except Exception:
                    pass

            # Periksa sesi aktif dari results_store
            active_run = None
            try:
                runs = rs.list_runs()
                cand = ([r for r in runs if r.get("status") == "running"] or runs)[-1] if runs else None
                if cand:
                    active_run = cand
                    if cand.get("status") == "running":
                        is_running = True
                        if not active_strat:
                            active_strat = cand.get("strategy")
                        if cand.get("config", {}).get("num_rounds"):
                            total_rounds = cand["config"]["num_rounds"]
                    if cand.get("rounds"):
                        for rd in cand["rounds"]:
                            r_num = rd.get("round", 0)
                            if r_num > 0:
                                current_round = max(current_round, r_num)
                            if not completed_durations:
                                if rd.get("duration"):
                                    completed_durations.append(float(rd["duration"]))
                                elif rd.get("clients"):
                                    cdurs = [c.get("train_duration") or c.get("duration") for c in rd["clients"].values() if c.get("train_duration") or c.get("duration")]
                                    if cdurs:
                                        completed_durations.append(max(cdurs))
            except Exception:
                pass

            # Deteksi dari SERVER_LOG_FILE secara presisi untuk SEMUA strategi
            if os.path.exists(SERVER_LOG_FILE):
                try:
                    with open(SERVER_LOG_FILE, "r", encoding="utf-8", errors="ignore") as lf:
                        current_logged_strat = None
                        for line in lf:
                            line_upper = line.upper()
                            for sk in all_strat_keys:
                                if f"MEMULAI SESI FL: {sk.upper()}" in line_upper or f"STRATEGI '{sk.upper()}'" in line_upper:
                                    current_logged_strat = sk
                                    active_strat = sk
                                    if sk not in strat_rounds:
                                        strat_rounds[sk] = 0
                                if f"SELESAI SESI EKSPERIMEN: {sk.upper()}" in line_upper:
                                    completed_strats.add(sk)
                            if "Memulai Sesi FL:" in line and "(" in line and "Rounds)" in line:
                                try:
                                    r_total = int(line.split("(")[1].split()[0])
                                    total_rounds = r_total
                                except Exception:
                                    pass
                            for marker in ["[SERVER] Ronde ", "[ROUND ", "Ronde "]:
                                if marker in line:
                                    part = line.split(marker)[1].split()[0].split("|")[0].split("/")[0].strip("]:,")
                                    if part.isdigit():
                                        r_val = int(part)
                                        current_round = max(current_round, r_val)
                                        if current_logged_strat:
                                             strat_rounds[current_logged_strat] = max(strat_rounds.get(current_logged_strat, 0), r_val)
                                        break
                except Exception:
                    pass

            # Sinkronkan ronde & durasi dari heartbeat klien aktif
            client_rounds = []
            client_durations = []
            client_elapsed = 0.0
            for hb in ClientTracker.heartbeats.values():
                if (time.time() - hb.get("last_seen", 0)) < 10.0:
                    cr = hb.get("current_round", 0)
                    if cr > 0:
                        client_rounds.append(cr)
                    md = hb.get("measured_duration", 0)
                    if md > 0:
                        client_durations.append(float(md))
                    el = hb.get("elapsed_in_round", 0)
                    if el > client_elapsed:
                        client_elapsed = float(el)

            if client_rounds:
                current_round = max(current_round, max(client_rounds))

            if not completed_durations and client_durations:
                completed_durations.append(max(client_durations))

            if not active_strat:
                strat_raw = str(pm.config.get("strategy", "fedbn")).lower()
                first_s = strat_raw.split(",")[0].strip()
                if first_s in all_strat_keys:
                    active_strat = first_s

            # Metrik NYATA dari results/runs (hasil evaluasi model sungguhan di tiap klien)
            if not is_running:
                rs.mark_stale_running()
            metrics_history = build_metrics_history(active_strat)

            # Kalkulasi Estimasi Waktu Selesai (ETA) berdasarkan durasi NYATA
            elapsed_sec = int(time.time() - pm.start_time) if (is_running and pm.start_time) else int(client_elapsed)

            def format_duration(seconds: int) -> str:
                if seconds <= 0:
                    return "Selesai"
                h = seconds // 3600
                m = (seconds % 3600) // 60
                s = seconds % 60
                if h > 0:
                    return f"{h}j {m}m"
                elif m > 0:
                    return f"{m}m {s}s"
                else:
                    return f"{s}s"

            current_round_start = timing_data.get(str(current_round), {}).get("start") if timing_data else None
            if not current_round_start and active_run and active_run.get("rounds"):
                for rd in reversed(active_run["rounds"]):
                    if rd.get("round") == current_round and rd.get("start"):
                        current_round_start = rd["start"]
                        break

            remaining_rounds = max(0, total_rounds - current_round)

            if len(completed_durations) >= 1:
                # 1 ronde atau lebih sudah selesai dan terukur secara nyata
                avg_sec_per_round = sum(completed_durations) / len(completed_durations)
                elapsed_in_current = (time.time() - current_round_start) if current_round_start else client_elapsed
                elapsed_in_current = max(0.0, elapsed_in_current)
                time_left_current = max(0.0, avg_sec_per_round - elapsed_in_current)
                eta_seconds = int((max(0, remaining_rounds - 1) * avg_sec_per_round) + time_left_current)

                speed_formatted = f"{avg_sec_per_round:.1f} s/rd"
                if remaining_rounds <= 0:
                    eta_formatted = "Pelatihan Selesai"
                else:
                    eta_formatted = f"~{format_duration(eta_seconds)} ({remaining_rounds} rd tersisa)"
            elif is_running and current_round > 0:
                # Sedang mengukur waktu ronde berjalan
                avg_sec_per_round = 0.0
                eta_seconds = 0
                elapsed_now = int(client_elapsed or (time.time() - current_round_start if current_round_start else (time.time() - pm.start_time if pm.start_time else 0)))
                speed_formatted = "Mengukur..."
                eta_formatted = f"Mengukur Ronde {current_round}... ({elapsed_now}s)"
            elif is_running:
                avg_sec_per_round = 0.0
                eta_seconds = 0
                speed_formatted = "Inisialisasi..."
                eta_formatted = "Menunggu ronde 1..."
            else:
                avg_sec_per_round = 0.0
                eta_seconds = 0
                speed_formatted = "--"
                eta_formatted = "Pelatihan Selesai" if (current_round >= total_rounds and total_rounds > 0) else "Siap dimulai"

            elapsed_formatted = format_duration(elapsed_sec) if is_running else "--"

            # Benchmark dibangun dari hasil run NYATA di results/runs (bukan data sintetis)
            benchmark_data = build_benchmark(is_running, active_strat)

            status_payload = {
                "status": "running" if is_running else "idle",
                "is_running": is_running,
                "server_status": "running" if is_running else "idle",
                "current_round": current_round,
                "total_rounds": total_rounds,
                "strategy": active_strat or pm.config.get("strategy", "fedbn"),
                "clients": clients_status,
                "connected_count": connected_count,
                "connected_clients": connected_count,
                "total_clients": len(KNOWN_CLIENTS),
                "metrics": metrics_history,
                "benchmark": benchmark_data,
                "eta_seconds": eta_seconds,
                "eta_formatted": eta_formatted,
                "elapsed_seconds": elapsed_sec,
                "elapsed_formatted": elapsed_formatted,
                "speed_formatted": speed_formatted,
                "avg_round_duration": round(avg_sec_per_round, 1) if avg_sec_per_round > 0 else None,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status_payload).encode("utf-8"))
            return

        elif self.path.startswith("/api/logs"):
            import urllib.parse
            parsed_url = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            source = query_params.get("source", ["server"])[0].lower()

            server_log = "Menunggu sesi training dimulai..."
            if os.path.exists(SERVER_LOG_FILE):
                try:
                    with open(SERVER_LOG_FILE, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        server_log = "".join(lines[-80:])
                except Exception as e:
                    server_log = f"Error membaca log server: {e}"

            if source == "server":
                logs = server_log
            elif source in ClientTracker.client_logs:
                client_log = ClientTracker.get_client_log(source)
                if client_log:
                    logs = client_log
                else:
                    client_name = KNOWN_CLIENTS.get(source, {}).get("name", source.upper())
                    logs = f"Menunggu log dari {client_name}...\n(Klien belum aktif atau belum mengirim data)"
            elif source == "all":
                combined = []
                combined.append("=== [SERVER ORCHESTRATOR LOGS] ===")
                combined.append(server_log.strip())
                for ckey, cinfo in KNOWN_CLIENTS.items():
                    clog = ClientTracker.get_client_log(ckey)
                    combined.append(f"\n=== [{cinfo['name']} LOGS] ===")
                    if clog:
                        combined.append(clog.strip())
                    else:
                        combined.append("Belum ada log diterima dari klien ini.")
                logs = "\n".join(combined)
            else:
                logs = server_log

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"logs": logs, "source": source}).encode("utf-8"))
            return

        elif self.path.startswith("/api/download-model"):
            import urllib.parse
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            tag = q.get("tag", ["best"])[0]
            run_id = q.get("run", [None])[0]
            strategy = q.get("strategy", [None])[0]
            if not run_id:
                cands = [r for r in rs.list_runs()
                         if (not strategy or r.get("strategy") == strategy) and tag in r.get("models", {})]
                run_id = cands[-1]["run_id"] if cands else None
            path = rs.model_path(run_id, tag) if run_id else None
            if not path:
                self.send_error(404, "Belum ada model hasil training yang tersimpan (jalankan eksperimen sampai ada ronde selesai).")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f'attachment; filename="{run_id}_{tag}.pt"')
            self.send_header("Content-Length", str(os.path.getsize(path)))
            self.end_headers()
            with open(path, "rb") as f:
                self.wfile.write(f.read())
            return

        elif self.path.startswith("/api/download-metrics"):
            import urllib.parse
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            run_id = q.get("run", [None])[0]
            if not run_id:
                runs = rs.list_runs()
                run_id = runs[-1]["run_id"] if runs else None
            if not run_id:
                self.send_error(404, "Run ID tidak ditemukan")
                return
            run_dir = rs.RUNS_DIR / run_id
            metrics_path = run_dir / rs.METRICS_FILE
            if not metrics_path.exists():
                self.send_error(404, "File metrics.json tidak ditemukan untuk run ini")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Disposition", f'attachment; filename="{run_id}_metrics.json"')
            self.send_header("Content-Length", str(os.path.getsize(metrics_path)))
            self.end_headers()
            with open(metrics_path, "rb") as f:
                self.wfile.write(f.read())
            return

        elif self.path == "/api/metrics":
            strat = _first_strategy(pm.config.get("strategy"))
            hist = build_metrics_history(strat)
            self._json(hist)
            return

        elif self.path == "/api/strategies":
            self._json({"strategies": STRATEGY_INFO})
            return

        elif self.path == "/api/runs":
            runs = []
            for r in rs.list_runs():
                done = [rd for rd in r.get("rounds", []) if "global" in rd]
                last = done[-1]["global"] if done else {}
                scored = [(rd["global"].get("f1_macro"), rd["round"]) for rd in done if rd["global"].get("f1_macro") is not None]
                best_f1, best_round = max(scored) if scored else (None, None)
                run_dir = rs.RUNS_DIR / r["run_id"]
                has_best = (run_dir / "best.pt").exists()
                has_final = (run_dir / "final.pt").exists()
                runs.append({
                    "run_id": r["run_id"],
                    "strategy": r["strategy"],
                    "status": r.get("status"),
                    "started_at": r.get("started_at"),
                    "finished_at": r.get("finished_at"),
                    "rounds_done": len(done),
                    "config": r.get("config"),
                    "models": r.get("models"),
                    "best_f1": best_f1,
                    "best_round": best_round,
                    "final_loss": last.get("val_loss"),
                    "final_f1": last.get("f1_macro"),
                    "final_auroc": last.get("auroc"),
                    "has_best_model": has_best,
                    "has_final_model": has_final,
                })
            # Urutkan dari yang terbaru (descending)
            runs.sort(key=lambda x: x.get("started_at") or 0, reverse=True)
            self._json({"runs": runs, "total_runs": len(runs)})
            return

        elif self.path.startswith("/api/runs/"):
            run = rs.load_run(self.path[len("/api/runs/"):].split("?")[0])
            if not run:
                self.send_error(404, "Run tidak ditemukan")
                return
            self._json(run)
            return

        elif self.path == "/api/benchmark":
            running = bool(pm.server_process and pm.server_process.poll() is None)
            self._json({
                "status": "ok",
                "active_strategy": pm.config.get("strategy", "fedbn"),
                "benchmark": build_benchmark(running, _first_strategy(pm.config.get("strategy"))),
                "strategies": STRATEGY_INFO,
            })
            return

        # Static file routing
        if self.path.startswith("/static/") or self.path == "/" or self.path == "/index.html":
            if self.path in ("/", "/index.html"):
                full_path = os.path.join(STATIC_DIR, "index.html")
            else:
                rel_path = self.path[len("/static/"):].split("?")[0]
                full_path = os.path.join(STATIC_DIR, rel_path)

            if os.path.exists(full_path) and os.path.isfile(full_path):
                self.send_response(200)
                content_type, _ = mimetypes.guess_type(full_path)
                self.send_header("Content-Type", content_type or "application/octet-stream")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                with open(full_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        return super().do_GET()


def run():
    print(f"🏥 CardioPulse FL Minimalist Dashboard Server active on http://{HOST}:{PORT}")
    server_address = (HOST, PORT)
    httpd = ThreadingHTTPServer(server_address, CardioPulseHTTPHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        if pm.server_process and pm.server_process.poll() is None:
            pm.server_process.terminate()
        httpd.server_close()


if __name__ == "__main__":
    run()
