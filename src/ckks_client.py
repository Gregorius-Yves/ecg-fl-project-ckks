import base64
import json
import os
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np


def _is_ckks_enabled() -> bool:
    return os.environ.get("CKKS_ENABLED", "0") == "1"


def _sidecar_url() -> str:
    return os.environ.get("CKKS_SIDECAR_URL", "http://localhost:9090")


class CKKSClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or _sidecar_url()).rstrip("/")

    def _post(self, path: str, data: dict) -> dict:
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(f"{self.base_url}{path}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    def health(self) -> dict:
        return self._get("/health")

    def init(self, param_set: str = "FL-D1-N16384") -> dict:
        return self._post("/init", {"param_set": param_set})

    def encrypt(self, weights: np.ndarray) -> Tuple[List[bytes], Dict]:
        """Encrypt a flat float64 weight vector. Returns (ciphertext_chunks, metrics)."""
        flat = weights.astype(np.float64).ravel()
        w_b64 = base64.b64encode(flat.tobytes()).decode()
        resp = self._post("/encrypt", {
            "weights_b64": w_b64,
            "num_params": len(flat),
        })
        chunks = [base64.b64decode(c) for c in resp["chunks_b64"]]
        metrics = {
            "ckks_encrypt_ms": resp["encrypt_ms"],
            "ckks_num_chunks": resp["num_chunks"],
            "ckks_bytes_total": resp["bytes_total"],
        }
        return chunks, metrics

    def aggregate(
        self,
        client_chunks: List[List[bytes]],
        client_weights: List[float],
    ) -> Tuple[List[bytes], Dict]:
        """Aggregate encrypted updates with given weights. Returns (aggregated_chunks, metrics)."""
        updates = []
        for chunks, weight in zip(client_chunks, client_weights):
            updates.append({
                "chunks_b64": [base64.b64encode(c).decode() for c in chunks],
                "weight": weight,
            })
        resp = self._post("/aggregate", {"updates": updates})
        agg_chunks = [base64.b64decode(c) for c in resp["chunks_b64"]]
        metrics = {
            "ckks_aggregate_ms": resp["aggregate_ms"],
            "ckks_num_clients": resp["num_clients"],
        }
        return agg_chunks, metrics

    def decrypt(self, chunks: List[bytes], num_params: int) -> Tuple[np.ndarray, Dict]:
        """Decrypt aggregated ciphertext chunks. Returns (float64_vector, metrics)."""
        resp = self._post("/decrypt", {
            "chunks_b64": [base64.b64encode(c).decode() for c in chunks],
            "num_params": num_params,
        })
        flat = np.frombuffer(base64.b64decode(resp["weights_b64"]), dtype=np.float64)
        metrics = {"ckks_decrypt_ms": resp["decrypt_ms"]}
        return flat.copy(), metrics

    def is_available(self) -> bool:
        try:
            h = self.health()
            return h.get("status") == "ok" and h.get("initialized", False)
        except Exception:
            return False


_client: Optional[CKKSClient] = None


def get_ckks_client() -> Optional[CKKSClient]:
    """Return a shared CKKSClient if CKKS is enabled and the sidecar is reachable."""
    global _client
    if not _is_ckks_enabled():
        return None
    if _client is None:
        _client = CKKSClient()
    return _client
