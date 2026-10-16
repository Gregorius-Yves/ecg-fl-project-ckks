"""
Metrik evaluasi multi-label EKG (dihitung dari prediksi model sebenarnya).

- Macro-F1 & AUROC dirata-ratakan hanya pada kelas yang BERMAKNA di data validasi
  (F1: kelas dengan >=1 sampel positif; AUROC: kelas dengan positif DAN negatif),
  karena kelas tanpa sampel membuat metrik tak terdefinisi.
"""

import math
from typing import Dict, List

import numpy as np


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], threshold: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import f1_score, roc_auc_score

    y_true = np.asarray(y_true).astype(np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int8)
    n = y_true.shape[0]
    support = y_true.sum(axis=0)

    out: Dict[str, float] = {}
    f1s, aucs = [], []
    for i, name in enumerate(class_names):
        if support[i] > 0:
            f = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            f1s.append(f)
            out[f"f1_c_{name}"] = float(f)
        else:
            out[f"f1_c_{name}"] = float("nan")
        if 0 < support[i] < n:
            try:
                aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
            except ValueError:
                pass

    out["f1_macro"] = float(np.mean(f1s)) if f1s else float("nan")
    out["auroc"] = float(np.mean(aucs)) if aucs else float("nan")
    out["f1_micro"] = float(f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0))
    out["n_classes_f1"] = float(len(f1s))
    out["n_classes_auc"] = float(len(aucs))
    return out


def clean(value):
    """NaN/Inf -> None agar JSON valid."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else v
