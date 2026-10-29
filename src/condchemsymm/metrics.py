from __future__ import annotations

import math
from typing import Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }
    if len(np.unique(y_true)) > 2:
        from scipy.stats import spearmanr

        rho = spearmanr(y_true, y_pred).correlation
        out["spearman"] = float(0.0 if np.isnan(rho) else rho)
    else:
        out["spearman"] = float("nan")
    return out


def binary_calibration_errors(y_true, prob, n_bins: int = 10) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == 0:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob > lo) & (prob <= hi)
        if not np.any(mask):
            continue
        gap = abs(float(y_true[mask].mean()) - float(prob[mask].mean()))
        ece += float(mask.mean()) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def classification_metrics(y_true, prob) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    clipped = np.clip(prob, 1e-7, 1.0 - 1e-7)
    ece10, mce10 = binary_calibration_errors(y_true, clipped, n_bins=10)
    pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": float((pred == y_true).mean()),
        "brier": float(brier_score_loss(y_true, clipped)),
        "nll": float(log_loss(y_true, clipped, labels=[0, 1])),
        "ece10": ece10,
        "mce10": mce10,
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
        out["pr_auc"] = float(average_precision_score(y_true, prob))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out
