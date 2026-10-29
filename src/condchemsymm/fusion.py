"""Numeric fusion kernels shared by the experiment adapters."""


from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import classification_metrics


def paired_stats(df: pd.DataFrame, pred_col: str, threshold: float = 0.25) -> dict[str, float]:
    pred_by_id = dict(zip(df["sample_id"], df[pred_col]))
    y_by_id = dict(zip(df["sample_id"], df["y"]))
    false_flags = []
    order_correct = []
    gaps = []
    seen = set()
    for _, row in df.iterrows():
        sid = row["sample_id"]
        pid = row.get("paired_id", "")
        if not pid or pid not in pred_by_id or sid in seen or pid in seen:
            continue
        seen.add(sid)
        seen.add(pid)
        pred_a = float(pred_by_id[sid])
        pred_b = float(pred_by_id[pid])
        y_a = float(y_by_id[sid])
        y_b = float(y_by_id[pid])
        gap = abs(pred_a - pred_b)
        gaps.append(gap)
        false_flags.append(gap < threshold)
        if y_a != y_b:
            order_correct.append(pred_a > pred_b if y_a > y_b else pred_b > pred_a)
    if not false_flags:
        return {
            "false_invariance_rate_test_pairs": float("nan"),
            "false_invariance_n_pairs": 0,
            "pair_order_accuracy": float("nan"),
            "pair_gap_mean": float("nan"),
            "pair_gap_p10": float("nan"),
        }
    return {
        "false_invariance_rate_test_pairs": float(np.mean(false_flags)),
        "false_invariance_n_pairs": int(len(false_flags)),
        "pair_order_accuracy": float(np.mean(order_correct)) if order_correct else float("nan"),
        "pair_gap_mean": float(np.mean(gaps)),
        "pair_gap_p10": float(np.quantile(gaps, 0.10)),
    }

def fast_auc_scores(y_true: np.ndarray, predictions: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    y_bool = y_true.astype(float) > 0.5
    pos = predictions[:, y_bool]
    neg = predictions[:, ~y_bool]
    if pos.shape[1] == 0 or neg.shape[1] == 0:
        return np.full(predictions.shape[0], np.nan)
    denom = float(pos.shape[1] * neg.shape[1])
    scores = np.empty(predictions.shape[0], dtype=float)
    for start in range(0, predictions.shape[0], chunk_size):
        stop = min(start + chunk_size, predictions.shape[0])
        diff = pos[start:stop, :, None] - neg[start:stop, None, :]
        scores[start:stop] = ((diff > 0).sum(axis=(1, 2)) + 0.5 * (diff == 0).sum(axis=(1, 2))) / denom
    return scores

def pair_indices(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    row_by_id = {sample_id: idx for idx, sample_id in enumerate(df["sample_id"])}
    left = []
    right = []
    seen = set()
    for _, row in df.iterrows():
        sid = row["sample_id"]
        pid = row.get("paired_id", "")
        if not pid or pid not in row_by_id or sid in seen or pid in seen:
            continue
        seen.add(sid)
        seen.add(pid)
        left.append(row_by_id[sid])
        right.append(row_by_id[pid])
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)

def fast_false_invariance_scores(predictions: np.ndarray, left: np.ndarray, right: np.ndarray, threshold: float = 0.25) -> np.ndarray:
    if len(left) == 0:
        return np.full(predictions.shape[0], np.nan)
    gaps = np.abs(predictions[:, left] - predictions[:, right])
    return np.mean(gaps < threshold, axis=1)

def safe_logit(prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob.astype(float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))

def sigmoid(score: np.ndarray) -> np.ndarray:
    clipped = np.clip(score, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))

def project_probability_norm(df: pd.DataFrame, raw_predictions: np.ndarray) -> np.ndarray:
    projected = raw_predictions.copy()
    left, right = pair_indices(df)
    if len(left) == 0:
        return projected
    left_prob = np.clip(raw_predictions[:, left], 1e-7, 1.0)
    right_prob = np.clip(raw_predictions[:, right], 1e-7, 1.0)
    pair_prob = left_prob / (left_prob + right_prob)
    projected[:, left] = pair_prob
    projected[:, right] = 1.0 - pair_prob
    return np.clip(projected, 1e-7, 1.0 - 1e-7)

def project_logit_difference(df: pd.DataFrame, scores: np.ndarray, scale: float) -> np.ndarray:
    projected = sigmoid(float(scale) * scores)
    left, right = pair_indices(df)
    if len(left) == 0:
        return projected
    pair_prob = sigmoid(float(scale) * (scores[:, left] - scores[:, right]))
    projected[:, left] = pair_prob
    projected[:, right] = 1.0 - pair_prob
    return np.clip(projected, 1e-7, 1.0 - 1e-7)

def vector_nll(y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    clipped = np.clip(predictions, 1e-7, 1.0 - 1e-7)
    return np.mean(-(y[None, :] * np.log(clipped) + (1.0 - y[None, :]) * np.log(1.0 - clipped)), axis=1)

def vector_brier(y: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return np.mean((predictions - y[None, :]) ** 2, axis=1)

def evaluate_predictions(df: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    eval_df = df[["sample_id", "y", "paired_id"]].copy()
    eval_df["prediction"] = prediction
    metrics = classification_metrics(eval_df["y"].to_numpy(dtype=float), prediction)
    pair = paired_stats(eval_df, "prediction")
    return {
        "test_accuracy": metrics["accuracy"],
        "test_brier": metrics["brier"],
        "test_nll": metrics["nll"],
        "test_ece10": metrics["ece10"],
        "test_mce10": metrics["mce10"],
        "test_roc_auc": metrics["roc_auc"],
        "test_pr_auc": metrics["pr_auc"],
        **pair,
    }

def simplex_weights(n_components: int, denom: int) -> np.ndarray:
    rows: list[tuple[int, ...]] = []

    def rec(prefix: tuple[int, ...], remaining_components: int, remaining: int) -> None:
        if remaining_components == 1:
            rows.append((*prefix, remaining))
            return
        for value in range(remaining + 1):
            rec((*prefix, value), remaining_components - 1, remaining - value)

    rec((), n_components, int(denom))
    return np.asarray(rows, dtype=float) / float(denom)

def policy_scale(policy: str) -> float:
    if policy == "pair_logitdiff_s1":
        return 1.0
    if policy == "pair_logitdiff_s1p2":
        return 1.2
    if policy == "pair_logitdiff_s1p5":
        return 1.5
    return 1.0

def policy_predictions(df: pd.DataFrame, columns: list[str], weights: np.ndarray, policy: str) -> np.ndarray:
    components = np.clip(df[columns].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    if policy == "prob_avg":
        return np.clip(weights @ components.T, 1e-7, 1.0 - 1e-7)
    if policy == "prob_pair_norm":
        return project_probability_norm(df, np.clip(weights @ components.T, 1e-7, 1.0 - 1e-7))
    if policy == "logit_avg":
        return sigmoid(weights @ safe_logit(components).T)
    if policy.startswith("pair_logitdiff_"):
        scores = weights @ safe_logit(components).T
        return project_logit_difference(df, scores, policy_scale(policy))
    raise ValueError(policy)

def choose_candidate(
    val: pd.DataFrame,
    columns: list[str],
    weights: np.ndarray,
    policy: str,
    objective_weight: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    predictions = policy_predictions(val, columns, weights, policy)
    y = val["y"].to_numpy(dtype=float)
    left, right = pair_indices(val)
    aucs = fast_auc_scores(y, predictions, chunk_size=256)
    false_inv = fast_false_invariance_scores(predictions, left, right)
    nll = vector_nll(y, predictions)
    brier = vector_brier(y, predictions)
    objective = aucs - float(objective_weight) * false_inv
    best_score = float(np.nanmax(objective))
    candidates = np.flatnonzero(np.isclose(objective, best_score, rtol=0.0, atol=1e-12))

    def tie_key(idx: int) -> tuple[float, float, float, float]:
        tp_weight = float(weights[idx, columns.index("tp_pairnorm")]) if "tp_pairnorm" in columns else 0.0
        external_weight = sum(float(weights[idx, columns.index(col)]) for col in columns if col.startswith(("tp_", "sphys")))
        return (float(nll[idx]), float(brier[idx]), float(false_inv[idx]), -tp_weight + 0.01 * external_weight)

    best_idx = int(min(candidates, key=tie_key))
    selected = weights[best_idx]
    metrics = evaluate_predictions(val, predictions[best_idx])
    metrics.update(
        {
            "val_selection_score": best_score,
            "val_selection_nll": float(nll[best_idx]),
            "val_selection_brier": float(brier[best_idx]),
        }
    )
    return selected, best_score, metrics
