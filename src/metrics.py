"""Metrics never silently replace undefined AUC with 0.5."""
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_metrics(probabilities, targets):
    p, y = np.asarray(probabilities), np.asarray(targets)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("Empty predictions or shape mismatch")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite predictions or targets")
    if np.any((p < 0) | (p > 1)) or np.any((y < 0) | (y > 1)):
        raise ValueError("Probabilities and targets must lie in [0, 1]")
    binary = (y > 0.5).astype(int).reshape(-1)
    auc = float(roc_auc_score(binary, p.reshape(-1))) if np.unique(binary).size == 2 else None
    return {"auc": auc, "mae": float(np.abs(p-y).mean()), "points": int(p.size)}
