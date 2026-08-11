"""Evaluation metrics for the imbalanced cross-project SVD setting.

Because MegaVul is highly imbalanced (~1 vulnerable : ~19 safe functions),
a fixed-0.5-threshold F1 is misleading. We therefore expose:

  * ``compute_metrics``       — threshold-dependent scores at a *given* threshold.
  * ``pr_auc``                — threshold-free area under the precision-recall curve
                                (our PRIMARY metric).
  * ``recall_at_precision``   — operating-point recall at a fixed precision floor
                                (useful for security triage).
  * ``tune_threshold``        — pick the decision threshold that maximises F1 on a
                                validation set; the SAME tuned threshold is then
                                applied at test time (no test-set peeking).

All methods in this study share these functions so the comparison is fair.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


# ---------------------------------------------------------------------------
# Threshold-free helpers
# ---------------------------------------------------------------------------

def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision)."""
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, np.asarray(y_prob, dtype=float)))


def recall_at_precision(y_true: np.ndarray, y_prob: np.ndarray,
                        min_precision: float = 0.5) -> float:
    """Highest recall achievable while precision >= ``min_precision``.

    Reported because a security team typically fixes an acceptable
    false-alarm rate (precision) and then cares about how many true
    vulnerabilities are caught (recall) at that operating point.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    feasible = rec[prec >= min_precision]
    return float(feasible.max()) if feasible.size > 0 else 0.0


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                   objective: str = "f1",
                   min_precision: float = 0.5) -> Tuple[float, float]:
    """Search the decision threshold on a validation set.

    Returns ``(best_threshold, best_score)``. ``objective='f1'`` maximises
    F1; ``objective='recall_at_precision'`` maximises recall subject to a
    precision floor. Candidate thresholds are the unique predicted scores.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan")
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thr) == len(prec) - 1
    prec, rec = prec[:-1], rec[:-1]
    if thr.size == 0:
        return 0.5, float("nan")
    if objective == "f1":
        denom = np.clip(prec + rec, 1e-12, None)
        f1s = 2 * prec * rec / denom
        best = int(np.nanargmax(f1s))
        return float(thr[best]), float(f1s[best])
    elif objective == "recall_at_precision":
        mask = prec >= min_precision
        if not mask.any():
            return 0.5, 0.0
        idx = np.where(mask)[0]
        best = idx[int(np.argmax(rec[idx]))]
        return float(thr[best]), float(rec[best])
    raise ValueError(f"Unknown objective {objective!r}")


# ---------------------------------------------------------------------------
# Threshold-dependent metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5,
                    group: Optional[np.ndarray] = None) -> Dict:
    """Binary classification metrics at a *given* threshold, optionally per-group.

    ``pr_auc`` and ``roc_auc`` are threshold-free and therefore stable; the
    F1/precision/recall reflect the supplied ``threshold`` (tuned on val).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "threshold": float(threshold),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc":    pr_auc(y_true, y_prob),
        "roc_auc":   float(roc_auc_score(y_true, y_prob))
                     if len(np.unique(y_true)) > 1 else float("nan"),
        "recall_at_p50": recall_at_precision(y_true, y_prob, 0.5),
        "n":         int(len(y_true)),
        "n_pos":     int(y_true.sum()),
        "pos_rate":  float(y_true.mean()) if len(y_true) else float("nan"),
    }
    if group is not None:
        group = np.asarray(group)
        groups = {}
        for g in np.unique(group):
            mask = group == g
            if mask.sum() == 0:
                continue
            groups[int(g)] = compute_metrics(y_true[mask], y_prob[mask],
                                             threshold=threshold)
        out["groups"] = groups
    return out
