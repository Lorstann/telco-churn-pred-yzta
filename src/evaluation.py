"""Business-grade evaluation utilities shared by training and notebooks.

These helpers replicate the analyses developed in notebook 03
(threshold optimization, lift, segment audit, bootstrap CIs) so the
production pipeline ships with the exact same definitions used during
exploratory benchmarking.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

from src.config import CostMatrix, SegmentSpec


@dataclass(frozen=True)
class ThresholdResult:
    """Result of a threshold-optimization sweep."""

    threshold: float
    objective: str
    score: float


def cumulative_gain_curve(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Return a (population_pct, cumulative_capture_pct, lift) DataFrame.

    Sorts customers from highest to lowest predicted churn probability,
    walks the population, and reports how much of the total churn mass
    has been captured at each percentile. Used for top-decile lift and
    cumulative gain charts.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_prob_arr = np.asarray(y_prob).ravel()
    order = np.argsort(-y_prob_arr)
    sorted_targets = y_true_arr[order]

    population_pct = np.arange(1, len(sorted_targets) + 1) / len(sorted_targets)
    cum_positives = np.cumsum(sorted_targets)
    total_positives = max(cum_positives[-1], 1)
    capture_pct = cum_positives / total_positives
    base_rate = total_positives / len(sorted_targets)
    precision = cum_positives / np.arange(1, len(sorted_targets) + 1)
    lift = precision / max(base_rate, 1e-9)

    return pd.DataFrame(
        {
            "population_pct": population_pct,
            "capture_pct": capture_pct,
            "precision_at_k": precision,
            "lift": lift,
        }
    )


def top_decile_lift(y_true: np.ndarray, y_prob: np.ndarray, decile: float = 0.1) -> float:
    """Return lift achieved within the top ``decile`` fraction of scored customers."""
    if not 0 < decile < 1:
        raise ValueError("decile must be in (0, 1)")
    gains = cumulative_gain_curve(y_true, y_prob)
    cutoff_idx = max(int(np.ceil(decile * len(gains))) - 1, 0)
    return float(gains.iloc[cutoff_idx]["lift"])


def f1_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> ThresholdResult:
    """Find the threshold that maximizes F1 on the supplied predictions."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions * recalls / np.maximum(precisions + recalls, 1e-12)
    f1_scores = f1_scores[:-1]
    if len(thresholds) == 0:
        return ThresholdResult(threshold=0.5, objective="f1", score=0.0)
    best_idx = int(np.argmax(f1_scores))
    return ThresholdResult(
        threshold=float(thresholds[best_idx]),
        objective="f1",
        score=float(f1_scores[best_idx]),
    )


def cost_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_matrix: CostMatrix,
    *,
    grid: np.ndarray | None = None,
) -> ThresholdResult:
    """Find the threshold that minimizes expected business cost.

    Cost(t) = FN_cost * #FN(t) + FP_cost * #FP(t).

    The optimum is searched on a 199-point grid in (0.01, 0.99) by
    default, matching notebook 03's sweep resolution.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_prob_arr = np.asarray(y_prob).ravel()
    if grid is None:
        grid = np.linspace(0.01, 0.99, 199)

    best_threshold = 0.5
    best_cost = np.inf
    for threshold in grid:
        y_pred = (y_prob_arr >= threshold).astype(int)
        false_negatives = int(((y_true_arr == 1) & (y_pred == 0)).sum())
        false_positives = int(((y_true_arr == 0) & (y_pred == 1)).sum())
        total_cost = (
            false_negatives * cost_matrix.fn_cost_usd
            + false_positives * cost_matrix.fp_cost_usd
        )
        if total_cost < best_cost:
            best_cost = float(total_cost)
            best_threshold = float(threshold)

    return ThresholdResult(threshold=best_threshold, objective="cost", score=best_cost)


def expected_business_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_matrix: CostMatrix,
) -> dict[str, float]:
    """Decompose realized FN/FP cost at a fixed decision threshold."""
    y_true_arr = np.asarray(y_true).ravel()
    y_pred = (np.asarray(y_prob).ravel() >= threshold).astype(int)
    false_negatives = int(((y_true_arr == 1) & (y_pred == 0)).sum())
    false_positives = int(((y_true_arr == 0) & (y_pred == 1)).sum())
    return {
        "threshold": float(threshold),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "fn_cost_usd": false_negatives * cost_matrix.fn_cost_usd,
        "fp_cost_usd": false_positives * cost_matrix.fp_cost_usd,
        "total_cost_usd": (
            false_negatives * cost_matrix.fn_cost_usd
            + false_positives * cost_matrix.fp_cost_usd
        ),
    }


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute a bootstrap confidence interval for ROC-AUC.

    Resamples (with replacement) preserving sample size, recomputes ROC
    AUC, and returns both the empirical CI and the spread (std). Mirrors
    the methodology in notebook 03, Section 10.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_prob_arr = np.asarray(y_prob).ravel()
    rng = np.random.default_rng(random_state)
    n_samples = len(y_true_arr)
    scores: list[float] = []
    for _ in range(n_iterations):
        idx = rng.integers(0, n_samples, n_samples)
        if y_true_arr[idx].sum() in (0, n_samples):
            continue
        scores.append(float(roc_auc_score(y_true_arr[idx], y_prob_arr[idx])))
    if not scores:
        point = float(roc_auc_score(y_true_arr, y_prob_arr))
        return {"mean": point, "std": 0.0, "lower": point, "upper": point}

    arr = np.asarray(scores)
    alpha = (1 - confidence) / 2
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "lower": float(np.quantile(arr, alpha)),
        "upper": float(np.quantile(arr, 1 - alpha)),
    }


def evaluate_segments(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_prob: np.ndarray,
    segment_specs: tuple[SegmentSpec, ...],
    *,
    threshold: float = 0.5,
    min_size: int = 30,
) -> pd.DataFrame:
    """Compute per-segment AUC / precision / recall / capture.

    Discrete columns use their unique values as segments. Numeric columns
    with a `bins` tuple are bucketed via :func:`pandas.cut`. Segments
    smaller than ``min_size`` are dropped to avoid noisy AUCs.
    """
    rows: list[dict[str, float]] = []
    y_true_arr = np.asarray(y_true).ravel()
    y_prob_arr = np.asarray(y_prob).ravel()
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    for spec in segment_specs:
        if spec.column not in df.columns:
            continue
        if spec.bins is not None:
            buckets = pd.cut(
                df[spec.column],
                bins=list(spec.bins),
                labels=list(spec.labels) if spec.labels else None,
            )
        else:
            buckets = df[spec.column].astype(str)

        bucket_arr = np.asarray(buckets)
        for segment_value in pd.unique(buckets.dropna()):
            mask = bucket_arr == segment_value
            count = int(mask.sum())
            if count < min_size:
                continue
            seg_y = y_true_arr[mask]
            seg_prob = y_prob_arr[mask]
            seg_pred = y_pred_arr[mask]
            try:
                seg_auc = float(roc_auc_score(seg_y, seg_prob))
            except ValueError:
                seg_auc = float("nan")
            tp = int(((seg_y == 1) & (seg_pred == 1)).sum())
            fp = int(((seg_y == 0) & (seg_pred == 1)).sum())
            fn = int(((seg_y == 1) & (seg_pred == 0)).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            rows.append(
                {
                    "dimension": spec.name,
                    "segment": str(segment_value),
                    "n": count,
                    "churn_rate": float(seg_y.mean()),
                    "auc": seg_auc,
                    "precision": precision,
                    "recall": recall,
                    "f1": (
                        2 * precision * recall / max(precision + recall, 1e-12)
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(["dimension", "auc"], ascending=[True, False])


def gain_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    """Capture rate, contact rate and F1 at a single decision threshold."""
    y_true_arr = np.asarray(y_true).ravel()
    y_pred = (np.asarray(y_prob).ravel() >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "contacts_pct": float(y_pred.mean()),
        "capture_pct": float(((y_true_arr == 1) & (y_pred == 1)).sum() / max(y_true_arr.sum(), 1)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
    }
