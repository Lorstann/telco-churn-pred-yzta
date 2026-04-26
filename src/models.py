"""Model registry and metric helpers for the telco churn pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.config import RANDOM_STATE


@dataclass
class ModelResult:
    """Container for a trained model and its evaluation metrics."""

    model_name: str
    metrics: dict[str, float]
    threshold: float
    estimator: Any


def compute_scale_pos_weight(y_train: pd.Series | np.ndarray) -> float:
    """Return XGBoost-style ``scale_pos_weight`` for an imbalanced binary target.

    XGBoost has no ``class_weight`` argument, so we mirror the same effect
    through ``scale_pos_weight = N(class=0) / N(class=1)``. Values are
    floored at 1.0 because the loss penalty for an over-represented
    positive class would only hurt PR-AUC.
    """
    y = np.asarray(y_train).ravel()
    positives = float((y == 1).sum())
    negatives = float((y == 0).sum())
    if positives <= 0:
        return 1.0
    return max(negatives / positives, 1.0)


def get_model_registry(
    class_weight: str | dict[int, float] | None = "balanced",
    *,
    scale_pos_weight: float | None = None,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Return the baseline model registry used in notebook 03.

    Args:
        class_weight: Forwarded to estimators that natively support it.
        scale_pos_weight: Optional override for XGBoost. Pass the value
            returned by :func:`compute_scale_pos_weight` to mirror
            ``class_weight="balanced"`` behaviour during boosting.
        random_state: Seed propagated to every stochastic estimator.
    """
    xgb_kwargs: dict[str, Any] = dict(
        n_estimators=450,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        eval_metric="logloss",
        tree_method="hist",
    )
    if scale_pos_weight is not None:
        xgb_kwargs["scale_pos_weight"] = float(scale_pos_weight)

    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000, class_weight=class_weight, solver="lbfgs"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
            class_weight=class_weight,
        ),
        "xgboost": XGBClassifier(**xgb_kwargs),
        "lightgbm": LGBMClassifier(
            n_estimators=450,
            learning_rate=0.03,
            random_state=random_state,
            class_weight=class_weight,
            verbosity=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=450,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_state,
            auto_class_weights="Balanced",
            verbose=False,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=random_state
        ),
    }


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the standard classification metric battery from probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }
