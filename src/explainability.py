"""Explainability utilities: coefficients, permutation, and SHAP helpers.

Notebook 04 produced three views of feature importance: the linear-model
coefficient table, model-agnostic permutation importance, and Tree SHAP
on an auxiliary XGBoost. This module ports those routines into reusable
production helpers so :mod:`src.train` can persist the same artifacts
that the notebook visualizes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def _get_feature_names(pipeline: Pipeline) -> np.ndarray:
    """Return post-transform feature names from a fitted pipeline."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None:
        raise ValueError("Pipeline does not expose a 'preprocessor' step.")
    return np.asarray(preprocessor.get_feature_names_out())


def linear_coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    """Extract a sorted coefficient table from a logistic-regression pipeline.

    Returns columns ``[feature, coefficient, odds_ratio, abs_coefficient]``.
    Useful both for production audit logs and for the explainability page
    of the dashboard.
    """
    estimator = pipeline.named_steps.get("model")
    if not isinstance(estimator, LogisticRegression):
        raise TypeError(
            "linear_coefficient_table requires a LogisticRegression estimator; "
            f"got {type(estimator).__name__}."
        )
    feature_names = _get_feature_names(pipeline)
    coefs = estimator.coef_.ravel()
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
            "abs_coefficient": np.abs(coefs),
        }
    )
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def permutation_importance_table(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_repeats: int = 15,
    random_state: int = 42,
    scoring: str = "roc_auc",
    top_k: int | None = None,
) -> pd.DataFrame:
    """Compute permutation importance on the *raw* model input.

    Args:
        pipeline: Fitted preprocessor + estimator pipeline.
        X: Raw (post-engineering) feature DataFrame.
        y: Ground-truth target.
        n_repeats: Number of shuffles per feature.
        scoring: sklearn-compatible scoring string.
        top_k: If provided, keep only the top-k entries by mean drop.
    """
    result = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    if top_k is not None:
        table = table.head(top_k)
    return table.reset_index(drop=True)


def tree_shap_importance_table(
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    sample_size: int = 1000,
    random_state: int = 42,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Mean |SHAP| importance for tree-based pipelines.

    Imports SHAP lazily because it has heavy native deps; missing SHAP
    is downgraded to a clean error so :mod:`src.train` can still ship
    other artifacts.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "tree_shap_importance_table requires the 'shap' package."
        ) from exc

    preprocessor = pipeline.named_steps.get("preprocessor")
    estimator = pipeline.named_steps.get("model")
    if preprocessor is None or estimator is None:
        raise ValueError("Pipeline must expose 'preprocessor' and 'model' steps.")

    rng = np.random.default_rng(random_state)
    if len(X) > sample_size:
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X.iloc[idx]
    else:
        X_sample = X

    transformed = preprocessor.transform(X_sample)
    explainer = shap.TreeExplainer(estimator)
    shap_values: Any = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    feature_names = _get_feature_names(pipeline)
    mean_abs = np.abs(shap_values).mean(axis=0)
    table = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)
    if top_k is not None:
        table = table.head(top_k)
    return table.reset_index(drop=True)
