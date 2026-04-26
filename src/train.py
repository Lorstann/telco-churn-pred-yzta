"""Production training pipeline for the telco churn project.

Implements the workflow validated across notebooks 01-04:
  * 70/15/15 stratified split (train / valid / test).
  * Imbalance handling via class weights and XGBoost ``scale_pos_weight``.
  * Six-model benchmark with full metric battery on train/valid/test.
  * Cost-optimal threshold derived from a configurable cost matrix.
  * Multi-criteria champion gate (ROC-AUC + calibration + segment).
  * Per-segment audit, permutation importance, bootstrap CIs persisted.
  * Linear coefficient table for the linear-model champion.
  * Curve gallery (ROC, PR, CM, calibration) per model for reports/.

Run ``python -m src.train`` after activating the virtualenv.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    BOOTSTRAP_CI_PATH,
    CHAMPION_METADATA_PATH,
    CHAMPION_PIPELINE_PATH,
    CURVES_DIR,
    DECISION_THRESHOLD_PATH,
    DEFAULT_DATA_PATH,
    FIGURES_DIR,
    LEADERBOARD_PATH,
    MODELS_DIR,
    PERMUTATION_IMPORTANCE_PATH,
    PREPROCESSING_REPORT_PATH,
    REPORTS_DIR,
    SEGMENT_AUDIT_PATH,
    SEGMENT_SPECS,
    TEST_SIZE,
    VALID_SIZE,
    ChampionGate,
    CostMatrix,
    RANDOM_STATE,
    ensure_directories,
)
from src.evaluation import (
    bootstrap_auc_ci,
    cost_optimal_threshold,
    evaluate_segments,
    expected_business_cost,
    f1_optimal_threshold,
    gain_at_threshold,
    top_decile_lift,
)
from src.explainability import (
    linear_coefficient_table,
    permutation_importance_table,
)
from src.models import compute_scale_pos_weight, evaluate_predictions, get_model_registry
from src.preprocessing import (
    TARGET_COLUMN,
    build_preprocessor,
    engineer_features,
    generate_data_quality_report,
    load_data,
    split_features_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train telco churn models.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--models-dir", default=str(MODELS_DIR))
    parser.add_argument(
        "--fn-cost", type=float, default=CostMatrix().fn_cost_usd,
        help="Business cost per false negative (missed churner)."
    )
    parser.add_argument(
        "--fp-cost", type=float, default=CostMatrix().fp_cost_usd,
        help="Business cost per false positive (wasted retention contact)."
    )
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=1000,
        help="Bootstrap iterations for AUC confidence intervals."
    )
    return parser.parse_args()


def _plot_eda(df: pd.DataFrame, figure_dir: Path) -> None:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=TARGET_COLUMN)
    plt.title("Target Distribution")
    plt.tight_layout()
    plt.savefig(figure_dir / "target_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        missing = pd.Series([0], index=["no_missing"])
    sns.barplot(x=missing.values, y=missing.index, orient="h")
    plt.title("Missing Values by Feature")
    plt.tight_layout()
    plt.savefig(figure_dir / "missing_values.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 8))
    corr_df = engineer_features(df).select_dtypes(include=np.number)
    sns.heatmap(corr_df.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numerical + Engineered)")
    plt.tight_layout()
    plt.savefig(figure_dir / "correlation_heatmap.png", dpi=160)
    plt.close()


def _save_model_curves(
    y_test: pd.Series,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    curve_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name=model_name)
    ax.set_title(f"ROC Curve - {model_name}")
    fig.tight_layout()
    fig.savefig(curve_dir / f"{model_name}_roc.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax, name=model_name)
    ax.set_title(f"Precision-Recall Curve - {model_name}")
    fig.tight_layout()
    fig.savefig(curve_dir / f"{model_name}_pr.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(curve_dir / f"{model_name}_cm.png", dpi=160)
    plt.close(fig)

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve - {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(curve_dir / f"{model_name}_calibration.png", dpi=160)
    plt.close(fig)


def _select_champion(
    leaderboard: pd.DataFrame,
    pipelines: dict[str, Pipeline],
    bootstrap_results: dict[str, dict[str, float]],
    *,
    gate: ChampionGate,
) -> str:
    """Apply the multi-criteria gate from notebook 03 (Section 12)."""
    ranked = leaderboard.sort_values("test_roc_auc", ascending=False)
    leader_brier = float(ranked.iloc[0]["test_brier_score"])
    threshold_brier = leader_brier * gate.calibration_multiplier

    for _, row in ranked.iterrows():
        name = str(row["model"])
        if float(row["test_brier_score"]) > threshold_brier:
            continue
        boot = bootstrap_results.get(name, {})
        if float(boot.get("std", 0.0)) > gate.max_bootstrap_std:
            continue
        return name
    return str(ranked.iloc[0]["model"])


def _persist_segment_audit(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    output_path: Path,
) -> pd.DataFrame:
    audit_df = evaluate_segments(
        df=X_test,
        y_true=y_test,
        y_prob=y_prob,
        segment_specs=SEGMENT_SPECS,
        threshold=threshold,
    )
    audit_df.to_csv(output_path, index=False)
    return audit_df


def _persist_permutation_importance(
    pipeline: Pipeline,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    output_path: Path,
    *,
    random_state: int,
) -> None:
    table = permutation_importance_table(
        pipeline,
        X_valid,
        y_valid,
        n_repeats=10,
        random_state=random_state,
        scoring="roc_auc",
    )
    table.to_csv(output_path, index=False)


def _save_linear_coefficients(pipeline: Pipeline, output_path: Path) -> None:
    if not isinstance(pipeline.named_steps.get("model"), LogisticRegression):
        return
    table = linear_coefficient_table(pipeline)
    table.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    models_dir = Path(args.models_dir)
    figure_dir = reports_dir / "figures"
    curve_dir = reports_dir / "model_curves"
    for directory in (reports_dir, figure_dir, curve_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)
    ensure_directories()

    cost_matrix = CostMatrix(fn_cost_usd=args.fn_cost, fp_cost_usd=args.fp_cost)
    gate = ChampionGate()

    raw_df = load_data(args.data_path)
    quality_report = generate_data_quality_report(raw_df)
    with PREPROCESSING_REPORT_PATH.open("w", encoding="utf-8") as fp:
        json.dump(quality_report, fp, indent=2)

    _plot_eda(raw_df, figure_dir)

    features_df = engineer_features(raw_df)
    prepared = split_features_target(features_df)
    X, y = prepared.X, prepared.y

    holdout_size = args.__dict__.get("holdout_size", VALID_SIZE + TEST_SIZE)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=holdout_size, random_state=args.random_state, stratify=y
    )
    relative_test = TEST_SIZE / (VALID_SIZE + TEST_SIZE)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test,
        random_state=args.random_state,
        stratify=y_temp,
    )

    scale_pos_weight = compute_scale_pos_weight(y_train)
    model_registry = get_model_registry(
        class_weight="balanced",
        scale_pos_weight=scale_pos_weight,
        random_state=args.random_state,
    )

    leaderboard_rows: list[dict[str, Any]] = []
    pipelines: dict[str, Pipeline] = {}
    test_probs: dict[str, np.ndarray] = {}
    bootstrap_results: dict[str, dict[str, float]] = {}

    for model_name, model in model_registry.items():
        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        fit_start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start

        y_prob_train = pipeline.predict_proba(X_train)[:, 1]
        y_prob_valid = pipeline.predict_proba(X_valid)[:, 1]
        y_prob_test = pipeline.predict_proba(X_test)[:, 1]

        train_metrics = evaluate_predictions(y_train, y_prob_train)
        valid_metrics = evaluate_predictions(y_valid, y_prob_valid)
        test_metrics = evaluate_predictions(y_test, y_prob_test)

        y_pred_test = (y_prob_test >= 0.5).astype(int)
        _save_model_curves(y_test, y_prob_test, y_pred_test, model_name, curve_dir)

        boot = bootstrap_auc_ci(
            y_test.to_numpy(),
            y_prob_test,
            n_iterations=args.bootstrap_iterations,
            random_state=args.random_state,
        )
        bootstrap_results[model_name] = boot

        leaderboard_rows.append(
            {
                "model": model_name,
                "fit_seconds": round(fit_seconds, 3),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"valid_{k}": v for k, v in valid_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
                "test_top_decile_lift": top_decile_lift(
                    y_test.to_numpy(), y_prob_test
                ),
                "test_roc_auc_ci_lower": boot["lower"],
                "test_roc_auc_ci_upper": boot["upper"],
                "test_roc_auc_std": boot["std"],
            }
        )
        pipelines[model_name] = pipeline
        test_probs[model_name] = y_prob_test

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(
        "test_roc_auc", ascending=False
    )
    leaderboard_df.to_csv(LEADERBOARD_PATH, index=False)

    bootstrap_df = pd.DataFrame(bootstrap_results).T.reset_index().rename(
        columns={"index": "model"}
    )
    bootstrap_df.to_csv(BOOTSTRAP_CI_PATH, index=False)

    champion_name = _select_champion(
        leaderboard_df, pipelines, bootstrap_results, gate=gate
    )
    champion_pipeline = pipelines[champion_name]
    champion_probs = test_probs[champion_name]

    f1_threshold = f1_optimal_threshold(y_test.to_numpy(), champion_probs)
    cost_threshold = cost_optimal_threshold(
        y_test.to_numpy(), champion_probs, cost_matrix=cost_matrix
    )
    production_threshold = cost_threshold.threshold
    cost_breakdown = expected_business_cost(
        y_test.to_numpy(), champion_probs, production_threshold, cost_matrix
    )
    threshold_summary = gain_at_threshold(
        y_test.to_numpy(), champion_probs, production_threshold
    )

    audit_df = _persist_segment_audit(
        champion_pipeline,
        X_test,
        y_test,
        champion_probs,
        production_threshold,
        SEGMENT_AUDIT_PATH,
    )

    _persist_permutation_importance(
        champion_pipeline,
        X_valid,
        y_valid,
        PERMUTATION_IMPORTANCE_PATH,
        random_state=args.random_state,
    )

    _save_linear_coefficients(
        champion_pipeline, REPORTS_DIR / "linear_coefficients.csv"
    )

    champion_test_metrics = evaluate_predictions(
        y_test, champion_probs, threshold=production_threshold
    )
    champion_metadata = {
        "model_name": champion_name,
        "selection_metric": "test_roc_auc",
        "selection_score": float(
            leaderboard_df.set_index("model").loc[champion_name, "test_roc_auc"]
        ),
        "scale_pos_weight": float(scale_pos_weight),
        "n_train": int(len(X_train)),
        "n_valid": int(len(X_valid)),
        "n_test": int(len(X_test)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "version": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        "cost_matrix": {
            "fn_cost_usd": cost_matrix.fn_cost_usd,
            "fp_cost_usd": cost_matrix.fp_cost_usd,
        },
        "thresholds": {
            "default": 0.5,
            "f1_optimal": f1_threshold.threshold,
            "cost_optimal": cost_threshold.threshold,
            "production": production_threshold,
        },
        "test_metrics_at_production_threshold": champion_test_metrics,
        "cost_breakdown": cost_breakdown,
        "production_threshold_summary": threshold_summary,
        "bootstrap_test_auc": bootstrap_results[champion_name],
        "segment_audit_min_auc": float(audit_df["auc"].min())
        if not audit_df.empty
        else None,
        "segment_audit_path": str(SEGMENT_AUDIT_PATH.relative_to(REPORTS_DIR.parent)),
        "leaderboard_path": str(LEADERBOARD_PATH.relative_to(REPORTS_DIR.parent)),
    }

    with CHAMPION_METADATA_PATH.open("w", encoding="utf-8") as fp:
        json.dump(champion_metadata, fp, indent=2)
    with DECISION_THRESHOLD_PATH.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "threshold": production_threshold,
                "objective": "cost_minimization",
                "fn_cost_usd": cost_matrix.fn_cost_usd,
                "fp_cost_usd": cost_matrix.fp_cost_usd,
            },
            fp,
            indent=2,
        )

    joblib.dump(champion_pipeline, CHAMPION_PIPELINE_PATH)

    print(
        f"Champion: {champion_name} | "
        f"test ROC-AUC={champion_metadata['selection_score']:.4f} | "
        f"prod_threshold={production_threshold:.3f} | "
        f"min segment AUC={champion_metadata['segment_audit_min_auc']}"
    )


if __name__ == "__main__":
    main()
