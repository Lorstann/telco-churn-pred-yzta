"""Telco churn package: production training, evaluation, and inference."""
from __future__ import annotations

__version__ = "1.1.0"

from src.config import (
    CHAMPION_METADATA_PATH,
    CHAMPION_PIPELINE_PATH,
    DECISION_THRESHOLD_PATH,
    DEFAULT_DATA_PATH,
    CostMatrix,
    ChampionGate,
    SEGMENT_SPECS,
)
from src.evaluation import (
    bootstrap_auc_ci,
    cost_optimal_threshold,
    cumulative_gain_curve,
    evaluate_segments,
    expected_business_cost,
    f1_optimal_threshold,
    top_decile_lift,
)
from src.explainability import (
    linear_coefficient_table,
    permutation_importance_table,
    tree_shap_importance_table,
)
from src.inference import (
    Artifacts,
    load_artifacts,
    predict_records,
    recommend_action,
    reset_cache,
)
from src.models import (
    compute_scale_pos_weight,
    evaluate_predictions,
    get_model_registry,
)
from src.preprocessing import (
    ENGINEERED_COLUMNS,
    RAW_REQUIRED_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    coerce_total_charges,
    engineer_features,
    load_data,
    prepare_inference_payload,
    split_features_target,
)

__all__ = [
    "__version__",
    "Artifacts",
    "ChampionGate",
    "CostMatrix",
    "DEFAULT_DATA_PATH",
    "CHAMPION_METADATA_PATH",
    "CHAMPION_PIPELINE_PATH",
    "DECISION_THRESHOLD_PATH",
    "ENGINEERED_COLUMNS",
    "RAW_REQUIRED_COLUMNS",
    "SEGMENT_SPECS",
    "TARGET_COLUMN",
    "bootstrap_auc_ci",
    "build_preprocessor",
    "coerce_total_charges",
    "compute_scale_pos_weight",
    "cost_optimal_threshold",
    "cumulative_gain_curve",
    "engineer_features",
    "evaluate_predictions",
    "evaluate_segments",
    "expected_business_cost",
    "f1_optimal_threshold",
    "get_model_registry",
    "linear_coefficient_table",
    "load_artifacts",
    "load_data",
    "permutation_importance_table",
    "predict_records",
    "prepare_inference_payload",
    "recommend_action",
    "reset_cache",
    "split_features_target",
    "top_decile_lift",
    "tree_shap_importance_table",
]
