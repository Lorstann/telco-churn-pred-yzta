"""Centralized configuration for the telco churn pipeline.

Single source of truth for paths, business cost matrix, champion-selection
gates, and segment dimensions. Imported by training, evaluation, inference,
and the API service so every layer uses the same values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEFAULT_DATA_PATH = RAW_DATA_DIR / "telco.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CURVES_DIR = REPORTS_DIR / "model_curves"

CHAMPION_PIPELINE_PATH = MODELS_DIR / "champion_pipeline.joblib"
CHAMPION_METADATA_PATH = MODELS_DIR / "champion_metadata.json"
DECISION_THRESHOLD_PATH = MODELS_DIR / "decision_threshold.json"
LEADERBOARD_PATH = REPORTS_DIR / "model_metrics_validation.csv"
PREPROCESSING_REPORT_PATH = REPORTS_DIR / "preprocessing_report.json"
SEGMENT_AUDIT_PATH = REPORTS_DIR / "segment_audit.csv"
PERMUTATION_IMPORTANCE_PATH = REPORTS_DIR / "permutation_importance.csv"
BOOTSTRAP_CI_PATH = REPORTS_DIR / "bootstrap_ci.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALID_SIZE = 0.15
TRAIN_SIZE = 1.0 - TEST_SIZE - VALID_SIZE


@dataclass(frozen=True)
class CostMatrix:
    """Business cost of a confusion-matrix outcome.

    The default values were established in notebook 03 (Section 6) from the
    EDA-driven revenue-at-risk analysis: missing a churner costs roughly
    a year of MRR (~$800) while contacting a non-churner only costs the
    retention campaign budget (~$50). Override via `from_env` at runtime.
    """

    fn_cost_usd: float = 800.0
    fp_cost_usd: float = 50.0

    @property
    def ratio(self) -> float:
        return self.fn_cost_usd / max(self.fp_cost_usd, 1e-9)


@dataclass(frozen=True)
class ChampionGate:
    """Multi-criteria gate for declaring a champion model.

    Replicates the rule applied in notebook 03 (Section 12): a model passes
    only if its calibration is comparable to the leader, every business
    segment exceeds a minimum AUC, and bootstrap stability is acceptable.
    """

    calibration_multiplier: float = 1.5
    min_segment_auc: float = 0.70
    max_bootstrap_std: float = 0.02


@dataclass(frozen=True)
class SegmentSpec:
    """Definition of a segmentation dimension used in audits."""

    name: str
    column: str
    bins: tuple[float, ...] | None = None
    labels: tuple[str, ...] | None = None


SEGMENT_SPECS: tuple[SegmentSpec, ...] = (
    SegmentSpec(name="Contract", column="Contract"),
    SegmentSpec(name="InternetService", column="InternetService"),
    SegmentSpec(name="PaymentMethod", column="PaymentMethod"),
    SegmentSpec(
        name="TenureBucket",
        column="tenure",
        bins=(-0.1, 6.0, 24.0, 72.0),
        labels=("0-6m", "7-24m", "25+m"),
    ),
)

ARPU_HORIZON_MONTHS = 12

DEFAULT_THRESHOLD = 0.5

API_TITLE = "Telco Churn API"
API_VERSION = "1.1.0"


@dataclass
class RuntimeConfig:
    """Runtime container that bundles configuration values.

    Use this when a function needs many config values; fields can be
    overridden in tests without monkey-patching module-level constants.
    """

    cost_matrix: CostMatrix = field(default_factory=CostMatrix)
    champion_gate: ChampionGate = field(default_factory=ChampionGate)
    random_state: int = RANDOM_STATE
    test_size: float = TEST_SIZE
    valid_size: float = VALID_SIZE


def ensure_directories() -> None:
    """Create writable output directories on demand."""
    for directory in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
                      REPORTS_DIR, FIGURES_DIR, CURVES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
