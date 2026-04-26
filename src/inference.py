"""Inference layer shared by the FastAPI service, Streamlit UI, and tests.

Loads model artifacts produced by :mod:`src.train`, runs the same feature
engineering pipeline used in training, and converts probabilities into
business-facing recommendations using the personas surfaced in
notebook 04.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CHAMPION_METADATA_PATH,
    CHAMPION_PIPELINE_PATH,
    DECISION_THRESHOLD_PATH,
    DEFAULT_THRESHOLD,
)
from src.preprocessing import prepare_inference_payload


@dataclass
class Artifacts:
    """Bundle the production-time artifacts produced by ``src.train``."""

    model: Any
    threshold: float
    metadata: dict[str, Any]


_ARTIFACTS_CACHE: Artifacts | None = None


@dataclass
class RecommendationItem:
    """Single retention action with business context."""

    priority: int
    action: str
    rationale: str
    expected_impact: str
    campaign_type: str


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_artifacts(
    *,
    model_path: Path = CHAMPION_PIPELINE_PATH,
    threshold_path: Path = DECISION_THRESHOLD_PATH,
    metadata_path: Path = CHAMPION_METADATA_PATH,
    refresh: bool = False,
) -> Artifacts:
    """Load (and cache) the champion pipeline, threshold, and metadata.

    Pass ``refresh=True`` after retraining to reload artifacts without
    restarting the host process.
    """
    global _ARTIFACTS_CACHE
    if _ARTIFACTS_CACHE is not None and not refresh:
        return _ARTIFACTS_CACHE

    if not model_path.exists():
        raise FileNotFoundError(
            f"Champion pipeline missing at {model_path}. Run `python -m src.train` first."
        )
    model = joblib.load(model_path)

    threshold_payload = _read_json(threshold_path, {"threshold": DEFAULT_THRESHOLD})
    threshold = float(threshold_payload.get("threshold", DEFAULT_THRESHOLD))

    metadata = _read_json(metadata_path, {})
    metadata.setdefault("threshold_payload", threshold_payload)

    _ARTIFACTS_CACHE = Artifacts(model=model, threshold=threshold, metadata=metadata)
    return _ARTIFACTS_CACHE


def reset_cache() -> None:
    """Clear the cached artifacts (used by tests and `/admin/reload`)."""
    global _ARTIFACTS_CACHE
    _ARTIFACTS_CACHE = None


def _risk_band(probability: float, threshold: float) -> str:
    if probability >= max(threshold, 0.5):
        return "high"
    if probability >= 0.5 * threshold:
        return "medium"
    return "low"


def _extract_customer_id(raw: Mapping[str, Any]) -> str | None:
    for key in ("customerID", "customer_id", "CustomerID"):
        value = raw.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def predict_records(
    records: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    artifacts: Artifacts | None = None,
) -> list[dict[str, Any]]:
    """Score one or many raw payload(s) and return enriched predictions."""
    artifacts = artifacts or load_artifacts()
    engineered = prepare_inference_payload(records)
    probabilities = artifacts.model.predict_proba(engineered)[:, 1]

    raw_records = (
        [records] if isinstance(records, Mapping) else list(records)
    )
    output: list[dict[str, Any]] = []
    for raw, prob in zip(raw_records, probabilities):
        prob_f = float(prob)
        decision = "Churn" if prob_f >= artifacts.threshold else "No Churn"
        output.append(
            {
                "customer_id": _extract_customer_id(raw),
                "prediction": decision,
                "probability_churn": prob_f,
                "threshold": artifacts.threshold,
                "risk_band": _risk_band(prob_f, artifacts.threshold),
                "model_name": artifacts.metadata.get("model_name", "unknown"),
                "model_version": artifacts.metadata.get(
                    "version", artifacts.metadata.get("trained_at", "unknown")
                ),
                "input": dict(raw),
            }
        )
    return output


def recommend_action(prediction: dict[str, Any]) -> dict[str, Any]:
    """Map a scored prediction to a retention action.

    The action catalog mirrors the personas and counterfactuals in
    notebook 04 (Section 11): contract upgrade, autopay nudge,
    fiber-quality outreach, and senior-citizen concierge support.
    """
    payload = prediction["input"]
    probability = float(prediction["probability_churn"])
    recommendations: list[RecommendationItem] = []

    contract = str(payload.get("Contract", ""))
    internet = str(payload.get("InternetService", ""))
    payment = str(payload.get("PaymentMethod", "")).lower()
    paperless = str(payload.get("PaperlessBilling", ""))
    monthly = float(payload.get("MonthlyCharges", 0.0) or 0.0)
    tenure = float(payload.get("tenure", 0.0) or 0.0)
    senior = int(payload.get("SeniorCitizen", 0) or 0)
    partner = str(payload.get("Partner", ""))
    dependents = str(payload.get("Dependents", ""))

    if contract == "Month-to-month":
        recommendations.append(
            RecommendationItem(
                priority=1,
                action="Convert to annual contract (10-15% first-year discount)",
                rationale="Month-to-month plan is the strongest structural churn driver.",
                expected_impact="High - reduces voluntary churn in first 90 days.",
                campaign_type="contract_migration",
            )
        )
    if internet.lower() == "fiber optic":
        recommendations.append(
            RecommendationItem(
                priority=1,
                action="Proactive fiber quality outreach + technician health-check",
                rationale="Fiber segment historically has the highest churn pressure.",
                expected_impact="High - improves experience before cancellation intent.",
                campaign_type="service_recovery",
            )
        )
    if paperless == "Yes" and "automatic" not in payment:
        recommendations.append(
            RecommendationItem(
                priority=2,
                action="Autopay enrollment with one-time bill credit",
                rationale="Paperless + non-automatic payment increases payment friction.",
                expected_impact="Medium/High - lowers missed-payment churn triggers.",
                campaign_type="payment_optimization",
            )
        )
    if monthly > 80 and tenure < 12:
        recommendations.append(
            RecommendationItem(
                priority=2,
                action="Price-protection bundle for first-year premium customers",
                rationale="High spenders in early tenure are highly price-sensitive.",
                expected_impact="Medium/High - stabilizes high-ARPU accounts.",
                campaign_type="pricing_retention",
            )
        )
    if tenure <= 3:
        recommendations.append(
            RecommendationItem(
                priority=1,
                action="First-90-day onboarding success call + setup checklist",
                rationale="Very new customers churn disproportionately during onboarding.",
                expected_impact="High - early intervention prevents fast drop-off.",
                campaign_type="onboarding",
            )
        )
    if tenure >= 24 and contract != "Two year":
        recommendations.append(
            RecommendationItem(
                priority=3,
                action="Loyalty upgrade offer (speed bump or streaming add-on)",
                rationale="Long-tenure accounts respond well to recognition rewards.",
                expected_impact="Medium - boosts stickiness with low discount cost.",
                campaign_type="loyalty",
            )
        )
    if senior == 1 and partner == "No" and dependents == "No":
        recommendations.append(
            RecommendationItem(
                priority=1,
                action="Senior concierge support lane + monthly check-in",
                rationale="Senior customers living alone require proactive support touchpoints.",
                expected_impact="High - prevents silent dissatisfaction churn.",
                campaign_type="care_support",
            )
        )
    if "electronic check" in payment:
        recommendations.append(
            RecommendationItem(
                priority=2,
                action="Migrate from electronic check to automatic payment",
                rationale="Electronic check cohort is repeatedly overrepresented in churn.",
                expected_impact="Medium - improves payment consistency and retention.",
                campaign_type="payment_optimization",
            )
        )
    if recommendations:
        recommendations.sort(key=lambda item: (item.priority, item.action))
    else:
        recommendations.append(
            RecommendationItem(
                priority=4,
                action="No aggressive intervention; monitor with monthly risk scoring",
                rationale="Current profile is within healthy churn range.",
                expected_impact="Low - keep normal engagement cadence.",
                campaign_type="monitoring",
            )
        )

    # Keep backward compatibility for UI and tests while returning richer plan.
    actions = [item.action for item in recommendations]
    rationale = [item.rationale for item in recommendations]

    return {
        "customer_id": prediction.get("customer_id"),
        "prediction": prediction["prediction"],
        "probability_churn": probability,
        "threshold": prediction["threshold"],
        "risk_band": prediction["risk_band"],
        "actions": actions,
        "rationale": rationale,
        "recommendation_plan": [
            {
                "priority": item.priority,
                "action": item.action,
                "rationale": item.rationale,
                "expected_impact": item.expected_impact,
                "campaign_type": item.campaign_type,
            }
            for item in recommendations
        ],
    }
