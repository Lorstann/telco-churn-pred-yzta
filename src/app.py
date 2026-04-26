"""FastAPI service exposing the production telco churn model."""
from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.config import API_TITLE, API_VERSION
from src.inference import (
    Artifacts,
    load_artifacts,
    predict_records,
    recommend_action,
    reset_cache,
)


app = FastAPI(title=API_TITLE, version=API_VERSION)


class CustomerPayload(BaseModel):
    """Strict customer schema matching the Kaggle dataset columns."""

    customerID: str | None = None
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    # Some raw files contain blank TotalCharges (new customers). We allow null
    # and let the model pipeline imputer handle it.
    TotalCharges: float | None = Field(default=None, ge=0)

    @field_validator(
        "customerID",
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    )
    @classmethod
    def non_empty_text(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class BatchPayload(BaseModel):
    """Container for batch scoring."""

    customers: list[CustomerPayload] = Field(..., min_length=1, max_length=500)


def _safe_artifacts() -> Artifacts:
    try:
        return load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/")
def root() -> dict[str, str]:
    """Root endpoint for platform health visibility."""
    return {
        "service": "telco-churn-api",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
    }


def _strip_input_fields(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for entry in results:
        entry.pop("input", None)
    return results


def _normalize_missing_values(record: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        if pd.isna(value):
            normalized[key] = None
        else:
            normalized[key] = value
    return normalized


def _risk_band(probability: float, threshold: float) -> str:
    if probability >= max(threshold, 0.5):
        return "high"
    if probability >= 0.5 * threshold:
        return "medium"
    return "low"


def _apply_threshold_override(
    results: list[dict[str, Any]],
    threshold_override: float | None,
) -> list[dict[str, Any]]:
    if threshold_override is None:
        return results
    for row in results:
        probability = float(row["probability_churn"])
        row["threshold"] = float(threshold_override)
        row["prediction"] = "Churn" if probability >= threshold_override else "No Churn"
        row["risk_band"] = _risk_band(probability, threshold_override)
    return results


async def _parse_uploaded_records(upload: UploadFile) -> list[dict[str, Any]]:
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    file_name = (upload.filename or "upload").lower()
    if file_name.endswith(".csv"):
        try:
            frame = pd.read_csv(io.BytesIO(content))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}") from exc
        if frame.empty:
            raise HTTPException(status_code=400, detail="CSV file has no rows.")
        return frame.to_dict(orient="records")

    if file_name.endswith(".json"):
        try:
            payload = json.loads(content.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {exc}") from exc
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list) and payload and all(isinstance(x, dict) for x in payload):
            return payload
        raise HTTPException(
            status_code=400,
            detail="JSON file must contain an object or a non-empty list of objects.",
        )

    raise HTTPException(status_code=415, detail="Only .csv and .json files are supported.")


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe + light artifact check."""
    try:
        artifacts = load_artifacts()
        model_loaded = True
        metadata = {
            "model_name": artifacts.metadata.get("model_name"),
            "version": artifacts.metadata.get("version"),
            "trained_at": artifacts.metadata.get("trained_at"),
            "threshold": artifacts.threshold,
        }
    except FileNotFoundError as exc:
        model_loaded = False
        metadata = {"error": str(exc)}
    return {"status": "ok", "model_loaded": model_loaded, "metadata": metadata}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    """Return the full champion metadata document persisted by training."""
    artifacts = _safe_artifacts()
    return {
        "model": artifacts.metadata,
        "threshold": artifacts.threshold,
        "title": API_TITLE,
        "version": API_VERSION,
    }


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile | None = File(default=None),
    threshold_override: float | None = Query(default=None, ge=0.0, le=1.0),
) -> dict[str, Any]:
    """Score customers from JSON body or uploaded CSV/JSON file.

    Modes:
      - `application/json` body: single customer payload.
      - `multipart/form-data` with `file`: batch scoring for `.csv`/`.json`.
    """
    artifacts = _safe_artifacts()

    if file is None:
        try:
            raw_payload = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Provide a valid JSON payload or upload a .csv/.json file.",
            ) from exc
        try:
            validated = CustomerPayload.model_validate(raw_payload).model_dump()
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        result = predict_records(validated, artifacts=artifacts)[0]
        _apply_threshold_override([result], threshold_override)
        result.pop("input", None)
        return {"count": 1, "predictions": [result]}

    records = await _parse_uploaded_records(file)
    cleaned_records = [_normalize_missing_values(record) for record in records]
    try:
        results = predict_records(cleaned_records, artifacts=artifacts)
        _apply_threshold_override(results, threshold_override)
    except KeyError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Uploaded file is missing required columns: {exc}. "
                "Include all raw telco fields used by training."
            ),
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Uploaded file contains invalid values: {exc}",
        ) from exc
    return {"count": len(results), "predictions": _strip_input_fields(results)}


@app.post("/predict/batch")
def predict_batch(
    payload: BatchPayload,
    threshold_override: float | None = Query(default=None, ge=0.0, le=1.0),
) -> dict[str, Any]:
    """Score up to 500 customers in one request."""
    artifacts = _safe_artifacts()
    records = [c.model_dump() for c in payload.customers]
    results = predict_records(records, artifacts=artifacts)
    _apply_threshold_override(results, threshold_override)
    return {"count": len(results), "predictions": _strip_input_fields(results)}


@app.post("/recommend")
def recommend(
    payload: CustomerPayload,
    threshold_override: float | None = Query(default=None, ge=0.0, le=1.0),
) -> dict[str, Any]:
    """Return prediction + retention actions/rationale for a customer."""
    artifacts = _safe_artifacts()
    prediction = predict_records(payload.model_dump(), artifacts=artifacts)[0]
    _apply_threshold_override([prediction], threshold_override)
    return recommend_action(prediction)


@app.post("/admin/reload")
def admin_reload() -> dict[str, str]:
    """Force-reload the joblib pipeline + threshold without restarting."""
    reset_cache()
    artifacts = _safe_artifacts()
    return {
        "status": "reloaded",
        "model_name": str(artifacts.metadata.get("model_name", "unknown")),
        "version": str(artifacts.metadata.get("version", "unknown")),
    }
