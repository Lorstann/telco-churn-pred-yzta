"""Smoke tests for the FastAPI churn service."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from src.app import app
from src.config import CHAMPION_PIPELINE_PATH

client = TestClient(app)


SAMPLE_PAYLOAD = {
    "customerID": "0001-ABCD",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 1100.0,
}


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_predict_validation_error() -> None:
    response = client.post("/predict", json={"gender": "Male"})
    assert response.status_code == 422


def test_predict_happy_path_when_model_present() -> None:
    if not CHAMPION_PIPELINE_PATH.exists():
        return
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()["predictions"][0]
    assert body["customer_id"] == "0001-ABCD"
    assert body["prediction"] in {"Churn", "No Churn"}
    assert 0.0 <= body["probability_churn"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}


def test_recommend_endpoint() -> None:
    if not CHAMPION_PIPELINE_PATH.exists():
        return
    response = client.post("/recommend", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == "0001-ABCD"
    assert isinstance(body["actions"], list) and body["actions"]
    assert len(body["actions"]) == len(body["rationale"])


def test_predict_batch_endpoint() -> None:
    if not CHAMPION_PIPELINE_PATH.exists():
        return
    response = client.post(
        "/predict/batch",
        json={"customers": [SAMPLE_PAYLOAD, SAMPLE_PAYLOAD]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_metadata_endpoint() -> None:
    if not CHAMPION_PIPELINE_PATH.exists():
        return
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert "model" in body and "threshold" in body


def test_predict_csv_upload() -> None:
    if not CHAMPION_PIPELINE_PATH.exists():
        return
    csv_content = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,"
        "InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,"
        "StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,"
        "MonthlyCharges,TotalCharges\n"
        "0001-ABCD,Female,0,Yes,No,12,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,"
        "Yes,Electronic check,95.5,1100.0\n"
        "0002-EFGH,Male,1,No,No,3,Yes,Yes,DSL,No,Yes,No,No,No,No,Month-to-month,"
        "No,Mailed check,55.2,220.8\n"
    )
    response = client.post(
        "/predict",
        files={"file": ("sample.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert body["predictions"][0]["customer_id"] == "0001-ABCD"


def test_predict_json_upload() -> None:
    if not CHAMPION_PIPELINE_PATH.exists():
        return
    response = client.post(
        "/predict",
        files={
            "file": (
                "sample.json",
                json.dumps(
                    [
                        SAMPLE_PAYLOAD,
                        {
                            "gender": "Male",
                            "customerID": "0002-EFGH",
                            "SeniorCitizen": 1,
                            "Partner": "No",
                            "Dependents": "No",
                            "tenure": 3,
                            "PhoneService": "Yes",
                            "MultipleLines": "Yes",
                            "InternetService": "DSL",
                            "OnlineSecurity": "No",
                            "OnlineBackup": "Yes",
                            "DeviceProtection": "No",
                            "TechSupport": "No",
                            "StreamingTV": "No",
                            "StreamingMovies": "No",
                            "Contract": "Month-to-month",
                            "PaperlessBilling": "No",
                            "PaymentMethod": "Mailed check",
                            "MonthlyCharges": 55.2,
                            "TotalCharges": 220.8,
                        },
                    ]
                ),
                "application/json",
            )
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert body["predictions"][1]["customer_id"] == "0002-EFGH"


def test_predict_csv_upload_with_blank_total_charges() -> None:
    if not CHAMPION_PIPELINE_PATH.exists():
        return
    csv_content = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,"
        "InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,"
        "StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,"
        "MonthlyCharges,TotalCharges\n"
        "0003-IJKL,Female,0,Yes,No,1,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,"
        "Yes,Electronic check,70.0,\n"
    )
    response = client.post(
        "/predict",
        files={"file": ("blank_totalcharges.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["predictions"][0]["customer_id"] == "0003-IJKL"
    assert body["predictions"][0]["prediction"] in {"Churn", "No Churn"}


def test_predict_csv_upload_missing_required_column_returns_422() -> None:
    if not CHAMPION_PIPELINE_PATH.exists():
        return
    # Intentionally omit PaymentMethod to validate error messaging.
    csv_content = (
        "gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,"
        "InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,"
        "StreamingTV,StreamingMovies,Contract,PaperlessBilling,MonthlyCharges,TotalCharges\n"
        "Female,0,Yes,No,1,Yes,No,Fiber optic,No,No,No,No,Yes,Yes,Month-to-month,Yes,70.0,\n"
    )
    response = client.post(
        "/predict",
        files={"file": ("missing_col.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 422


def test_predict_threshold_override_changes_response_threshold() -> None:
    if not CHAMPION_PIPELINE_PATH.exists():
        return
    response = client.post("/predict?threshold_override=0.9", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()["predictions"][0]
    assert abs(body["threshold"] - 0.9) < 1e-9
