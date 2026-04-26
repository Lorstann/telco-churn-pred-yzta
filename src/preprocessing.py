from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

ORDINAL_COLUMNS = ["Contract"]
ORDINAL_CATEGORIES = [["Month-to-month", "One year", "Two year"]]

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

NOMINAL_COLUMNS = [
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
    "PaperlessBilling",
    "PaymentMethod",
]

RAW_REQUIRED_COLUMNS: tuple[str, ...] = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
)

ENGINEERED_COLUMNS: tuple[str, ...] = (
    "Tenure_Years",
    "Is_Fiber_Optic",
    "Service_Count",
    "Charges_Per_Service",
    "Contract_Tenure_Interaction",
    "MonthlyCharge_Tenure_Ratio",
    "SeniorCitizen_AloneRisk",
    "Paperless_AutoPay_Flag",
)


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series


def generate_data_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """Build a lightweight data quality report for EDA diagnostics."""
    report = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_values": df.isna().sum().sort_values(ascending=False).to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "nunique": df.nunique(dropna=False).to_dict(),
        "numeric_summary": df.select_dtypes(include=np.number).describe().to_dict(),
    }
    if ID_COLUMN in df.columns:
        report["customerID_is_unique"] = bool(df[ID_COLUMN].is_unique)
    return report


def load_data(path: str) -> pd.DataFrame:
    """Load raw telco churn data from CSV."""
    return pd.read_csv(path)


def coerce_total_charges(series: pd.Series) -> pd.Series:
    """Convert the TotalCharges column to numeric.

    The Kaggle file ships TotalCharges as a string with 11 blank entries
    that pandas reads as " " or "". This helper centralizes that cleanup
    so the same logic runs in training, batch inference, and online
    scoring. NaNs are left for the median imputer to handle.
    """
    cleaned = series.astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def _service_to_binary(series: pd.Series) -> pd.Series:
    yes_like = {"yes", "fiber optic", "dsl"}
    return series.astype(str).str.lower().isin(yes_like).astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply telco-specific cleaning and feature engineering.

    Mirrors the pipeline established in notebook 02; safe to call on raw
    inference payloads as long as the schema in `RAW_REQUIRED_COLUMNS`
    is present.
    """
    features = df.copy()

    if ID_COLUMN in features.columns:
        features = features.drop(columns=[ID_COLUMN])

    if "TotalCharges" in features.columns:
        features["TotalCharges"] = coerce_total_charges(features["TotalCharges"])

    if "tenure" in features.columns:
        features["Tenure_Years"] = features["tenure"] / 12.0
    if "InternetService" in features.columns:
        features["Is_Fiber_Optic"] = (
            features["InternetService"].astype(str).str.lower().eq("fiber optic").astype(int)
        )

    service_count = np.zeros(len(features), dtype=int)
    for col in SERVICE_COLUMNS:
        if col in features.columns:
            service_count += _service_to_binary(features[col]).to_numpy()
    if "InternetService" in features.columns:
        service_count += (
            features["InternetService"]
            .astype(str)
            .str.lower()
            .isin(["dsl", "fiber optic"])
            .astype(int)
            .to_numpy()
        )

    features["Service_Count"] = np.maximum(service_count, 0)
    if "MonthlyCharges" in features.columns:
        features["Charges_Per_Service"] = (
            features["MonthlyCharges"] / np.maximum(features["Service_Count"], 1)
        )
        if "tenure" in features.columns:
            features["MonthlyCharge_Tenure_Ratio"] = features["MonthlyCharges"] / np.maximum(
                features["tenure"], 1
            )
    if "tenure" in features.columns and "Contract" in features.columns:
        features["Contract_Tenure_Interaction"] = (
            features["tenure"] * (features["Contract"] == "Two year").astype(int)
        )
    if {"SeniorCitizen", "Partner", "Dependents"}.issubset(features.columns):
        features["SeniorCitizen_AloneRisk"] = (
            (features["SeniorCitizen"] == 1)
            & (features["Partner"] == "No")
            & (features["Dependents"] == "No")
        ).astype(int)
    if {"PaperlessBilling", "PaymentMethod"}.issubset(features.columns):
        features["Paperless_AutoPay_Flag"] = (
            (features["PaperlessBilling"] == "Yes")
            & (
                features["PaymentMethod"]
                .astype(str)
                .str.contains("automatic", case=False, na=False)
            )
        ).astype(int)

    return features


def prepare_inference_payload(
    records: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Convert raw API/Streamlit payload(s) into a model-ready DataFrame.

    Validates that every column in `RAW_REQUIRED_COLUMNS` is present and
    runs the full feature-engineering pipeline so the resulting frame is
    a drop-in input for `champion_pipeline.predict_proba`.

    Raises:
        KeyError: when one or more required raw fields are missing.
    """
    if isinstance(records, Mapping):
        rows: list[Mapping[str, Any]] = [records]
    else:
        rows = list(records)
    if not rows:
        raise ValueError("No records provided for inference.")

    raw_df = pd.DataFrame(rows)
    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in raw_df.columns]
    if missing:
        raise KeyError(f"Missing required raw fields: {sorted(missing)}")

    return engineer_features(raw_df)


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> PreparedData:
    """Split features and target with standard churn encoding."""
    local_df = df.copy()
    y = local_df[target_col].map({"No": 0, "Yes": 1}).astype(int)
    X = local_df.drop(columns=[target_col])
    return PreparedData(X=X, y=y)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a leakage-safe preprocessing pipeline."""
    available_nominal = [col for col in NOMINAL_COLUMNS if col in X.columns]
    available_ordinal = [col for col in ORDINAL_COLUMNS if col in X.columns]

    numerical_columns = [
        col
        for col in X.columns
        if col not in set(available_nominal + available_ordinal)
    ]

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ORDINAL_CATEGORIES[: len(available_ordinal)],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if numerical_columns:
        transformers.append(("num", numerical_pipeline, numerical_columns))
    if available_nominal:
        transformers.append(("nom", nominal_pipeline, available_nominal))
    if available_ordinal:
        transformers.append(("ord", ordinal_pipeline, available_ordinal))

    return ColumnTransformer(transformers=transformers, remainder="drop")

