"""Streamlit dashboard wrapping the FastAPI churn service."""
from __future__ import annotations

import json

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Telco Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

_BAND_COLORS = {"low": "#2E7D32", "medium": "#F9A825", "high": "#C62828"}


def _extract_http_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            detail = exc.response.json().get("detail")
            return f"{exc} | detail: {detail}"
        except Exception:
            return f"{exc} | response: {exc.response.text}"
    return str(exc)


def _render_risk_card(probability: float, band: str, prediction: str) -> None:
    color = _BAND_COLORS.get(band, "#1f77b4")
    st.markdown(
        f"""
        <div style="border:1px solid {color};border-radius:12px;padding:18px;">
            <h3 style="margin:0;color:{color};">{prediction}</h3>
            <p style="margin:8px 0 0 0;font-size:1.1rem;">
                Churn probability: <b>{probability:.2%}</b> (risk band: <b>{band}</b>)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_payload() -> dict[str, object]:
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    with col2:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multi = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    with col3:
        support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.0)
        total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=850.0)
    return {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": sec,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }


st.title("📡 Telco Churn Intelligence Studio")
st.caption("Production scoring, recommendations, and batch file scoring from one UI.")

api_base = st.sidebar.text_input("FastAPI base URL", value="http://127.0.0.1:8000")
predict_url = f"{api_base.rstrip('/')}/predict"
recommend_url = f"{api_base.rstrip('/')}/recommend"
metadata_url = f"{api_base.rstrip('/')}/metadata"

with st.sidebar:
    st.markdown("### Model Status")
    selected_threshold: float | None = None
    try:
        meta_resp = requests.get(metadata_url, timeout=5)
        meta_resp.raise_for_status()
        meta = meta_resp.json()
        model_meta = meta.get("model", {})
        thresholds = model_meta.get("thresholds", {})
        st.success("API connected")
        st.metric("Model", model_meta["model_name"])
        st.metric("Threshold", f"{meta['threshold']:.3f}")
        st.caption(f"Version: {model_meta.get('version', 'unknown')}")

        fallback_threshold = float(meta["threshold"])
        selected_threshold = float(
            st.slider(
                "Decision Threshold",
                min_value=0.01,
                max_value=0.99,
                value=float(fallback_threshold),
                step=0.01,
                help="Kaydırarak churn karar eşiğini manuel belirleyin.",
            )
        )
        if isinstance(thresholds, dict):
            st.caption(
                "Referanslar — "
                f"Production: {float(thresholds.get('production', fallback_threshold)):.3f} | "
                f"F1 Optimal: {float(thresholds.get('f1_optimal', fallback_threshold)):.3f} | "
                f"Cost Optimal: {float(thresholds.get('cost_optimal', fallback_threshold)):.3f}"
            )
        st.caption(f"Aktif threshold: {selected_threshold:.2f}")
    except Exception as exc:
        st.error(f"Metadata endpoint unreachable: {exc}")
        selected_threshold = None

single_tab, batch_tab, sample_tab = st.tabs(
    ["Single Customer", "Batch File Scoring", "Sample Templates"]
)

with single_tab:
    st.subheader("Single Customer Scoring + Retention Playbook")
    payload = _build_payload()
    c1, c2 = st.columns([1, 1])
    with c1:
        run_predict = st.button("Predict Risk", type="primary")
    with c2:
        run_recommend = st.button("Predict + Recommend", type="secondary")

    if run_predict:
        try:
            if selected_threshold is not None:
                response = requests.post(
                    predict_url,
                    params={"threshold_override": selected_threshold},
                    json=payload,
                    timeout=20,
                )
            else:
                response = requests.post(predict_url, json=payload, timeout=20)
            response.raise_for_status()
            out = response.json()["predictions"][0]
            _render_risk_card(out["probability_churn"], out["risk_band"], out["prediction"])
            st.progress(min(float(out["probability_churn"]), 1.0))
            st.json(out)
        except Exception as exc:
            st.error(f"Predict request failed: {_extract_http_error(exc)}")

    if run_recommend:
        try:
            if selected_threshold is not None:
                response = requests.post(
                    recommend_url,
                    params={"threshold_override": selected_threshold},
                    json=payload,
                    timeout=20,
                )
            else:
                response = requests.post(recommend_url, json=payload, timeout=20)
            response.raise_for_status()
            out = response.json()
            _render_risk_card(out["probability_churn"], out["risk_band"], out["prediction"])
            st.progress(min(float(out["probability_churn"]), 1.0))
            st.caption(f"Decision threshold: {out['threshold']:.3f}")
            st.markdown("#### Recommended Retention Actions")
            plan = out.get("recommendation_plan", [])
            if plan:
                for item in plan:
                    st.markdown(
                        f"**P{item['priority']} - {item['action']}**  \n"
                        f"- Why: {item['rationale']}  \n"
                        f"- Expected impact: {item['expected_impact']}  \n"
                        f"- Campaign: `{item['campaign_type']}`"
                    )
            else:
                for action, why in zip(out["actions"], out["rationale"]):
                    st.write(f"- **{action}** _(why: {why})_")
        except Exception as exc:
            st.error(f"Recommend request failed: {_extract_http_error(exc)}")

with batch_tab:
    st.subheader("Upload CSV/JSON for Batch Scoring")
    st.caption("Upload a `.csv` or `.json` file with the raw telco feature columns.")
    uploaded = st.file_uploader("Batch file", type=["csv", "json"])
    if st.button("Run Batch Predict", type="primary", disabled=uploaded is None):
        try:
            response = requests.post(
                predict_url,
                params=(
                    {"threshold_override": selected_threshold}
                    if selected_threshold is not None
                    else None
                ),
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                timeout=60,
            )
            response.raise_for_status()
            out = response.json()
            rows = out["predictions"]
            batch_df = pd.DataFrame(rows)
            st.success(f"Scored {out['count']} records")
            st.dataframe(batch_df, use_container_width=True, hide_index=True)

            summary = batch_df["risk_band"].value_counts().rename_axis("risk_band").reset_index(name="count")
            st.bar_chart(summary.set_index("risk_band"))

            csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions CSV",
                data=csv_bytes,
                file_name="churn_predictions.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Batch scoring failed: {_extract_http_error(exc)}")

with sample_tab:
    st.subheader("Ready-to-use Templates")
    sample_record = {
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
    st.code(json.dumps(sample_record, indent=2), language="json")
    sample_df = pd.DataFrame([sample_record, sample_record])
    st.download_button(
        "Download sample CSV",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_churn_payload.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download sample JSON",
        data=json.dumps([sample_record, sample_record], indent=2).encode("utf-8"),
        file_name="sample_churn_payload.json",
        mime="application/json",
    )
