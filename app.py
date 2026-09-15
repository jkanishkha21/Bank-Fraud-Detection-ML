"""
Streamlit interface for the Bank Fraud Detection project.
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "models/fraud_pipeline.joblib"
METADATA_PATH = "models/model_metadata.json"
DATA_PATH = "data/transactions.csv"

st.set_page_config(
    page_title="Bank Fraud Detection",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Bank Fraud Detection System")
st.caption("Machine-learning based transaction risk screening")

if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model not found. Run `python generate_dataset.py` and "
        "`python train_model.py` before starting the app."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

threshold = metadata["threshold"]

st.info(
    "This project is a demonstration system using synthetic transaction data. "
    "It is not intended for real banking decisions."
)

tab1, tab2 = st.tabs(["🔍 Check Transaction", "📊 Dataset Overview"])

with tab1:
    st.subheader("Enter transaction details")

    col1, col2, col3 = st.columns(3)

    with col1:
        amount = st.number_input(
            "Transaction amount",
            min_value=1.0,
            max_value=5000.0,
            value=250.0,
            step=10.0,
        )
        hour = st.slider("Transaction hour", 0, 23, 14)
        transaction_type = st.selectbox(
            "Transaction type",
            ["POS", "Online", "ATM", "Bank Transfer"],
        )

    with col2:
        merchant_category = st.selectbox(
            "Merchant category",
            [
                "Grocery",
                "Travel",
                "Electronics",
                "Food",
                "Entertainment",
                "Utilities",
            ],
        )
        country = st.selectbox(
            "Country",
            ["India", "USA", "UK", "Singapore", "UAE"],
        )
        device_risk_score = st.slider(
            "Device risk score",
            0.0,
            1.0,
            0.20,
            0.01,
        )

    with col3:
        ip_risk_score = st.slider(
            "IP risk score",
            0.0,
            1.0,
            0.20,
            0.01,
        )
        account_age_days = st.number_input(
            "Account age (days)",
            min_value=1,
            max_value=3000,
            value=365,
        )
        previous_transactions = st.number_input(
            "Previous transactions",
            min_value=0,
            max_value=100,
            value=10,
        )

    international = int(country != "India")

    input_df = pd.DataFrame(
        [
            {
                "amount": amount,
                "hour": hour,
                "transaction_type": transaction_type,
                "merchant_category": merchant_category,
                "country": country,
                "device_risk_score": device_risk_score,
                "ip_risk_score": ip_risk_score,
                "account_age_days": account_age_days,
                "previous_transactions": previous_transactions,
                "international": international,
            }
        ]
    )

    if st.button("Analyze Transaction", type="primary", use_container_width=True):
        probability = float(model.predict_proba(input_df)[0, 1])
        prediction = int(probability >= threshold)

        st.divider()

        if prediction == 1:
            st.error(
                f"⚠️ Potential Fraud Detected — risk probability: "
                f"{probability:.1%}"
            )
            st.write(
                "Recommended action: flag the transaction for additional review."
            )
        else:
            st.success(
                f"✅ Transaction Appears Normal — risk probability: "
                f"{probability:.1%}"
            )
            st.write("Recommended action: allow transaction subject to policy.")

        st.metric("Fraud Risk Probability", f"{probability:.1%}")
        st.caption(
            f"Decision threshold used by the trained model: {threshold:.2f}"
        )

with tab2:
    st.subheader("Synthetic dataset overview")

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)

        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions", f"{len(df):,}")
        c2.metric("Fraud Cases", f"{int(df['is_fraud'].sum()):,}")
        c3.metric("Fraud Rate", f"{df['is_fraud'].mean():.2%}")

        st.write("Sample records")
        st.dataframe(df.head(20), use_container_width=True)

        st.write("Fraud distribution")
        distribution = (
            df["is_fraud"]
            .value_counts()
            .rename(index={0: "Normal", 1: "Fraud"})
        )
        st.bar_chart(distribution)
    else:
        st.warning("Dataset file not found.")
