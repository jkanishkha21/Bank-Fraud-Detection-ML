"""
Sample dataset generator for the Bank Fraud Detection project.

This creates a synthetic transaction dataset so the project can be
run without downloading a private or external banking dataset.
"""

import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_SAMPLES = 5000


def generate_dataset(n_samples=N_SAMPLES, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)

    amount = np.round(rng.lognormal(mean=4.2, sigma=1.0, size=n_samples), 2)
    amount = np.clip(amount, 1, 5000)

    hour = rng.integers(0, 24, n_samples)
    transaction_type = rng.choice(
        ["POS", "Online", "ATM", "Bank Transfer"],
        size=n_samples,
        p=[0.35, 0.35, 0.15, 0.15],
    )
    merchant_category = rng.choice(
        ["Grocery", "Travel", "Electronics", "Food", "Entertainment", "Utilities"],
        size=n_samples,
    )
    country = rng.choice(
        ["India", "USA", "UK", "Singapore", "UAE"],
        size=n_samples,
        p=[0.65, 0.10, 0.08, 0.07, 0.10],
    )

    device_risk_score = np.round(rng.beta(2, 5, n_samples), 3)
    ip_risk_score = np.round(rng.beta(2, 5, n_samples), 3)
    account_age_days = rng.integers(1, 3000, n_samples)
    previous_transactions = rng.poisson(12, n_samples)
    international = (country != "India").astype(int)

    # Synthetic fraud signal:
    # larger transactions, unusual hours, high risk scores,
    # young accounts, and international activity increase risk.
    logit = (
        -6.2
        + 0.00055 * amount
        + 2.4 * device_risk_score
        + 2.0 * ip_risk_score
        + 0.9 * international
        + 0.8 * (account_age_days < 60)
        + 0.7 * ((hour <= 4) | (hour >= 23))
        + 0.45 * (transaction_type == "Online")
        + 0.35 * (merchant_category == "Electronics")
        + rng.normal(0, 0.7, n_samples)
    )

    fraud_probability = 1 / (1 + np.exp(-logit))
    is_fraud = rng.binomial(1, fraud_probability)

    # Keep fraud rare enough to resemble the highly imbalanced setting
    # described in the internship project brief.
    if is_fraud.mean() > 0.08:
        threshold = np.quantile(fraud_probability, 0.92)
        is_fraud = (fraud_probability >= threshold).astype(int)

    df = pd.DataFrame(
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
            "is_fraud": is_fraud,
        }
    )

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    output_path = "data/transactions.csv"
    df.to_csv(output_path, index=False)

    print(f"Dataset created: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Fraud cases: {df['is_fraud'].sum()}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
