"""Generate a reproducible synthetic dataset for portfolio demonstration."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "raw" / "customer_data.csv"


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-value))


def generate_customer_data(rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 61, rows)
    monthly_spend_eur = np.round(rng.gamma(4.2, 18, rows) + 15, 2)
    logins_30d = rng.poisson(8, rows)
    content_completions_30d = rng.poisson(3.5, rows)
    support_tickets_90d = rng.poisson(1.2, rows)
    satisfaction_score = np.clip(rng.normal(3.8, 0.8, rows), 1, 5).round(1)
    payment_delays_12m = rng.poisson(0.7, rows)
    days_since_last_activity = np.clip(rng.gamma(2.1, 6.5, rows), 0, 90).round().astype(int)
    plan_type = rng.choice(["Basic", "Professional", "Enterprise"], rows, p=[0.48, 0.37, 0.15])
    acquisition_channel = rng.choice(["Organic", "Referral", "Paid", "Partner"], rows, p=[0.35, 0.25, 0.25, 0.15])
    region = rng.choice(["Germany", "United Kingdom", "Portugal"], rows, p=[0.48, 0.32, 0.20])

    log_odds = (
        -1.5
        - 0.025 * tenure_months
        - 0.11 * logins_30d
        - 0.16 * content_completions_30d
        - 0.55 * (satisfaction_score - 3)
        + 0.30 * support_tickets_90d
        + 0.42 * payment_delays_12m
        + 0.075 * days_since_last_activity
        + 0.35 * (plan_type == "Basic")
        + rng.normal(0, 0.55, rows)
    )
    churn_probability = sigmoid(log_odds)
    churned = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "customer_id": [f"SP-{index:05d}" for index in range(1, rows + 1)],
            "tenure_months": tenure_months,
            "monthly_spend_eur": monthly_spend_eur,
            "logins_30d": logins_30d,
            "content_completions_30d": content_completions_30d,
            "support_tickets_90d": support_tickets_90d,
            "satisfaction_score": satisfaction_score,
            "payment_delays_12m": payment_delays_12m,
            "days_since_last_activity": days_since_last_activity,
            "plan_type": plan_type,
            "acquisition_channel": acquisition_channel,
            "region": region,
            "churned": churned,
        }
    )


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = generate_customer_data()
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(data):,} synthetic customer rows to {OUTPUT_PATH}")

