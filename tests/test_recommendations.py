import pandas as pd

from src.recommendations import recommended_action, risk_band


def test_risk_band_boundaries():
    assert risk_band(0.29) == "Low"
    assert risk_band(0.30) == "Medium"
    assert risk_band(0.60) == "High"
    assert risk_band(0.80) == "Critical"


def test_critical_customer_gets_urgent_action():
    row = pd.Series(
        {
            "churn_probability": 0.88,
            "payment_delays_12m": 0,
            "support_tickets_90d": 0,
            "days_since_last_activity": 2,
        }
    )
    assert recommended_action(row) == "Urgent retention call and tailored offer"


def test_payment_problem_has_specific_action():
    row = pd.Series(
        {
            "churn_probability": 0.55,
            "payment_delays_12m": 3,
            "support_tickets_90d": 0,
            "days_since_last_activity": 2,
        }
    )
    assert recommended_action(row) == "Offer billing support or flexible payment plan"
