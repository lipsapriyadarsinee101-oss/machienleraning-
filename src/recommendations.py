"""Transparent business rules layered on top of model risk scores."""

import pandas as pd


def risk_band(probability: float) -> str:
    if probability >= 0.80:
        return "Critical"
    if probability >= 0.60:
        return "High"
    if probability >= 0.30:
        return "Medium"
    return "Low"


def recommended_action(row: pd.Series) -> str:
    probability = float(row["churn_probability"])
    if probability >= 0.80:
        return "Urgent retention call and tailored offer"
    if row["payment_delays_12m"] >= 2:
        return "Offer billing support or flexible payment plan"
    if row["support_tickets_90d"] >= 3:
        return "Escalate unresolved service issues"
    if row["days_since_last_activity"] >= 21:
        return "Send re-engagement journey"
    if probability >= 0.60:
        return "Proactive account-manager outreach"
    if probability >= 0.30:
        return "Send personalized content recommendation"
    return "Maintain regular engagement"

