"""Train, evaluate, select, and save the churn prediction pipeline."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from recommendations import recommended_action, risk_band


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "customer_data.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
SCORES_PATH = ROOT / "data" / "processed" / "customer_risk_scores.csv"

TARGET = "churned"
ID_COLUMN = "customer_id"
NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_spend_eur",
    "logins_30d",
    "content_completions_30d",
    "support_tickets_90d",
    "satisfaction_score",
    "payment_delays_12m",
    "days_since_last_activity",
]
CATEGORICAL_FEATURES = ["plan_type", "acquisition_channel", "region"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def evaluate(y_true: pd.Series, probabilities) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    if data[FEATURES + [TARGET, ID_COLUMN]].isnull().any().any():
        print("Missing values detected; preprocessing will impute feature values.")

    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.25, random_state=42, stratify=data[TARGET]
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.07, max_iter=250, max_leaf_nodes=15, l2_regularization=0.5, random_state=42
        ),
    }
    fitted = {}
    results = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        fitted[name] = pipeline
        results[name] = evaluate(y_test, probabilities)

    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_pipeline = fitted[best_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    summary = {
        "data_note": "Synthetic portfolio demonstration data - not SpringPod production data",
        "rows": len(data),
        "test_rows": len(x_test),
        "churn_rate": round(float(data[TARGET].mean()), 4),
        "selected_model": best_name,
        "models": results,
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    scored = data.copy()
    scored["churn_probability"] = best_pipeline.predict_proba(scored[FEATURES])[:, 1]
    scored["risk_band"] = scored["churn_probability"].apply(risk_band)
    scored["recommended_action"] = scored.apply(recommended_action, axis=1)
    scored.sort_values("churn_probability", ascending=False).to_csv(SCORES_PATH, index=False)

    print(json.dumps(summary, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved scores: {SCORES_PATH}")


if __name__ == "__main__":
    main()

