import os
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware   # ✅ NEW
from fastapi.responses import JSONResponse          # ✅ NEW
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ============================================================
# 1. CONFIG
# ============================================================

MODEL_PATH = "soltrice_fraud_model_v1.pkl"
DATA_PATH = "soltrice_fraud_data.csv"

numeric_features = [
    "rental_amount",
    "rental_duration_days",
    "lead_time_hours",
    "distance_from_home_to_branch_km",
    "customer_age",
    "customer_tenure_days",
    "previous_rentals_count",
    "previous_chargebacks_count",
    "failed_attempts_last_24h",
]

categorical_features = [
    "booking_channel",
    "card_present",
    "same_name_on_card",
    "billing_matches_id",
    "ip_country_matches_id_country",
]


# ============================================================
# 2. REQUEST / RESPONSE SCHEMAS
# ============================================================

class FraudRequest(BaseModel):
    rental_amount: float = Field(..., example=120.5)
    rental_duration_days: int = Field(..., example=3)
    booking_channel: Literal["web", "app", "phone", "walk_in"] = Field(
        ..., example="web"
    )
    lead_time_hours: float = Field(..., example=48)
    distance_from_home_to_branch_km: float = Field(..., example=12.3)
    customer_age: int = Field(..., example=42)
    customer_tenure_days: int = Field(..., example=365)
    previous_rentals_count: int = Field(..., example=4)
    previous_chargebacks_count: int = Field(..., example=0)
    card_present: Literal["yes", "no"] = Field(..., example="yes")
    same_name_on_card: Literal["yes", "no"] = Field(..., example="yes")
    billing_matches_id: Literal["yes", "no"] = Field(..., example="yes")
    failed_attempts_last_24h: int = Field(..., example=0)
    ip_country_matches_id_country: Literal["yes", "no"] = Field(..., example="yes")


class FraudResponse(BaseModel):
    fraud_probability: float
    risk_bucket: Literal["Low", "Medium", "High"]
    customer_tier: Literal["Premium", "Standard", "Risky"]
    model_version: str
    auc_on_last_train: float


# ============================================================
# 3. TRAINING LOGIC
# ============================================================

def build_pipeline() -> Pipeline:
    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    clf = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )
    return clf


def train_and_save_model(data_path: str = DATA_PATH, model_path: str = MODEL_PATH):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            f"Create {data_path} with your historical/synthetic data."
        )

    df = pd.read_csv(data_path)

    if "is_fraud" not in df.columns:
        raise ValueError("Column 'is_fraud' is missing from the dataset.")

    X = df.drop(columns=["is_fraud", "customer_tier"])
    y = df["is_fraud"]

    clf = build_pipeline()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    clf.fit(X_train, y_train)

    # Evaluate
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba) if y_test.nunique() > 1 else 0.5
    print(f"[Soltrice] Trained Logistic Regression model. ROC-AUC: {auc:.3f}")

    payload = {
        "pipeline": clf,
        "auc": float(auc),
        "version": "v1.0-logreg",
    }
    joblib.dump(payload, model_path)
    print(f"[Soltrice] Saved model to {model_path}")


def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        print("[Soltrice] Model not found. Training a new one...")
        train_and_save_model(model_path=model_path)

    payload = joblib.load(model_path)
    return payload["pipeline"], payload["auc"], payload["version"]


# ============================================================
# 4. BUSINESS LOGIC FOR BUCKETS / TIERS
# ============================================================

def risk_bucket_from_probability(p: float) -> str:
    if p < 0.3:
        return "Low"
    elif p < 0.7:
        return "Medium"
    else:
        return "High"


def customer_tier_from_features_and_risk(req: FraudRequest, risk_bucket: str) -> str:
    """
    Very simple tier logic for v1:

    - Premium: Low risk AND tenure > 300 days AND prev_rentals >= 3 AND no chargebacks
    - Standard: not fraud-high-risk and not Premium
    - Risky: High risk
    """
    if risk_bucket == "High":
        return "Risky"

    if (
        risk_bucket == "Low"
        and req.customer_tenure_days > 300
        and req.previous_rentals_count >= 3
        and req.previous_chargebacks_count == 0
    ):
        return "Premium"

    return "Standard"


def heuristic_risk_from_features(req: FraudRequest) -> float:
    """
    Simple rule-based risk score in [0,1] for demo purposes.
    We will blend this with the ML model output so that the UI
    moves in a more intuitive way.
    """
    risk = 0.0

    # Rental amount
    if req.rental_amount > 800:
        risk += 0.25
    elif req.rental_amount > 400:
        risk += 0.15
    elif req.rental_amount > 200:
        risk += 0.08

    # Lead time – last-minute bookings are risky
    if req.lead_time_hours < 4:
        risk += 0.15
    elif req.lead_time_hours < 12:
        risk += 0.10

    # Distance from home to branch
    if req.distance_from_home_to_branch_km > 150:
        risk += 0.10
    elif req.distance_from_home_to_branch_km > 80:
        risk += 0.05

    # Customer history
    if req.previous_rentals_count == 0:
        risk += 0.05
    if req.previous_chargebacks_count >= 2:
        risk += 0.40
    elif req.previous_chargebacks_count == 1:
        risk += 0.30

    # Payment & identity
    if req.card_present == "no":
        risk += 0.10
    if req.same_name_on_card == "no":
        risk += 0.10
    if req.billing_matches_id == "no":
        risk += 0.10

    # Behaviour
    if req.failed_attempts_last_24h >= 5:
        risk += 0.15
    elif req.failed_attempts_last_24h >= 1:
        risk += 0.08

    if req.ip_country_matches_id_country == "no":
        risk += 0.10

    # Clamp into [0.01, 0.99] so it never becomes exactly 0 or 1
    risk = max(0.01, min(0.99, risk))
    return risk


# ============================================================
# 5. FASTAPI APP + CORS
# ============================================================

app = FastAPI(
    title="Soltrice Fraud & Customer Tier API",
    description="v1 fraud scoring model + customer tiering.",
    version="1.0.0",
)

# ✅ CORS so browser can call /score from your demo.html
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                   # later you can restrict to ["https://soltrice.com"]
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

model_pipeline, last_auc, model_version = load_model()


@app.get("/health")
def health_check():
    return {"status": "ok", "model_version": model_version, "auc": last_auc}


@app.post("/score", response_model=FraudResponse)
def score(request: FraudRequest):
    try:
        data = pd.DataFrame([request.dict()])

        # 1) ML model probability
        try:
            proba_model = float(model_pipeline.predict_proba(data)[0, 1])
        except Exception:
            # Fallback if model fails for any reason
            proba_model = 0.5

        # 2) Heuristic probability from business rules
        proba_rule = heuristic_risk_from_features(request)

        # 3) Blend them (60% ML, 40% rules for now)
        blended_proba = 0.6 * proba_model + 0.4 * proba_rule

        risk_bucket = risk_bucket_from_probability(blended_proba)
        tier = customer_tier_from_features_and_risk(request, risk_bucket)

        return FraudResponse(
            fraud_probability=round(blended_proba, 4),
            risk_bucket=risk_bucket,
            customer_tier=tier,
            model_version=model_version,
            auc_on_last_train=last_auc,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ✅ Explicit OPTIONS handler for browser preflight
@app.options("/score")
def options_score():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
            "Access-Control-Allow-Headers": "*",
        },
    )


# ============================================================
# 6. ENTRY POINT NEW
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # no auto-reload in cloud
    )

