import os
from typing import Literal, Optional, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
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
    """
    Demo-friendly request model.

    Supports:
    - UI/demo payload fields (emailAge, tenure, amount, geoDistance, etc.)
    - Existing model fields (rental_amount, booking_channel, etc.)
    - Normalizes UI fields into model fields so the pipeline can score.
    """

    # ---- NEW UI fields (accepted, optional) ----
    emailAge: Optional[int] = None
    tenure: Optional[int] = None                  # UI often uses months; we convert to days
    chargebacks: Optional[int] = None
    premiumFlag: Optional[int] = None
    driverAge: Optional[int] = None
    licenseYears: Optional[int] = None
    previousRentals: Optional[int] = None
    phoneType: Optional[int] = None
    amount: Optional[float] = None                # maps to rental_amount
    velocity: Optional[int] = None
    binRisk: Optional[int] = None
    oddHour: Optional[int] = None
    deviceTrust: Optional[int] = None
    ipRisk: Optional[int] = None
    geoDistance: Optional[float] = None           # maps to distance_from_home_to_branch_km
    proxyFlag: Optional[int] = None

    # ---- MODEL fields (these are what the ML pipeline actually uses) ----
    rental_amount: Optional[float] = Field(None, example=120.5)
    rental_duration_days: int = Field(3, example=3)
    booking_channel: Literal["web", "app", "phone", "walk_in"] = Field("web", example="web")
    lead_time_hours: float = Field(24, example=48)
    distance_from_home_to_branch_km: Optional[float] = Field(None, example=12.3)
    customer_age: Optional[int] = Field(None, example=42)
    customer_tenure_days: Optional[int] = Field(None, example=365)
    previous_rentals_count: Optional[int] = Field(None, example=4)
    previous_chargebacks_count: Optional[int] = Field(None, example=0)
    card_present: Literal["yes", "no"] = Field("yes", example="yes")
    same_name_on_card: Literal["yes", "no"] = Field("yes", example="yes")
    billing_matches_id: Literal["yes", "no"] = Field("yes", example="yes")
    failed_attempts_last_24h: int = Field(0, example=0)
    ip_country_matches_id_country: Literal["yes", "no"] = Field("yes", example="yes")

    @model_validator(mode="before")
    @classmethod
    def normalize_ui_to_model_fields(cls, values: Any):
        # values is typically a dict when parsing JSON
        if not isinstance(values, dict):
            return values

        # amount -> rental_amount
        if values.get("rental_amount") is None and values.get("amount") is not None:
            values["rental_amount"] = values["amount"]

        # geoDistance -> distance_from_home_to_branch_km
        if values.get("distance_from_home_to_branch_km") is None and values.get("geoDistance") is not None:
            values["distance_from_home_to_branch_km"] = values["geoDistance"]

        # driverAge -> customer_age
        if values.get("customer_age") is None and values.get("driverAge") is not None:
            values["customer_age"] = values["driverAge"]

        # tenure (months) -> customer_tenure_days
        if values.get("customer_tenure_days") is None and values.get("tenure") is not None:
            try:
                values["customer_tenure_days"] = int(values["tenure"]) * 30
            except Exception:
                pass

        # previousRentals -> previous_rentals_count
        if values.get("previous_rentals_count") is None and values.get("previousRentals") is not None:
            values["previous_rentals_count"] = values["previousRentals"]

        # chargebacks -> previous_chargebacks_count
        if values.get("previous_chargebacks_count") is None and values.get("chargebacks") is not None:
            values["previous_chargebacks_count"] = values["chargebacks"]

        return values

    @model_validator(mode="after")
    def ensure_minimums_for_model(self):
        """
        Make sure the ML pipeline always receives valid values for required model columns.
        This prevents 'missing column' or None-related issues.
        """

        # Required numerics for the pipeline
        if self.rental_amount is None:
            self.rental_amount = 180.0
        if self.distance_from_home_to_branch_km is None:
            self.distance_from_home_to_branch_km = 10.0
        if self.customer_age is None:
            self.customer_age = 35
        if self.customer_tenure_days is None:
            self.customer_tenure_days = 180
        if self.previous_rentals_count is None:
            self.previous_rentals_count = 1
        if self.previous_chargebacks_count is None:
            self.previous_chargebacks_count = 0

        # Keep lead_time_hours sane
        if self.lead_time_hours is None:
            self.lead_time_hours = 24

        return self


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

    # Keep consistent columns
    drop_cols = [c for c in ["is_fraud", "customer_tier"] if c in df.columns]
    X = df.drop(columns=drop_cols)
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
    if risk_bucket == "High":
        return "Risky"

    if (
        risk_bucket == "Low"
        and (req.customer_tenure_days or 0) > 300
        and (req.previous_rentals_count or 0) >= 3
        and (req.previous_chargebacks_count or 0) == 0
    ):
        return "Premium"

    return "Standard"


def heuristic_risk_from_features(req: FraudRequest) -> float:
    """
    Simple rule-based risk score in [0,1] for demo purposes.
    We will blend this with the ML model output so the UI feels intuitive.
    """
    risk = 0.0

    # Rental amount
    if req.rental_amount > 800:
        risk += 0.25
    elif req.rental_amount > 400:
        risk += 0.15
    elif req.rental_amount > 200:
        risk += 0.08

    # Lead time
    if req.lead_time_hours < 4:
        risk += 0.15
    elif req.lead_time_hours < 12:
        risk += 0.10

    # Distance
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

    return max(0.01, min(0.99, risk))


# ============================================================
# 5. FASTAPI APP + CORS
# ============================================================

app = FastAPI(
    title="Soltrice Fraud & Customer Tier API",
    description="v1 fraud scoring model + customer tiering.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to ["https://soltrice.com"]
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

model_pipeline, last_auc, model_version = load_model()


@app.get("/health")
def health_check():
    return {"status": "ok", "model_version": model_version, "auc": last_auc}

@app.get("/")
def root():
    return {"status": "ok", "service": "soltrice-sfm-api"}


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


@app.post("/score", response_model=FraudResponse)
def score(request: FraudRequest):
    try:
        # Pydantic v2: model_dump
        row = request.model_dump()

        # IMPORTANT: only pass the columns the pipeline expects
        data = pd.DataFrame([row])[numeric_features + categorical_features]

        # 1) ML model probability
        try:
            proba_model = float(model_pipeline.predict_proba(data)[0, 1])
        except Exception:
            proba_model = 0.5

        # 2) Heuristic probability
        proba_rule = heuristic_risk_from_features(request)

        # 3) Blend them
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


# ============================================================
# 6. ENTRY POINT (local dev)
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
