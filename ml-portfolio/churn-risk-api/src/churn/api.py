"""FastAPI serving inference with a persisted, versioned feature contract."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="Telco Churn Risk API", version="1.0.0")


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)


@lru_cache(maxsize=1)
def load_model():
    path = Path(os.environ.get("MODEL_PATH", "artifacts/churn_model.joblib"))
    if not path.is_file():
        return None
    return joblib.load(path)


@app.get("/health")
def health() -> dict:
    model = load_model()
    return {"status": "ready" if model else "model_not_trained"}


@app.post("/predict")
def predict(customer: Customer) -> dict:
    artifact = load_model()
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model missing: run training first")
    frame = pd.DataFrame([customer.model_dump()])
    if customer.tenure == 0 and customer.TotalCharges is None:
        frame.loc[0, "TotalCharges"] = 0.0
    probability = float(artifact["pipeline"].predict_proba(
        frame[artifact["feature_names"]]
    )[0, 1])
    threshold = float(artifact["threshold"])
    return {
        "churn_probability": round(probability, 6),
        "decision_threshold": threshold,
        "recommend_retention_review": probability >= threshold,
        "model": artifact["selected_model"],
        "data_source": artifact["source"],
        "disclaimer": "Illustrative model; not validated for real-world decisions.",
    }
