"""Explicit real-data source, checksum and synthetic-only smoke test generator."""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

IBM_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
IBM_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
RAW_PATH = Path("data/raw/Telco-Customer-Churn.csv")
TARGET = "Churn"
NUMERIC = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
CATEGORICAL = (
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
)
FEATURES = NUMERIC + CATEGORICAL


def download_ibm(path: Path = RAW_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == IBM_SHA256:
            return path
        raise ValueError(f"Cached dataset checksum mismatch at {path}; remove it and retry")
    try:
        with urlopen(IBM_URL, timeout=40) as response:
            content = response.read()
    except Exception as exc:
        raise RuntimeError(
            "IBM Telco download needs network access. Use '--source synthetic' for an "
            "OFFLINE FUNCTIONAL TEST only; do not report its metrics as a real benchmark."
        ) from exc
    digest = hashlib.sha256(content).hexdigest()
    if digest != IBM_SHA256:
        raise ValueError("Downloaded dataset checksum mismatch. The source may have changed.")
    path.write_bytes(content)
    return path


def make_synthetic_telco(n: int = 650, seed: int = 42) -> pd.DataFrame:
    """Fictional examples for offline system checks, *not* an empirical churn dataset."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(19.9, 115, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, .24, .21])
    total = tenure * monthly * rng.uniform(.92, 1.08, n)
    score = (-1.8 + 0.012 * (monthly - 65) - 0.028 * tenure +
             1.2 * (contract == "Month-to-month") +
             0.25 * (contract == "One year") + rng.normal(0, 0.4, n))
    prob = 1 / (1 + np.exp(-score))
    churn = rng.random(n) < prob
    choices = lambda opts: rng.choice(opts, n)
    frame = pd.DataFrame({
        "customerID": [f"TEST-{i:05d}" for i in range(n)],
        "gender": choices(["Male", "Female"]),
        "SeniorCitizen": rng.choice([0, 1], n, p=[.84, .16]),
        "Partner": choices(["Yes", "No"]),
        "Dependents": choices(["Yes", "No"]),
        "tenure": tenure,
        "PhoneService": choices(["Yes", "No"]),
        "MultipleLines": choices(["Yes", "No", "No phone service"]),
        "InternetService": choices(["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": choices(["Yes", "No", "No internet service"]),
        "OnlineBackup": choices(["Yes", "No", "No internet service"]),
        "DeviceProtection": choices(["Yes", "No", "No internet service"]),
        "TechSupport": choices(["Yes", "No", "No internet service"]),
        "StreamingTV": choices(["Yes", "No", "No internet service"]),
        "StreamingMovies": choices(["Yes", "No", "No internet service"]),
        "Contract": contract,
        "PaperlessBilling": choices(["Yes", "No"]),
        "PaymentMethod": choices([
            "Electronic check", "Mailed check", "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]),
        "MonthlyCharges": monthly.round(2),
        "TotalCharges": [" " if months == 0 else f"{v:.2f}" for months, v in zip(tenure, total)],
        TARGET: np.where(churn, "Yes", "No"),
    })
    return frame


def load_telco(source: str = "ibm", path: Path = RAW_PATH,
               synthetic_rows: int = 650) -> pd.DataFrame:
    if source == "synthetic":
        return make_synthetic_telco(n=synthetic_rows)
    if source != "ibm":
        raise ValueError("source must be 'ibm' or 'synthetic'")
    return pd.read_csv(download_ibm(path), dtype={"customerID": "string"})


def features_and_target(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = (set(FEATURES) | {TARGET}) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    X = raw[list(FEATURES)].copy()
    for column in NUMERIC:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    # New clients have no prior charges: zero is a known value, not imputation.
    X.loc[(X["tenure"] == 0) & X["TotalCharges"].isna(), "TotalCharges"] = 0.0
    labels = raw[TARGET].astype(str).str.strip()
    if not set(labels.unique()).issubset({"Yes", "No"}):
        raise ValueError("Churn must consist of Yes/No values")
    y = labels.eq("Yes").astype(int)
    return X, y
