import json

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from churn.data import FEATURES, TARGET, make_synthetic_telco, features_and_target
from churn.modeling import expected_cost, optimize_threshold
from churn.train import train


def test_customer_id_and_churn_not_features():
    data = make_synthetic_telco(140)
    X, y = features_and_target(data)
    assert TARGET not in X.columns
    assert "customerID" not in X.columns
    assert list(X.columns) == list(FEATURES)
    assert set(y.unique()) == {0, 1}


def test_blank_total_charges_for_new_customer():
    data = make_synthetic_telco(140)
    data.loc[0, "tenure"] = 0
    data.loc[0, "TotalCharges"] = " "
    X, _ = features_and_target(data)
    assert X.loc[0, "TotalCharges"] == 0


def test_threshold_optimizer_never_worse_on_validation():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 1])
    p = np.array([.45, .65, .05, .15, .8, .1, .35, .18])
    result = optimize_threshold(y, p, fn_cost=100, fp_cost=10)
    assert expected_cost(y, p, result["threshold"], 100, 10) <= expected_cost(
        y, p, .5, 100, 10
    )


def test_synthetic_end_to_end_api(tmp_path, monkeypatch):
    out = tmp_path / "artifacts"
    report = train(source="synthetic", output_dir=out, synthetic_rows=180,
                   use_catboost=False)
    assert report["dataset"] == "SYNTHETIC SMOKE TEST"
    assert (out / "model_comparison.csv").exists()
    assert json.loads((out / "metrics.json").read_text())["source"] == "synthetic"
    saved = joblib.load(out / "churn_model.joblib")
    assert saved["source"] == "synthetic"
    monkeypatch.setenv("MODEL_PATH", str(out / "churn_model.joblib"))
    from churn.api import app, load_model
    load_model.cache_clear()
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ready"
    customer = make_synthetic_telco(1).drop(columns=["customerID", TARGET])
    customer["TotalCharges"] = 0.0
    response = client.post("/predict", json=customer.iloc[0].to_dict())
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["data_source"] == "synthetic"
    load_model.cache_clear()


def test_api_rejects_unexpected_input():
    from churn.api import app
    client = TestClient(app)
    assert client.post("/predict", json={"totally_wrong": 1}).status_code == 422


def test_eda_report_synthetic(tmp_path):
    from churn.eda import run_eda
    result = run_eda("synthetic", tmp_path, synthetic_rows=100)
    assert result["source"] == "synthetic"
    assert (tmp_path / "eda_overview.png").is_file()
    assert (tmp_path / "churn_by_contract.csv").is_file()
