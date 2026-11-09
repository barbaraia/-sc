import numpy as np
from housing.data import FEATURES, TARGET, make_synthetic_housing, location_groups
from housing.modeling import candidate_models, regression_metrics
from housing.train import make_splits, train


def test_geographic_splits_disjoint():
    data = make_synthetic_housing(300)
    X = data[list(FEATURES)]
    groups = location_groups(X)
    train_idx, val_idx, test_idx = make_splits(X, groups)
    assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))
    assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
    assert len(set(np.r_[train_idx, val_idx, test_idx])) == len(X)


def test_models_handle_missing_data():
    df = make_synthetic_housing(140)
    X = df[list(FEATURES)].copy()
    X.loc[0, "MedInc"] = np.nan
    pipeline = candidate_models(use_catboost=False)["ridge"]
    pipeline.fit(X, df[TARGET])
    assert np.isfinite(pipeline.predict(X.head())).all()


def test_metrics_exact_prediction():
    metrics = regression_metrics([1, 2, 3], [1, 2, 3])
    assert metrics["rmse_100k_usd"] == 0
    assert metrics["r2"] == 1


def test_end_to_end_synthetic_smoke(tmp_path):
    report = train("synthetic", tmp_path, synthetic_rows=160, use_catboost=False)
    assert report["source"] == "synthetic"
    assert report["selected_model"] != "dummy_median"
    for name in ("housing_model.joblib", "metrics.json", "model_comparison.csv", "predicted_vs_actual.png"):
        assert (tmp_path / name).exists()


def test_eda_report_synthetic(tmp_path):
    from housing.eda import run_eda
    result = run_eda("synthetic", tmp_path, synthetic_rows=100)
    assert result["source"] == "synthetic"
    assert (tmp_path / "eda_overview.png").is_file()
    assert (tmp_path / "numeric_summary.csv").is_file()
