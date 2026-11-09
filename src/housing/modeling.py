"""Pipelines and evaluation helpers. All preprocessing is fitted within a split."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import FEATURES


class CatBoostRegressorAdapter(RegressorMixin, BaseEstimator):
    """Wrap CatBoost for sklearn>=1.6 estimator tags and Pipeline/clone support."""

    def __init__(self, iterations: int = 250, depth: int = 5,
                 learning_rate: float = 0.07, seed: int = 42):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.seed = seed

    def fit(self, X, y):
        from catboost import CatBoostRegressor

        self.model_ = CatBoostRegressor(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            verbose=False, thread_count=2, random_seed=self.seed,
            allow_writing_files=False,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self)
        return self.model_.predict(X)


def candidate_models(seed: int = 42, use_catboost: bool = True) -> dict:
    standard = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), list(FEATURES))
    ])

    def pipe(estimator):
        return Pipeline([("preprocessing", standard), ("regressor", estimator)])

    models = {
        "dummy_median": pipe(DummyRegressor(strategy="median")),
        "ridge": pipe(Ridge(alpha=2.0)),
        "random_forest": pipe(RandomForestRegressor(
            n_estimators=80, min_samples_leaf=3, n_jobs=-1, random_state=seed
        )),
        "hist_gradient_boosting": pipe(HistGradientBoostingRegressor(
            max_iter=120, max_leaf_nodes=21, learning_rate=0.08, random_state=seed
        )),
    }
    if use_catboost:
        try:
            import catboost  # noqa: F401
        except ImportError:
            pass
        else:
            models["catboost"] = pipe(CatBoostRegressorAdapter(seed=seed))
    return models


def regression_metrics(actual, predicted) -> dict[str, float]:
    y = np.asarray(actual)
    p = np.asarray(predicted)
    return {
        "mae_100k_usd": round(float(mean_absolute_error(y, p)), 5),
        "rmse_100k_usd": round(float(np.sqrt(mean_squared_error(y, p))), 5),
        "r2": round(float(r2_score(y, p)), 5),
    }
