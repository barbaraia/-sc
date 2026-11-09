"""Model candidates and business-aware thresholding (assumed costs, not actual ROI)."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from .data import NUMERIC, CATEGORICAL


class CatBoostClassifierAdapter(ClassifierMixin, BaseEstimator):
    """Wrap CatBoost for modern sklearn estimator tags and Pipeline/clone support."""

    def __init__(self, iterations: int = 220, depth: int = 5,
                 learning_rate: float = 0.06, seed: int = 42):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.seed = seed

    def fit(self, X, y):
        from catboost import CatBoostClassifier

        self.model_ = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            loss_function="Logloss", verbose=False, thread_count=2, random_seed=self.seed,
            allow_writing_files=False,
        )
        self.model_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self)
        return self.model_.predict(X).reshape(-1)

    def predict_proba(self, X):
        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self)
        return self.model_.predict_proba(X)


def candidate_models(seed: int = 42, use_catboost: bool = True) -> dict:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, list(NUMERIC)),
        ("categorical", categorical, list(CATEGORICAL)),
    ])

    def pipe(model):
        return Pipeline([("preprocessing", preprocessing), ("classifier", model)])

    models = {
        "dummy_prior": pipe(DummyClassifier(strategy="prior")),
        "logistic_regression": pipe(LogisticRegression(
            max_iter=600, class_weight="balanced", random_state=seed,
        )),
        "random_forest": pipe(RandomForestClassifier(
            n_estimators=100, min_samples_leaf=4, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        )),
    }
    if use_catboost:
        try:
            import catboost  # noqa: F401
        except ImportError:
            pass
        else:
            models["catboost"] = pipe(CatBoostClassifierAdapter(seed=seed))
    return models


def expected_cost(y_true, probabilities, threshold: float,
                  false_negative_cost: float = 150.0,
                  false_positive_cost: float = 25.0) -> float:
    """Hypothetical per-case cost model. Not measured customer lifetime value."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fn * false_negative_cost + fp * false_positive_cost)


def optimize_threshold(y_true, probabilities, fn_cost: float = 150,
                       fp_cost: float = 25) -> dict:
    options = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [(float(t), expected_cost(y_true, probabilities, t, fn_cost, fp_cost))
              for t in options]
    threshold, cost = min(scores, key=lambda tc: (tc[1], abs(tc[0] - 0.5)))
    return {"threshold": threshold, "validation_expected_cost": round(cost, 2),
            "assumed_fn_cost": fn_cost, "assumed_fp_cost": fp_cost}


def classification_metrics(y_true, probabilities, threshold: float,
                           fn_cost: float = 150, fp_cost: float = 25) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probabilities)
    guess = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, guess, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 5),
        "average_precision_pr_auc": round(float(average_precision_score(y, p)), 5),
        "brier_score": round(float(brier_score_loss(y, p)), 5),
        "precision": round(float(precision_score(y, guess, zero_division=0)), 5),
        "recall": round(float(recall_score(y, guess, zero_division=0)), 5),
        "f1": round(float(f1_score(y, guess, zero_division=0)), 5),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "hypothetical_cost": round(expected_cost(y, p, threshold, fn_cost, fp_cost), 2),
    }
