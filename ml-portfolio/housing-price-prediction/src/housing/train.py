"""Train with grouped spatial holdout and GroupKFold model comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score

from .data import FEATURES, TARGET, load_housing, location_groups
from .modeling import candidate_models, regression_metrics


def make_splits(X: pd.DataFrame, groups: np.ndarray, seed: int = 42):
    idx = np.arange(len(X))
    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval, test = next(first.split(X, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    subtrain, valid = next(second.split(X.iloc[trainval], groups=groups[trainval]))
    train, val = trainval[subtrain], trainval[valid]
    for left, right in [(train, val), (train, test), (val, test)]:
        assert set(groups[left]).isdisjoint(set(groups[right]))
    return train, val, test


def train(source: str, output_dir: Path, seed: int = 42, synthetic_rows: int = 700,
          use_catboost: bool = True) -> dict:
    df = load_housing(source=source, synthetic_rows=synthetic_rows)
    X = df[list(FEATURES)].copy()
    y = df[TARGET].copy()
    groups = location_groups(X)
    train_idx, val_idx, test_idx = make_splits(X, groups, seed=seed)
    Xtrain, Xval, Xtest = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    ytrain, yval, ytest = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]
    splits = GroupKFold(n_splits=3)
    scores = []
    models = candidate_models(seed=seed, use_catboost=use_catboost)

    for name, pipeline in models.items():
        cv = cross_val_score(
            pipeline, Xtrain, ytrain, groups=groups[train_idx], cv=splits,
            scoring="neg_root_mean_squared_error", n_jobs=1,
        )
        scores.append({"model": name, "cv_rmse_mean": round(float(-cv.mean()), 5),
                       "cv_rmse_std": round(float(cv.std()), 5)})

    comparison = pd.DataFrame(scores).sort_values("cv_rmse_mean")
    # The baseline is not allowed to win, except for a degenerate dataset.
    eligible = comparison[comparison.model != "dummy_median"]
    winner = str(eligible.iloc[0]["model"])
    selected = clone(models[winner]).fit(Xtrain, ytrain)
    validation = regression_metrics(yval, selected.predict(Xval))

    trainval_idx = np.concatenate([train_idx, val_idx])
    final = clone(models[winner]).fit(X.iloc[trainval_idx], y.iloc[trainval_idx])
    pred = final.predict(Xtest)
    test_metrics = regression_metrics(ytest, pred)
    baseline = clone(models["dummy_median"]).fit(X.iloc[trainval_idx], y.iloc[trainval_idx])
    baseline_test = regression_metrics(ytest, baseline.predict(Xtest))

    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / "housing_model.joblib"
    joblib.dump(final, model_file)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    report = {
        "dataset": "California Housing 1990 census" if source == "california" else "SYNTHETIC DEMO ONLY",
        "source": source,
        "target": TARGET,
        "target_unit": "100000 USD",
        "split_strategy": "disjoint 0.5-degree geographic grid cells",
        "model_selection": "3-fold GroupKFold on train split only",
        "n_samples": len(df),
        "train_rows": len(train_idx), "validation_rows": len(val_idx), "test_rows": len(test_idx),
        "random_seed": seed,
        "selected_model": winner,
        "validation_metrics": validation,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_test,
        "warning": "Synthetic results are not real census benchmarks" if source == "synthetic" else None,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.scatter(ytest, pred, s=9, alpha=0.5)
    lo = float(min(ytest.min(), pred.min()))
    hi = float(max(ytest.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
    ax.set_xlabel("Actual median house value (×$100k)")
    ax.set_ylabel("Prediction (×$100k)")
    ax.set_title("Geographic holdout: predicted vs actual")
    fig.tight_layout()
    fig.savefig(output_dir / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["california", "synthetic"], default="california")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-rows", type=int, default=700)
    parser.add_argument("--no-catboost", action="store_true")
    args = parser.parse_args()
    result = train(args.source, args.output_dir, args.seed, args.synthetic_rows,
                   use_catboost=not args.no_catboost)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
