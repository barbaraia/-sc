"""Select via stratified CV; tune decision threshold on validation only; test once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .data import FEATURES, load_telco, features_and_target
from .modeling import candidate_models, classification_metrics, optimize_threshold


def train(source: str = "ibm", output_dir: Path = Path("artifacts"),
          seed: int = 42, synthetic_rows: int = 650,
          use_catboost: bool = True, fn_cost: float = 150, fp_cost: float = 25) -> dict:
    raw = load_telco(source=source, synthetic_rows=synthetic_rows)
    X, y = features_and_target(raw)
    if len(y) < 80 or y.nunique() != 2:
        raise ValueError("At least 80 rows and both classes are required")
    trainval_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed,
    )
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.25, stratify=y.iloc[trainval_idx], random_state=seed + 1,
    )
    candidates = candidate_models(seed=seed, use_catboost=use_catboost)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    comparison = []
    for name, pipeline in candidates.items():
        pr_scores = cross_val_score(
            pipeline, X.iloc[train_idx], y.iloc[train_idx], cv=cv,
            scoring="average_precision", n_jobs=1,
        )
        comparison.append({"model": name, "cv_average_precision": round(float(pr_scores.mean()), 5),
                           "cv_std": round(float(pr_scores.std()), 5)})
    frame = pd.DataFrame(comparison).sort_values("cv_average_precision", ascending=False)
    eligible = frame[frame.model != "dummy_prior"]
    winner = str(eligible.iloc[0]["model"])
    validation_model = clone(candidates[winner]).fit(X.iloc[train_idx], y.iloc[train_idx])
    val_prob = validation_model.predict_proba(X.iloc[val_idx])[:, 1]
    threshold_info = optimize_threshold(y.iloc[val_idx], val_prob, fn_cost, fp_cost)
    threshold = threshold_info["threshold"]
    validation_metrics = classification_metrics(y.iloc[val_idx], val_prob, threshold, fn_cost, fp_cost)

    # Only now refit chosen model on train + validation and open sealed test set.
    final_model = clone(candidates[winner]).fit(X.iloc[trainval_idx], y.iloc[trainval_idx])
    test_prob = final_model.predict_proba(X.iloc[test_idx])[:, 1]
    test_metrics = classification_metrics(y.iloc[test_idx], test_prob, threshold, fn_cost, fp_cost)
    dummy = clone(candidates["dummy_prior"]).fit(X.iloc[trainval_idx], y.iloc[trainval_idx])
    dummy_test = classification_metrics(y.iloc[test_idx],
                                        dummy.predict_proba(X.iloc[test_idx])[:, 1],
                                        threshold, fn_cost, fp_cost)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {"pipeline": final_model, "feature_names": list(FEATURES),
                "threshold": threshold, "selected_model": winner, "source": source}
    joblib.dump(artifact, output_dir / "churn_model.joblib")
    frame.to_csv(output_dir / "model_comparison.csv", index=False)
    report = {
        "dataset": "IBM Telco Customer Churn (fictional telco sample)" if source == "ibm" else "SYNTHETIC SMOKE TEST",
        "source": source, "total_rows": len(y), "positive_rate": round(float(y.mean()), 5),
        "train_rows": len(train_idx), "validation_rows": len(val_idx), "test_rows": len(test_idx),
        "random_seed": seed, "selected_model": winner,
        "selection_method": "3-fold stratified CV average precision on train only",
        "threshold_optimization": threshold_info,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "baseline_test_metrics": dummy_test,
        "note": "Business costs are illustrative assumptions, not measured savings. "
                "Synthetic metrics are not actual IBM Telco benchmarks." if source == "synthetic"
                else "Business costs are illustrative assumptions, not measured savings.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["ibm", "synthetic"], default="ibm")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--synthetic-rows", type=int, default=650)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fn-cost", type=float, default=150,
                        help="Hypothetical cost for each missed churner")
    parser.add_argument("--fp-cost", type=float, default=25,
                        help="Hypothetical cost for each false retention alert")
    parser.add_argument("--no-catboost", action="store_true")
    args = parser.parse_args()
    report = train(args.source, args.output_dir, args.seed, args.synthetic_rows,
                   not args.no_catboost, args.fn_cost, args.fp_cost)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
