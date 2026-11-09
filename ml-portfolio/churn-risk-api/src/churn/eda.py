"""EDA report: churn prevalence, contract comparison and tenure distribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data import TARGET, features_and_target, load_telco


def run_eda(source: str, output_dir: Path, synthetic_rows: int = 650) -> dict:
    data = load_telco(source=source, synthetic_rows=synthetic_rows)
    X, y = features_and_target(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    group = pd.DataFrame({"Contract": X["Contract"], "churn": y})
    by_contract = (group.groupby("Contract", dropna=False)
                   .agg(customers=("churn", "size"), churn_rate=("churn", "mean"))
                   .sort_values("churn_rate", ascending=False))
    by_contract.to_csv(output_dir / "churn_by_contract.csv")
    report = {
        "source": source,
        "n_rows": int(len(y)),
        "churn_rate": round(float(y.mean()), 5),
        "missing_total_charges_after_cleaning": int(X["TotalCharges"].isna().sum()),
        "warning": "SYNTHETIC DEMO, not real IBM data" if source == "synthetic" else None,
    }
    (output_dir / "eda_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(by_contract.index.astype(str), by_contract["churn_rate"])
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].set_title("Churn rate by contract" + (" — SYNTHETIC" if source == "synthetic" else ""))
    axes[0].set_ylabel("Fraction churned")
    axes[1].hist([X.loc[y == 0, "tenure"], X.loc[y == 1, "tenure"]],
                 bins=20, label=["Retained", "Churn"], alpha=0.7)
    axes[1].set_title("Tenure by outcome")
    axes[1].set_xlabel("Months")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "eda_overview.png", dpi=130)
    plt.close(fig)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["ibm", "synthetic"], default="ibm")
    p.add_argument("--output-dir", type=Path, default=Path("reports"))
    p.add_argument("--synthetic-rows", type=int, default=650)
    args = p.parse_args()
    print(json.dumps(run_eda(args.source, args.output_dir, args.synthetic_rows), indent=2))


if __name__ == "__main__":
    main()
