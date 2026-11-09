"""Offline-safe EDA report: real source or clearly labelled synthetic demonstration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data import FEATURES, TARGET, load_housing


def run_eda(source: str, output_dir: Path, synthetic_rows: int = 700) -> dict:
    data = load_housing(source=source, synthetic_rows=synthetic_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": source,
        "n_rows": int(len(data)),
        "missing_values": {col: int(data[col].isna().sum()) for col in (*FEATURES, TARGET)},
        "target_describe": {k: round(float(v), 4) for k, v in data[TARGET].describe().items()},
        "warning": "SYNTHETIC DEMO, not real housing census" if source == "synthetic" else None,
    }
    (output_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    data[[*FEATURES, TARGET]].describe().round(3).to_csv(output_dir / "numeric_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(data[TARGET], bins=35, edgecolor="white")
    axes[0].set_title("Target distribution" + (" — SYNTHETIC" if source == "synthetic" else ""))
    axes[0].set_xlabel("Median house value (×$100k)")
    axes[1].scatter(data["MedInc"], data[TARGET], s=4, alpha=0.25)
    axes[1].set_title("Income vs target")
    axes[1].set_xlabel("Median income (dataset units)")
    axes[1].set_ylabel("Median house value (×$100k)")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_overview.png", dpi=130)
    plt.close(fig)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["california", "synthetic"], default="california")
    p.add_argument("--output-dir", type=Path, default=Path("reports"))
    p.add_argument("--synthetic-rows", type=int, default=700)
    args = p.parse_args()
    print(json.dumps(run_eda(args.source, args.output_dir, args.synthetic_rows), indent=2))


if __name__ == "__main__":
    main()
