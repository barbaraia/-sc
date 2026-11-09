# California Housing — leakage-aware regression

**Portfolio case | Python · scikit-learn · CatBoost · grouped cross-validation · model comparison**

Predict the median home value for a California census block group. The real dataset has **20,640 observations and 8 numeric features**. The target `MedHouseVal` is expressed in **$100,000 units** (e.g., `2.5` = $250,000). This is *not* an estimate of modern 2026 housing prices.

> **На русском:** [разбор методологии и вопросы для собеседования](docs/GUIDE_RU.md). Код подготовлен с помощью ИИ; перед включением в резюме самостоятельно выполните обучение, проверьте метрики и разберитесь в решениях.

## Why this project is not another `train_test_split` notebook

- Geographic binning (0.5° cells) holds out entire locations; rows from a geographic cell cannot appear in multiple train/validation/test partitions. This is a **stricter spatial generalization test**, not proof that neighboring cells are statistically independent.
- Preprocessing lives inside `sklearn.Pipeline` and is fitted **only on the training rows of each fold**.
- `GroupKFold` compares DummyRegressor, Ridge, Random Forest, HistGradientBoosting and optional CatBoost without touching the test set.
- Validation diagnostics precede refitting on train+validation; the test set is evaluated only once.
- The median-value dummy baseline is measured on the same holdout; no fabricated metrics are printed in this README.
- Results are written to `artifacts/metrics.json`, `artifacts/model_comparison.csv` and a diagnostic PNG.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e '.[boosting,dev]'
python -m housing.eda --source california --output-dir reports
python -m housing.train --source california --output-dir artifacts
python -m pytest -q
```

The **first real-data run downloads** California Housing through `sklearn.datasets.fetch_california_housing`; an internet connection is required for the initial cache fill. Data is not included in the repository.

### Run without internet (functional smoke test only)

```bash
python -m housing.eda --source synthetic --synthetic-rows 700
python -m housing.train --source synthetic --synthetic-rows 700 --no-catboost
```

**Do not cite synthetic metrics as performance on California Housing.** The synthetic generator is intended only for checking pipelines, model selection, metrics and plots offline.

## Expected output

`artifacts/metrics.json` records the dataset source, geographical split, selected model, baseline and validation/test MAE, RMSE and R². The metrics are expressed in $100k units; multiply MAE/RMSE by 100,000 to interpret dollars.

## Limitations and next experiments

1. The census data is from 1990, and the target is top-coded; its performance is not indicative of current housing prices.
2. Random group holdouts can still have nearby geographic cells and do not simulate future-time performance.
3. Extend spatial cross-validation to larger geographic buffers; compare performance with a random split **without selecting a final model on the test set**.
4. Repeat over several random seeds to estimate uncertainty and test for regional bias.

## Data provenance

Source: [scikit-learn California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html), derived from the 1990 US census / StatLib housing dataset. Code is MIT-licensed; no claims are made about ownership of the data.
