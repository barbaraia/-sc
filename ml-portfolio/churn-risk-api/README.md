# IBM Telco Customer Churn — classification & FastAPI

**Portfolio case | Python · pandas · scikit-learn · CatBoost · SQL · FastAPI · Docker**

Train a classifier to prioritize potentially churning customers of a **fictional telecom company**, then serve probabilities through a JSON API. The IBM sample dataset contains **7,043 customers**. This is a demonstration, **not a deployed retention system**.

> **На русском:** [разбор методологии и вопросы для собеседования](docs/GUIDE_RU.md). Код подготовлен с помощью ИИ; перед включением в резюме самостоятельно выполните обучение, проверьте метрики и разберитесь в решениях.

## What this project actually demonstrates

1. Download IBM's public sample CSV and check its **SHA-256**. The dataset itself is not committed.
2. Clean `TotalCharges` (11 new customers have blank charges). Drop `customerID` and `Churn` from training features.
3. Stratified **60/20/20 train/validation/test** split. Feature imputation, scaling and categorical encoding are fitted **within pipelines and CV folds**, never on the full dataset.
4. Compare a prevalence baseline, logistic regression, random forest and optional CatBoost using **3-fold CV average precision** on the training partition only.
5. Choose a decision threshold on the validation partition using **hypothetical missed-churn vs false-alarm costs**, then refit the selected model on train+validation, and evaluate the test partition **once**.
6. Save a `joblib` model and `metrics.json`, including ROC-AUC, PR-AUC (average precision), precision, recall, F1, Brier score and confusion matrix.
7. Expose `/health` and `/predict` via FastAPI, with Pydantic request validation. A PostgreSQL feature-building design example illustrates a point-in-time JOIN + CTE + window query.

## Setup & training

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e '.[boosting,dev]'
python -m churn.eda --source ibm --output-dir reports
python -m churn.train --source ibm --output-dir artifacts
pytest -q
```

The first IBM run needs an internet connection to download the CSV. Dataset source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv). It is a **sample based on fictional customers**; do not represent it as real customer records.

### Offline smoke test (synthetic data only)

```bash
python -m churn.eda --source synthetic --synthetic-rows 650
python -m churn.train --source synthetic --synthetic-rows 650 --no-catboost
```

**Never report synthetic metrics as results on the IBM dataset.** Check `artifacts/metrics.json` for the data source used in each run.

## Run API

```bash
uvicorn churn.api:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
curl -X POST 'http://127.0.0.1:8000/predict' -H 'Content-Type: application/json' \
    --data @examples/request.json
```

`/predict` returns `churn_probability`, a validation-selected `decision_threshold` and `recommend_retention_review`. This is **not** a causal estimate of whether an intervention will help.

## Docker deployment

**Train before building** so the saved model is in `artifacts/`.

```bash
docker build -t telco-churn-api .
docker run --rm -p 8000:8000 telco-churn-api
```

The API returns HTTP 503 if there is no trained artifact. `/health` reports `model_not_trained` in that case.

## SQL feature engineering

See [`sql/customer_features.sql`](sql/customer_features.sql): joins customer, billing and service event tables with multiple CTEs and `ROW_NUMBER`, ensuring **each source event is earlier than the prediction timestamp**. It is a schema design example, not a SQL statement that can be run on IBM's single CSV table.

## Caveats

- ROC-AUC and PR-AUC are discrimination metrics, not business ROI. False-positive cost = 25 and false-negative cost = 150 are **invented illustrative assumptions**, not empirically measured costs.
- IBM data is a teaching sample, and random splits are **not** proof of performance under changes over time. A real implementation would require a temporal holdout, calibration/drift monitoring, experiment design and intervention impact measurement.
- Sensitive demographic features are present in IBM's example data; a real project requires legality, necessity and fairness review. No recommendation to make automated consequential decisions based on this demo.
- Hyperparameters are predefined. Better generalization analysis would include nested model selection, confidence intervals and multiple runs.

## Results

Run training and inspect `artifacts/metrics.json`. There are intentionally **no unverified performance claims** in this README.

## License

Source code is MIT-licensed. IBM's sample dataset is downloaded from its publisher and is not redistributed here.
