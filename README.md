# Fraud Detection API

> Production-grade fraud scoring service.
> XGBoost classifier on the IEEE-CIS dataset, served behind a FastAPI REST API,
> with **per-prediction SHAP explanations** so every score is auditable.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-ff7f0e)
![SHAP](https://img.shields.io/badge/SHAP-0.46-7e57c2)
![Tests](https://img.shields.io/badge/tests-12%20passing-brightgreen)

---
![API](https://fraud-detection-api-qos3.onrender.com/)

## Why this project

Most fraud-model demos stop at "I trained a classifier and got 99% accuracy on
an imbalanced dataset." This one goes further:

- **Honest evaluation** for a 3.5%-fraud problem: ROC-AUC, PR-AUC, F1, precision,
  recall, full confusion matrix at multiple thresholds.
- **No data leakage**: time-based train/validation split (random K-fold would
  let the model peek at the future), and SMOTE is applied **inside each CV
  fold** via `imblearn.Pipeline`.
- **Explainable**: every API response includes the top 3 SHAP features driving
  the prediction, signed by direction. A risk officer can see *why*, not just
  *what*.
- **Reproducible**: pinned dependencies, seeded splits, MLflow run tracking.

---

## Demo

Send a transaction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_001",
    "TransactionDT": 86400,
    "TransactionAmt": 125.50,
    "ProductCD": "W",
    "card1": 13926,
    "card4": "visa",
    "card6": "debit",
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "gmail.com",
    "extra": { "V12": 1.0, "V13": 1.0 }
  }'
```

Get a scored response with explanations:

```json
{
  "transaction_id": "txn_001",
  "fraud_probability": 0.021,
  "prediction": "NOT_FRAUD",
  "risk_level": "LOW",
  "top_reasons": [
    { "feature": "V308",          "impact": 0.374, "direction": "decreases_risk" },
    { "feature": "P_emaildomain", "impact": 0.352, "direction": "decreases_risk" },
    { "feature": "V317",          "impact": 0.338, "direction": "decreases_risk" }
  ],
  "model_version": "0.1.0"
}
```

Interactive docs live at **`http://127.0.0.1:8000/docs`** once the API is up.

---

## Architecture

```
┌──────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│  IEEE-CIS CSVs   │───▶│  preprocess.py      │───▶│  imblearn       │
│  (train_txn +    │    │  - merge on TxnID   │    │  Pipeline       │
│   train_id)      │    │  - drop high-NaN    │    │  ┌────────────┐ │
└──────────────────┘    │  - engineer (×5)    │    │  │ preprocess │ │
                        │  - freq encode      │    │  │   SMOTE    │ │
                        │  - OHE + median imp │    │  │  XGBoost   │ │
                        └─────────────────────┘    │  └────────────┘ │
                                                   └────────┬────────┘
                                                            │
                                                            ▼
   ┌─────────────────────┐    ┌──────────────────┐  ┌──────────────┐
   │  FastAPI            │◀───│  predict.py      │◀─│  fraud_xgb   │
   │  POST /predict      │    │  - reindex cols  │  │    .pkl      │
   │  GET  /health       │    │  - predict_proba │  │              │
   │  GET  /docs         │    │  - SHAP top-3    │  └──────────────┘
   └─────────────────────┘    └──────────────────┘
```

**Why these choices**

| Decision | Reason |
|---|---|
| XGBoost over a deep net | Tabular data with mixed types and lots of NaN; trees handle both natively |
| SMOTE for imbalance | Cleaner than `scale_pos_weight` when you also want calibrated probabilities |
| `imblearn.Pipeline` | Guarantees SMOTE runs **only** in `fit` — validation folds stay pristine |
| Time-based split | Random K-fold leaks future info on time-ordered data |
| SHAP `TreeExplainer` | Exact SHAP values, milliseconds per prediction (vs. KernelExplainer's seconds) |

---

## Setup

Tested on Python 3.10. Should work on 3.10–3.12.

```bash
# 1. Clone and enter the repo
git clone https://github.com/Mathias-Kabango3/fraud-detection.git
cd fraud-detection

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies (~1.5 GB; takes 5–10 minutes the first time)
pip install --upgrade pip
pip install -r requirements.txt

# 4. Copy the env template
cp .env.example .env
```

### Get the data

Download the IEEE-CIS Fraud Detection dataset from Kaggle:
<https://www.kaggle.com/c/ieee-fraud-detection>

Drop these into `data/raw/` (gitignored — never committed):

- `train_transaction.csv` (~650 MB)
- `train_identity.csv` (~25 MB)

---

## Train the model

```bash
# Full training run — 5-fold CV + final fit on the 7-day holdout
# Takes ~30 minutes and uses several GB of RAM
python -m src.model.train

# Quick iteration during development
python -m src.model.train --sample-size 100000 --cv-folds 3
python -m src.model.train --no-mlflow --skip-cv   # fit + holdout only
```

Outputs:
- `models/fraud_xgb.pkl` — pickled pipeline the API loads at startup
- `mlruns/` — MLflow run with params, per-fold metrics, holdout metrics, model artifact
- Console summary with `FINAL holdout ROC-AUC` and PASS/FAIL against the 0.88 target

Browse experiment runs:

```bash
mlflow ui   # http://127.0.0.1:5000
```

### Extended evaluation

```bash
python -m src.model.evaluate              # full holdout eval + plots
```

Writes `models/eval_roc.png`, `models/eval_pr.png`, `models/eval_confusion.png`,
and `models/eval_summary.json`, plus a threshold sweep table on stdout.

### SHAP exploration

```bash
python -m src.model.explain --n-rows 5    # demo + global summary plot
```

---

## Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Then open <http://127.0.0.1:8000/docs> for the interactive Swagger UI.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health`       | Liveness/readiness, returns `model_loaded` and `model_version` |
| `POST` | `/predict`      | Score one transaction, return probability + top-3 SHAP reasons |
| `GET`  | `/docs`         | Swagger UI (auto-generated from the Pydantic schemas) |
| `GET`  | `/openapi.json` | Machine-readable OpenAPI spec |

### Health check

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

```json
{ "status": "ok", "model_loaded": true, "model_version": "0.1.0" }
```

### Risk-level buckets

The `risk_level` field is derived from `fraud_probability` using two thresholds
(env-configurable):

| Probability    | `risk_level` |
|---|---|
| `< 0.30`        | `LOW`    |
| `0.30 – 0.70`   | `MEDIUM` |
| `> 0.70`        | `HIGH`   |

The hard label (`FRAUD` / `NOT_FRAUD`) uses a separate decision threshold
(default `0.50`; tune via `DECISION_THRESHOLD` in `.env`). The evaluation
report recommends an **F1-optimal threshold around 0.25** for this dataset —
worth lowering once you have a deployment context.

---

## Metrics

Latest smoke results (model trained on a 50k-row sample with 2 CV folds):

| Metric                       | Value     | Target  |
|---|---:|---:|
| CV ROC-AUC (mean ± std)      | 0.893 ± 0.004 | — |
| Holdout ROC-AUC              | 0.883     | > 0.88 ✓ |
| Holdout PR-AUC               | 0.41      | — |
| Best-F1 threshold            | 0.25      | — |
| F1 @ best threshold          | 0.61      | — |

Full 590k-row training is expected to push these meaningfully higher; numbers
will be updated after the full run.

---

## Tests

```bash
pytest -v tests/
```

The suite uses FastAPI's `TestClient` plus a session-scoped fixture that
trains a tiny model on synthetic data (no real CSVs needed), then monkeypatches
the global predictor. 12 tests covering:

- `/health` with model loaded and unloaded
- `/predict` happy path + response-shape assertions
- SHAP top-reasons structure
- Validation errors (missing field, negative amount, malformed JSON)
- Graceful 503 when the model isn't on disk
- `/docs` and `/openapi.json` surface

Whole suite runs in ~2 seconds.

---

## Deployment (Render + Hugging Face Hub)

The API is designed to deploy as a free-tier Render web service. The trained
model artifact lives on Hugging Face Hub and is downloaded at startup, so the
repo itself stays small.

### One-time setup

**1. Upload the trained pickle to Hugging Face Hub**

```bash
pip install huggingface_hub                   # already in requirements.txt
huggingface-cli login                         # paste a write token
huggingface-cli repo create fraud-detection-model --type=model
huggingface-cli upload \
  YOUR_HF_USERNAME/fraud-detection-model \
  models/fraud_xgb.pkl \
  fraud_xgb.pkl
```

Public repos need no token at download time. Private repos require
`HF_TOKEN` to be set in Render.

**2. Connect the GitHub repo to Render**

In the Render dashboard:

1. **New +** → **Blueprint** → select this repo → Render reads `render.yaml`.
2. Set the secret env var the blueprint marks as `sync: false`:
   - `HF_MODEL_REPO_ID` = `YOUR_HF_USERNAME/fraud-detection-model`
   - (Optional) `HF_TOKEN` if the repo is private.
3. Click **Create Blueprint**. First build takes ~5–8 minutes (pip install).

### What the blueprint provisions

```
service:  fraud-detection-api
plan:     free  (512 MB RAM, sleeps after 15 min idle)
build:    pip install -r requirements.txt
start:    uvicorn src.api.main:app --host 0.0.0.0 --port $PORT --workers 1
health:   GET /health  (Render polls this)
```

At cold start the API:
1. Boots uvicorn.
2. Inside the FastAPI lifespan: checks for `models/fraud_xgb.pkl` locally.
3. Not found → reads `HF_MODEL_REPO_ID` and pulls the artifact (~few seconds for a small pickle).
4. Loads the pipeline + builds the SHAP explainer.
5. Reports `model_loaded: true` on `/health` and starts serving `/predict`.

### Operational notes

- **Cold starts**: free tier sleeps after 15 minutes idle. First request after sleep
  is ~30–60 seconds (boot + download + load). Subsequent requests are fast.
- **One worker**: the start command pins `--workers 1` because each worker keeps a
  full copy of the model + SHAP explainer in memory. Two workers won't fit in 512 MB.
- **No training in prod**: `MLFLOW_TRACKING_URI` is intentionally empty in
  `render.yaml`. Training runs locally; only inference runs on Render.
- **Updating the model**: re-train locally → re-upload to HF Hub → Render's next
  cold start picks up the new artifact. No re-deploy needed unless code changes.

### Tuning thresholds in production

`DECISION_THRESHOLD`, `RISK_MEDIUM_THRESHOLD`, and `RISK_HIGH_THRESHOLD` are
all env vars on the Render service. Adjust them in the dashboard and click
"Save and deploy" — no code change required. The evaluation report's
F1-optimal threshold of ~0.25 is a sensible production starting point.

---

## Project layout

```
fraud-detection/
├── data/
│   └── raw/                    # IEEE-CIS CSVs (gitignored)
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory data analysis
├── src/
│   ├── data/
│   │   └── preprocess.py       # load, engineer, encode, split (sklearn-compatible)
│   ├── model/
│   │   ├── train.py            # CV + final fit + MLflow + pickle
│   │   ├── evaluate.py         # metrics, threshold sweep, ROC/PR/confusion plots
│   │   └── explain.py          # SHAP TreeExplainer + per-prediction top-K
│   └── api/
│       ├── schema.py           # Pydantic request/response models
│       ├── predict.py          # FraudPredictor: load + score + explain
│       └── main.py             # FastAPI app with lifespan model loading
├── tests/
│   ├── conftest.py             # tiny synthetic-data model fixture
│   └── test_api.py             # 12 endpoint tests
├── models/                     # fraud_xgb.pkl + plots (gitignored)
├── mlruns/                     # MLflow run store (gitignored)
├── .env.example
├── render.yaml                 # Render Blueprint for deployment
├── requirements.txt
└── README.md
```

---

## Tech stack

| Layer            | Library |
|---|---|
| Modeling         | XGBoost 2.1, scikit-learn 1.5, imbalanced-learn 0.12 (SMOTE) |
| Explainability   | SHAP 0.46 (TreeExplainer) |
| API              | FastAPI 0.115, Uvicorn 0.30, Pydantic 2.9 |
| Data             | pandas 2.2, numpy 1.26 |
| Experiment track | MLflow 2.16 (local file store) |
| Testing          | pytest 8.3, httpx 0.27 |
| Notebooks        | Jupyter, matplotlib 3.9, seaborn 0.13 |

---

## Roadmap

- [x] Project scaffolding + virtual environment
- [x] EDA notebook (`01_eda.ipynb`)
- [x] Preprocessing pipeline (sklearn-compatible transformers)
- [x] Training pipeline (CV + SMOTE + XGBoost + MLflow)
- [x] Evaluation + SHAP explanations
- [x] FastAPI service (`/predict`, `/health`, `/docs`)
- [x] API test suite
- [x] README polish
- [x] Render deployment config (`render.yaml` + HF Hub artifact fetch)

---

## License

MIT — feel free to fork and adapt.
