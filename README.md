# Fraud Detection

End-to-end fraud detection system: XGBoost classifier trained on the
IEEE-CIS dataset, served behind a FastAPI REST API with SHAP-based
per-prediction explanations.

> Status: **scaffolding** (step 1 of the build plan). Code for
> preprocessing, training, the API, and deployment will land in later steps.

## Project layout

```
fraud-detection/
├── data/raw/              # IEEE-CIS CSVs (gitignored)
├── notebooks/             # 01_eda, 02_training
├── src/
│   ├── data/preprocess.py
│   ├── model/{train,evaluate,explain}.py
│   └── api/{main,schema,predict}.py
├── models/                # Saved artifacts (gitignored)
├── tests/test_api.py
├── .env.example
├── requirements.txt
└── render.yaml
```

## Setup

Requires Python 3.11 (3.10–3.12 should also work).

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Copy env template and edit as needed
cp .env.example .env
```

## Dataset

Download the IEEE-CIS Fraud Detection dataset from Kaggle:
<https://www.kaggle.com/c/ieee-fraud-detection>

Place these two files into `data/raw/`:

- `train_transaction.csv`
- `train_identity.csv`

Both files are gitignored and must never be committed.

## Roadmap

1. Project structure + venv  ← **current**
2. EDA notebook
3. Preprocessing pipeline
4. Training pipeline with CV
5. Evaluation + SHAP
6. FastAPI app (`/predict`, `/health`, `/docs`)
7. API tests
8. README polish + sample `curl` commands
9. Render deployment
