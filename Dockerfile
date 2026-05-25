FROM python:3.11.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    MODEL_DIR=models \
    MODEL_FILENAME=fraud_xgb.pkl \
    MODEL_VERSION=0.1.0 \
    HF_MODEL_FILENAME=fraud_xgb.pkl \
    API_LOG_LEVEL=info \
    DECISION_THRESHOLD=0.5 \
    RISK_MEDIUM_THRESHOLD=0.3 \
    RISK_HIGH_THRESHOLD=0.7 \
    MLFLOW_TRACKING_URI=""

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --root-user-action=ignore -r requirements.txt

COPY . .
RUN chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
