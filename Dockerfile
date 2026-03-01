# =============================================================================
# Optikal — Multi-stage Dockerfile
#
# Stages:
#   base     — Python + system deps (shared)
#   trainer  — Full training environment (CPU-only; use GPU variant for CUDA)
#   inference — Lean inference image for the Kafka consumer
#
# Build examples:
#   docker build --target trainer   -t optikal:trainer   .
#   docker build --target inference -t optikal:inference .
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: base
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

LABEL maintainer="Cogensec <info@cogensec.com>"
LABEL description="Optikal GPU-Accelerated AI Threat Detection"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install core Python dependencies first (layer-cached independently of code)
COPY optikal/requirements.txt requirements.txt
RUN pip install --no-cache-dir numpy pandas scikit-learn joblib

# ---------------------------------------------------------------------------
# Stage 2: trainer
# ---------------------------------------------------------------------------
FROM base AS trainer

# Install full training stack (LSTM + tracking + explainability)
RUN pip install --no-cache-dir \
        tensorflow>=2.13.0 \
        shap>=0.42.0 \
        mlflow>=2.7.0 \
        optuna>=3.3.0 \
        pyyaml>=6.0.0

COPY . .
RUN pip install --no-cache-dir -e ".[explain,tracking,tuning]"

# Model output directory (mount as volume in production)
RUN mkdir -p /app/optikal_models

ENV MLFLOW_TRACKING_URI=http://mlflow:5000

CMD ["python", "-m", "optikal.train_quick"]

# ---------------------------------------------------------------------------
# Stage 3: inference (Kafka consumer — lean image)
# ---------------------------------------------------------------------------
FROM base AS inference

# Install streaming + explainability extras only
RUN pip install --no-cache-dir \
        confluent-kafka>=2.2.0 \
        shap>=0.42.0

COPY . .
RUN pip install --no-cache-dir -e ".[explain,streaming]"

# Pre-trained model artifacts (bind-mount or copy in)
VOLUME ["/app/optikal_models"]

ENV KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
    MODEL_DIR=/app/optikal_models

EXPOSE 8080

CMD ["python", "-c", "\
import os, sys, logging, joblib; \
sys.path.insert(0,'optikal'); \
from feature_engineering import OptikalFeatureEngineer; \
from optikal_trainer import OptikalIsolationForest, ThreatClassifier; \
from kafka_consumer import KafkaConfig, OptikalKafkaConsumer; \
logging.basicConfig(level=logging.INFO); \
model_dir = os.environ.get('MODEL_DIR', 'optikal_models'); \
eng = OptikalFeatureEngineer(); eng.load_scaler(f'{model_dir}/optikal_scaler.pkl'); \
from sklearn.ensemble import IsolationForest; \
import joblib; raw = joblib.load(f'{model_dir}/optikal_isolation_forest.pkl'); \
if_model = OptikalIsolationForest(); if_model.model = raw; if_model.fitted = True; \
cfg = KafkaConfig(bootstrap_servers=os.environ.get('KAFKA_BOOTSTRAP_SERVERS','localhost:9092')); \
OptikalKafkaConsumer(eng, if_model, cfg).run() \
"]
