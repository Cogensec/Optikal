# Optikal Deployment Guide

## Quick Deployment to GitHub

### 1. Initialize Git Repository

```bash
cd optikal-repo
git init
git add .
git commit -m "Initial commit: Optikal v2.0.0"
```

### 2. Connect to GitHub

```bash
git remote add origin https://github.com/Cogensec/Optikal.git
git branch -M main
git push -u origin main
```

### 3. Create Release

```bash
git tag -a v2.0.0 -m "Optikal v2.0.0 — full five-model ensemble"
git push origin v2.0.0
```

## Installation from GitHub

```bash
# Core install
pip install git+https://github.com/Cogensec/Optikal.git

# With specific extras
pip install "optikal[lstm,gbm,explain,tracking,streaming] @ git+https://github.com/Cogensec/Optikal.git"

# Everything
pip install "optikal[all] @ git+https://github.com/Cogensec/Optikal.git"
```

## Docker Deployment

The repository ships a real multi-stage `Dockerfile` with three targets:

| Stage | Purpose | Key extras installed |
|-------|---------|----------------------|
| `base` | Shared Python + system deps | `numpy`, `pandas`, `scikit-learn`, `joblib` |
| `trainer` | Full training environment | `tensorflow`, `shap`, `mlflow`, `optuna` |
| `inference` | Lean Kafka consumer image | `confluent-kafka`, `shap` |

### Building

```bash
# Trainer image
docker build --target trainer -t cogensec/optikal:trainer .

# Inference / Kafka consumer image
docker build --target inference -t cogensec/optikal:inference .
```

### Running

```bash
# Train and write models to a named volume
docker run \
    -v optikal_models:/app/optikal_models \
    -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
    cogensec/optikal:trainer

# Start the Kafka inference consumer
docker run \
    -v optikal_models:/app/optikal_models:ro \
    -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
    cogensec/optikal:inference
```

### Full Dev Stack (docker-compose)

`docker-compose.yml` at the repo root brings up the complete development environment:

```
Services:
  trainer      — trains Optikal models, writes to shared model_store volume
  inference    — Kafka consumer running live inference
  mlflow       — experiment tracking UI at http://localhost:5000
  kafka        — Confluent Kafka 7.5.0
  zookeeper    — ZooKeeper (required by Kafka)
```

```bash
# Start everything
docker compose up

# Train only
docker compose up trainer

# Start inference consumer (after models are trained)
docker compose up inference

# View MLflow experiments
docker compose up mlflow
# Then open http://localhost:5000
```

## NVIDIA Triton Inference Server

Export models to ONNX for GPU deployment:

```bash
pip install -e ".[onnx]"
cd optikal && python export_onnx.py
```

Triton model repository structure created by `export_onnx.py`:

```
triton_models/
├── optikal_isolation_forest/
│   ├── 1/model.onnx
│   └── config.pbtxt          # max_batch_size=1000, KIND_GPU
└── optikal_lstm/
    ├── 1/model.onnx
    └── config.pbtxt          # dynamic_batching: [100, 500, 1000]
```

Start Triton:

```bash
docker run --gpus all \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/triton_models:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models
```

Test readiness:

```bash
curl http://localhost:8000/v2/models/optikal_isolation_forest/ready
```

## PyPI Distribution

```bash
python setup.py sdist bdist_wheel
pip install twine
twine upload dist/*
```

Then: `pip install optikal`

## Kubernetes Deployment

```yaml
# k8s/optikal-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: optikal-inference
spec:
  replicas: 2
  selector:
    matchLabels:
      app: optikal
  template:
    metadata:
      labels:
        app: optikal
    spec:
      containers:
      - name: optikal
        image: cogensec/optikal:inference
        env:
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka-service:9092"
        - name: MODEL_DIR
          value: "/models"
        volumeMounts:
        - name: model-store
          mountPath: /models
          readOnly: true
      volumes:
      - name: model-store
        persistentVolumeClaim:
          claimName: optikal-models-pvc
```

```bash
kubectl apply -f k8s/optikal-deployment.yaml
```

## CI/CD — GitHub Actions

`.github/workflows/ci.yml` ships with the repository and runs automatically on every push and pull request. It defines five jobs:

| Job | Trigger | What it does |
|-----|---------|--------------|
| `lint` | Every push/PR | `black --check` + `flake8` (max 100 chars) |
| `typecheck` | Every push/PR | `mypy` with `--ignore-missing-imports` |
| `test` | Every push/PR | `pytest tests/ --cov=optikal --cov-fail-under=75` across Python 3.9/3.10/3.11 |
| `smoke_train` | Every push/PR (after test) | Mini dataset (300 normal + 50×6 threats) end-to-end training run |
| `onnx_export` | Tagged releases only | Exports IF + LSTM to ONNX, uploads as release artifact |

No additional configuration is needed — the workflow file is already in place.

## MLflow Experiment Tracking

MLflow tracking is integrated directly into the training pipeline:

```python
from optikal.optikal_trainer import train_optikal_model

train_optikal_model(
    data_path="optikal_training_data.csv",
    output_dir="optikal_models",
    use_mlflow=True,       # enables tracking
    tune=True,             # also logs Optuna best params
    n_hp_trials=50,
)
```

What gets logged automatically:
- **Params**: `n_features`, `training_samples`, `contamination`, `n_estimators`, Optuna best params
- **Metrics**: `if_f1`, `if_roc_auc`, `lstm_f1`, `lstm_roc_auc`, `gbm_f1`, `gbm_roc_auc`, `ensemble_f1`, `ensemble_roc_auc`
- **Artifacts**: `isolation_forest` and `gbm` sklearn models

Start the UI:

```bash
mlflow ui     # http://localhost:5000
```

Or use the MLflow service in `docker-compose.yml` which starts at `http://localhost:5000` automatically.

## Model Versioning with Git LFS

For large model files:

```bash
git lfs install
git lfs track "*.pkl" "*.h5" "*.onnx"
git add .gitattributes
git commit -m "Add Git LFS tracking for model files"
```

## Security Best Practices

1. Never commit API keys or credentials — use environment variables or secrets managers
2. Sign model artefacts before deployment (`sha256sum` + store hash in metadata)
3. Restrict Triton endpoints with a reverse proxy / API gateway
4. Audit all inference predictions via structured logging (`configure_logging(format="json")`)
5. Rate-limit the Kafka consumer topic to prevent alert flooding
6. Rotate model artefacts on drift detection (PSI >= 0.2 on any feature)
