# Optikal

**GPU-Accelerated AI Threat Detection Model for AI Agent Security**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/Cogensec/Optikal/actions/workflows/ci.yml/badge.svg)](https://github.com/Cogensec/Optikal/actions/workflows/ci.yml)

Optikal is a production-grade machine learning model for detecting security threats in AI agent behavior. It combines five complementary models — Isolation Forest, attention-based LSTM, Gradient Boosting, a multi-class ThreatClassifier, and a learned MetaLearner ensemble — to achieve high-accuracy, low-latency threat detection with calibrated uncertainty estimates.

## Overview

Optikal is part of the ARGUS AI Agent Security Platform and analyzes behavioral metrics, resource usage patterns, and temporal sequences to identify:

- **Credential Abuse**: Excessive API usage, repetitive low-diversity queries, high error rates
- **Data Exfiltration**: Unusual network volume, heavy file access, night-time activity preference
- **Resource Abuse**: CPU/memory spikes, compute hijacking, thousands of API calls per minute
- **Behavioral Drift**: Gradual, slow-burn deviation from an agent's established baseline
- **Prompt Injection**: Adversarial input probing — high query diversity, many rejected calls, bimodal response times
- **Privilege Escalation**: Two-phase attack — recon (access-denied flood) followed by exploitation spike

## Architecture

Optikal uses a five-model ensemble pipeline:

### 1. Isolation Forest — Unsupervised Anomaly Detection
- Detects statistical outliers without requiring labeled threat data
- Fast CPU inference (~2 ms per batch)
- Contamination and tree count tunable via Optuna hyperparameter search
- Accuracy: ~88–92%

### 2. LSTM with Bahdanau Attention — Sequential Analysis
- Analyzes 10-timestep behavioral windows to detect threats invisible to point-in-time models
- Bahdanau attention mechanism surfaces the most anomalous timesteps for interpretability
- Monte Carlo Dropout (50 forward passes) produces calibrated `confidence` and `lstm_uncertainty` scores
- GPU-accelerated inference (~5 ms)
- Accuracy: ~93–97%

### 3. Gradient Boosting — Supervised Classification
- Fully supervised third ensemble member; leverages labeled `is_threat` and `threat_type` columns
- Uses LightGBM when available; falls back to sklearn `GradientBoostingClassifier`
- 200 estimators, lr=0.05; early stopping on validation loss
- Captures non-linear feature interactions that IF and LSTM miss

### 4. ThreatClassifier — Multi-Class Threat Identification
- `RandomForestClassifier` trained on threat samples only
- Answers *what kind* of threat was detected across all 6 threat classes
- Enables differentiated operator responses (block outbound for exfiltration vs. throttle compute for resource abuse)

### 5. MetaLearner Ensemble — Learned Fusion
- `LogisticRegression` trained on out-of-validation predictions from all base models
- Learns optimal per-model weighting empirically rather than using fixed 40/60 ratios
- Falls back to static weights when not fitted
- Final ensemble accuracy: ~95–98%

## Quick Start

### Installation

```bash
git clone https://github.com/Cogensec/Optikal.git
cd Optikal

# Core dependencies only
pip install -e .

# With all extras (LSTM + GBM + explainability + tracking + tuning + streaming)
pip install -e ".[all]"
```

### Training

```bash
cd optikal

# Quick start — Isolation Forest only, no GPU required
python train_quick.py

# Full pipeline — IF + Attention LSTM + GBM + ThreatClassifier + MetaLearner
python optikal_trainer.py

# Full pipeline with Optuna hyperparameter search and MLflow tracking
python -c "
from optikal_trainer import train_optikal_model
train_optikal_model(tune=True, use_mlflow=True, n_hp_trials=50)
"
```

### Inference

```python
import joblib
from optikal.feature_engineering import OptikalFeatureEngineer
from optikal.optikal_trainer import OptikalIsolationForest, OptikalGBM, OptikalEnsemble

# Load artefacts
engineer = OptikalFeatureEngineer()
engineer.load_scaler("optikal_models/optikal_scaler.pkl")

if_model        = OptikalIsolationForest()
if_model.model  = joblib.load("optikal_models/optikal_isolation_forest.pkl")
if_model.fitted = True

gbm_model = OptikalGBM.load("optikal_models/optikal_gbm.pkl")
ensemble  = OptikalEnsemble(if_model, gbm=gbm_model)

# Prepare features and score
features   = engineer.extract_features(agent_df)
X_scaled,_ = engineer.transform(features)

preds, detail = ensemble.predict(X_scaled, mc_passes=50)

print("Threat score:  ", detail["ensemble"])
print("Confidence:    ", detail.get("confidence"))    # 1 - MC Dropout std
print("Threat type:   ", detail.get("threat_type"))   # e.g. "credential_abuse"
```

## Features

**18 Security-Relevant Features** extracted from 11 raw behavioral metrics:

| Category | Features |
|----------|----------|
| Rate-based (4) | `api_calls_per_min`, `errors_per_min`, `file_ops_per_min`, `network_calls_per_min` |
| Resource (3) | `cpu_usage`, `memory_usage_gb`, `resource_intensity` |
| Behavioral (3) | `error_rate`, `query_diversity_score`, `success_rate_score` |
| Temporal (4) | `is_night`, `is_weekend`, `hour_sin`, `hour_cos` |
| Response time (2) | `response_time_sec`, `response_time_log` |
| Composite (2) | `activity_score`, `suspicion_score` |

## Training Data

Optikal trains on synthetic scenarios covering all six attack types:

| Scenario | Samples | Key signals |
|----------|---------|-------------|
| Normal Behavior | 5,000 | Baseline metrics |
| Credential Abuse | 500 | 500 API/min, low diversity, high errors |
| Data Exfiltration | 500 | 500 network calls, night hours |
| Resource Abuse | 500 | CPU ≈90%, 1,000 API/min |
| Behavioral Drift | 500 | Progressive deviation over time |
| Prompt Injection | 500 | Diversity ≈0.9, 25 errors, bimodal latency |
| Privilege Escalation | 500 | Recon → exploitation phase transition |
| **Total** | **8,000** | |

The `add_noise()` and `temporal_span_days` parameters add sensor realism and multi-month temporal variation. The Active Learning pipeline closes the gap to real production data over time.

## Performance

| Metric | Target | Achieved (Synthetic Data) |
|--------|--------|---------------------------|
| Accuracy | >95% | ~96% (ensemble) |
| Precision | >90% | ~95% |
| Recall | >95% | ~97% |
| False Positive Rate | <2% | ~1.5% |
| Inference Latency (GPU) | <10 ms | ~7 ms (ensemble) |
| Throughput | >10K/sec | ~15K/sec |

## Uncertainty Quantification

The LSTM supports Monte Carlo Dropout for calibrated confidence estimates:

```python
mean_scores, std_scores = lstm_model.predict_score_with_uncertainty(X_seq, n_passes=50)
confidence = 1.0 - std_scores   # 1.0 = fully certain, 0.0 = maximally uncertain
```

Uncertainty is also surfaced automatically in `OptikalEnsemble.predict(mc_passes=50)` via `detail["confidence"]` and `detail["lstm_uncertainty"]`.

## Hyperparameter Tuning

Optuna-driven Bayesian search over Isolation Forest contamination, n_estimators, and decision threshold:

```python
from optikal.hyperparameter_search import run_hyperparameter_search

results = run_hyperparameter_search(X_train, y_train, X_val, y_val, n_trials=50)
print(results["best_params"])   # {"contamination": 0.12, "n_estimators": 200, "threshold": 0.47}
print(results["best_value"])    # 0.963 (F1)
```

Or enable it directly in the training pipeline: `train_optikal_model(tune=True)`.

## Feature Drift Monitoring

`FeatureDriftDetector` tracks Population Stability Index (PSI) per feature between training data and live inference inputs. PSI >= 0.2 flags significant drift before it silently degrades accuracy:

```python
from optikal.drift_detector import FeatureDriftDetector

detector = FeatureDriftDetector(feature_names, window_size=500)
detector.fit_reference(X_train)

# During inference — auto-checks when 500 samples have buffered
detector.update(X_batch)
```

## Active Learning

`ActiveLearningBuffer` routes high-uncertainty inputs to human analysts. Once `min_labeled` samples are labeled, the retrain callback fires automatically:

```python
from optikal.active_learning import ActiveLearningBuffer

buffer = ActiveLearningBuffer(
    uncertainty_threshold=0.3,
    min_labeled=200,
    retrain_callback=my_retrain_fn,
)
buffer.add_uncertain(X_batch, scores, sample_ids)
buffer.add_label("sample_123", label=1, threat_type="credential_abuse")
print(buffer.stats)
```

## Kafka Stream Integration

`OptikalKafkaConsumer` connects to ARGUS Kafka topics, runs inference on every event, and publishes structured threat alerts:

```python
from optikal.kafka_consumer import KafkaConfig, OptikalKafkaConsumer

consumer = OptikalKafkaConsumer(
    engineer=engineer,
    if_model=if_model,
    config=KafkaConfig(bootstrap_servers="kafka:9092"),
    threat_classifier=threat_clf,
)
consumer.run()   # Blocking; Ctrl-C to stop
```

Input topics: `argus.agent.metrics`, `argus.io.validation`, `argus.policy.decisions`
Output topic: `argus.threat.alerts`

## Deployment

### Docker

```bash
# Train models
docker build --target trainer -t optikal:trainer .
docker run -v $(pwd)/optikal_models:/app/optikal_models optikal:trainer

# Run inference consumer
docker build --target inference -t optikal:inference .
docker run -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 optikal:inference

# Full dev stack — trainer + inference + Kafka + MLflow + ZooKeeper
docker compose up
```

### NVIDIA Triton Inference Server

```bash
pip install -e ".[onnx]"
cd optikal && python export_onnx.py

docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/triton_models:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models
```

### MLflow Experiment Tracking

MLflow tracking is built into the training pipeline — no boilerplate required:

```python
train_optikal_model(use_mlflow=True)
# Then: mlflow ui  →  http://localhost:5000
```

## Repository Structure

```
Optikal/
├── Dockerfile                          # Multi-stage: base / trainer / inference
├── docker-compose.yml                  # Full dev stack (Kafka + MLflow + ZooKeeper)
├── .dockerignore
├── .github/
│   └── workflows/
│       └── ci.yml                      # lint / typecheck / test / smoke_train / onnx_export
├── setup.py                            # Package setup + extras groups
├── pytest.ini
├── README.md                           # This file
├── LICENSE
├── DEPLOYMENT.md
├── optikal/
│   ├── __init__.py                     # Package init + configure_logging()
│   ├── config.py                       # Typed dataclass configuration (JSON/YAML I/O)
│   ├── data_generator.py               # Synthetic data — 6 threat scenarios
│   ├── feature_engineering.py          # 18-feature extraction, validation, scaler
│   ├── optikal_trainer.py              # All 5 model classes + training pipeline
│   ├── train_quick.py                  # Quick-start: Isolation Forest only
│   ├── export_onnx.py                  # ONNX export for Triton
│   ├── explainability.py               # SHAP-based feature attribution
│   ├── drift_detector.py               # PSI feature drift monitoring
│   ├── hyperparameter_search.py        # Optuna hyperparameter optimization
│   ├── kafka_consumer.py               # ARGUS Kafka consumer + alert publisher
│   ├── active_learning.py              # Uncertainty buffer + retrain trigger
│   └── requirements.txt
├── tests/
│   ├── conftest.py
│   ├── test_data_generator.py
│   ├── test_feature_engineering.py
│   └── test_trainer.py
└── examples/
    └── inference_example.py
```

## Integration with ARGUS

1. Deploy ARGUS services (ASR, ASG, PDE) — Kafka producers are already wired
2. Deploy Optikal models to Triton or run the inference Docker image
3. Start `OptikalKafkaConsumer` to bridge ARGUS topics to Optikal
4. Threat alerts flow to `argus.threat.alerts` for downstream response automation

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/), [TensorFlow](https://www.tensorflow.org/), [LightGBM](https://lightgbm.readthedocs.io/), and [NVIDIA Triton](https://github.com/triton-inference-server)
- Hyperparameter search powered by [Optuna](https://optuna.org/)
- Explainability via [SHAP](https://shap.readthedocs.io/)
- Part of the ARGUS AI Agent Security Platform by Cogensec

## Contact

**Cogensec**
- Website: [cogensec.com](https://cogensec.com)
- GitHub: [@Cogensec](https://github.com/Cogensec)
- Email: info@cogensec.com

## Roadmap

- [x] Isolation Forest anomaly detection
- [x] LSTM with Bahdanau attention
- [x] Gradient Boosting third ensemble member
- [x] Multi-class ThreatClassifier (6 threat types)
- [x] MetaLearner learned ensemble fusion
- [x] SHAP explainability
- [x] Prompt Injection + Privilege Escalation threat generators
- [x] Feature drift detection (PSI)
- [x] Monte Carlo Dropout uncertainty quantification
- [x] Optuna hyperparameter search
- [x] Active learning pipeline
- [x] Kafka stream integration (ARGUS)
- [x] Docker / docker-compose deployment
- [x] GitHub Actions CI/CD (lint / test / smoke_train / onnx_export)
- [ ] Pre-trained models with production threat data
- [ ] Kubernetes Helm chart
- [ ] Integration with popular SIEM platforms

---

**Optikal** — GPU-Accelerated AI Threat Detection for AI Agent Security
