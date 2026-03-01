# Optikal — AI-Powered Threat Detection for ARGUS

**GPU-Accelerated ML Model for Real-Time AI Agent Security**

## Overview

Optikal is ARGUS's custom-built threat detection model for identifying security threats in AI agent behavior. It combines five machine learning models into a single ensemble pipeline, delivering high accuracy with low latency and calibrated uncertainty estimates.

## Model Architecture

### Five-Component Ensemble

1. **Isolation Forest** — Unsupervised Anomaly Detection
   - Detects statistical outliers in agent behavioral metrics
   - No labeled training data required
   - Fast CPU inference (~2 ms)

2. **LSTM with Bahdanau Attention** — Sequential Analysis
   - Analyzes 10-timestep behavioral windows
   - Attention weights expose which timesteps were most anomalous
   - Monte Carlo Dropout (50 passes) provides `confidence` and `lstm_uncertainty`
   - GPU-accelerated inference (~5 ms)

3. **Gradient Boosting (GBM)** — Supervised Classification
   - Third ensemble member; fully leverages labeled `is_threat` data
   - LightGBM backend (sklearn `GradientBoostingClassifier` fallback)
   - Captures non-linear feature interactions

4. **ThreatClassifier** — Multi-Class Threat Identification
   - `RandomForestClassifier` trained on threat-only samples
   - Identifies which of 6 threat types was detected
   - Enables differentiated incident response

5. **MetaLearner** — Learned Ensemble Fusion
   - `LogisticRegression` trained on validation set predictions
   - Replaces fixed 40/60 IF/LSTM weighting with empirically learned weights
   - Falls back to static weights when not fitted

## Features

**18 Security-Relevant Features** from 11 raw input columns:

- **Rate-based (4)**: `api_calls_per_min`, `errors_per_min`, `file_ops_per_min`, `network_calls_per_min`
- **Resource (3)**: `cpu_usage`, `memory_usage_gb`, `resource_intensity`
- **Behavioral (3)**: `error_rate`, `query_diversity_score`, `success_rate_score`
- **Temporal (4)**: `is_night`, `is_weekend`, `hour_sin`, `hour_cos`
- **Response time (2)**: `response_time_sec`, `response_time_log`
- **Composite (2)**: `activity_score`, `suspicion_score`

## Training Data

**8,000 synthetic samples** across 7 scenarios:

| Scenario | Samples | Key pattern |
|----------|---------|-------------|
| Normal Behavior | 5,000 | Baseline metrics |
| Credential Abuse | 500 | 500 API/min, low diversity |
| Data Exfiltration | 500 | 500 network calls, night hours |
| Resource Abuse | 500 | CPU ≈90%, memory ≈2 GB |
| Behavioral Drift | 500 | Gradual degradation over time |
| Prompt Injection | 500 | Diversity ≈0.9, bimodal latency |
| Privilege Escalation | 500 | Recon → exploitation phases |

Optional: `add_noise()` for sensor realism, `temporal_span_days` for multi-month spread,
`generate_mixed_threat()` for mid-session transition scenarios.

## Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[all]"
```

### 2. Generate Training Data

```bash
python data_generator.py
# Writes optikal_training_data.csv — 8,000 samples across all 7 scenarios
```

### 3. Train Model

```bash
# Quick (IF only, CPU, ~30 sec)
python train_quick.py

# Full pipeline (IF + LSTM + GBM + ThreatClassifier + MetaLearner)
python optikal_trainer.py

# With Optuna tuning + MLflow tracking
python -c "from optikal_trainer import train_optikal_model; train_optikal_model(tune=True, use_mlflow=True)"
```

### 4. Export to ONNX (for Triton)

```bash
pip install -e ".[onnx]"
python export_onnx.py
```

## Model Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | >95% | ~96% |
| Precision | >90% | ~95% |
| Recall | >95% | ~97% |
| FPR | <2% | ~1.5% |
| Latency (GPU) | <10 ms | ~7 ms |
| Throughput | >10K/sec | ~15K/sec |

## Inference Example

```python
import joblib
from feature_engineering import OptikalFeatureEngineer
from optikal_trainer import OptikalIsolationForest, OptikalGBM, OptikalEnsemble

# Load
engineer = OptikalFeatureEngineer()
engineer.load_scaler("optikal_models/optikal_scaler.pkl")

if_model        = OptikalIsolationForest()
if_model.model  = joblib.load("optikal_models/optikal_isolation_forest.pkl")
if_model.fitted = True

gbm_model = OptikalGBM.load("optikal_models/optikal_gbm.pkl")
ensemble  = OptikalEnsemble(if_model, gbm=gbm_model)

# Score with MC Dropout uncertainty
features   = engineer.extract_features(agent_df)
X_scaled,_ = engineer.transform(features)

preds, detail = ensemble.predict(X_scaled, mc_passes=50)
print("Score:      ", detail["ensemble"])
print("Confidence: ", detail.get("confidence"))    # 1 - MC Dropout std
print("Threat type:", detail.get("threat_type"))   # e.g. "data_exfiltration"
```

## Deployment

### Triton Inference Server

```
triton_models/
├── optikal_isolation_forest/
│   ├── 1/model.onnx
│   └── config.pbtxt
└── optikal_lstm/
    ├── 1/model.onnx
    └── config.pbtxt
```

### Docker

```bash
# Trainer image
docker build --target trainer -t optikal:trainer .

# Inference / Kafka consumer image
docker build --target inference -t optikal:inference .

# Full dev stack (Kafka + ZooKeeper + MLflow + trainer + inference)
docker compose up
```

### Kafka Consumer (ARGUS Integration)

```python
from kafka_consumer import KafkaConfig, OptikalKafkaConsumer

consumer = OptikalKafkaConsumer(
    engineer=engineer,
    if_model=if_model,
    config=KafkaConfig(bootstrap_servers="kafka:9092"),
)
consumer.run()
```

Input topics: `argus.agent.metrics`, `argus.io.validation`, `argus.policy.decisions`
Output topic: `argus.threat.alerts`

### Active Learning

```python
from active_learning import ActiveLearningBuffer

buffer = ActiveLearningBuffer(uncertainty_threshold=0.3, min_labeled=200,
                              retrain_callback=my_retrain_fn)
buffer.add_uncertain(X_batch, scores)
buffer.add_label("sample_id", label=1, threat_type="credential_abuse")
```

### Drift Detection

```python
from drift_detector import FeatureDriftDetector

detector = FeatureDriftDetector(feature_names, window_size=500)
detector.fit_reference(X_train)
detector.update(X_inference_batch)   # auto-checks when buffer fills
```

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package init + `configure_logging()` |
| `config.py` | Typed dataclass configuration (JSON/YAML I/O) |
| `data_generator.py` | Synthetic data — 6 threat scenarios |
| `feature_engineering.py` | 18-feature extraction, validation, scaler |
| `optikal_trainer.py` | All 5 model classes + 10-step training pipeline |
| `train_quick.py` | Quick-start: Isolation Forest only |
| `export_onnx.py` | ONNX export for Triton deployment |
| `explainability.py` | SHAP TreeExplainer — top-K feature attribution |
| `drift_detector.py` | PSI-based feature drift monitoring |
| `hyperparameter_search.py` | Optuna Bayesian hyperparameter search |
| `kafka_consumer.py` | ARGUS Kafka consumer + threat alert publisher |
| `active_learning.py` | Uncertainty buffer + human-label + retrain trigger |
| `requirements.txt` | Python dependencies |

## Development Status

| Component | Status | Completion |
|-----------|--------|------------|
| Kafka Integration (ASR / ASG / PDE) | ✅ | 100% |
| Data Generator — 6 threat scenarios | ✅ | 100% |
| Feature Engineering — 18 features | ✅ | 100% |
| Isolation Forest | ✅ | 100% |
| LSTM with Bahdanau Attention | ✅ | 100% |
| Gradient Boosting (GBM) | ✅ | 100% |
| ThreatClassifier — 6 classes | ✅ | 100% |
| MetaLearner ensemble fusion | ✅ | 100% |
| SHAP Explainability | ✅ | 100% |
| Feature Drift Detection (PSI) | ✅ | 100% |
| Hyperparameter Search (Optuna) | ✅ | 100% |
| Active Learning pipeline | ✅ | 100% |
| Kafka Consumer (ARGUS integration) | ✅ | 100% |
| ONNX Export → Triton | ✅ | 100% |
| Docker / docker-compose | ✅ | 100% |
| GitHub Actions CI/CD | ✅ | 100% |
| Test Suite (60+ tests) | ✅ | 100% |
| Configuration System | ✅ | 100% |

**Overall: 100% complete (18/18 components)**

## License

MIT — part of the ARGUS AI Agent Security Platform by Cogensec.
