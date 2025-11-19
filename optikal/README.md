# Optikal - AI-Powered Threat Detection for ARGUS

**GPU-Accelerated ML Model for Real-Time AI Agent Security**

## Overview

Optikal is ARGUS's custom-built threat detection model designed specifically for identifying security threats in AI agent behavior. The model combines multiple machine learning techniques to achieve high accuracy with low latency.

## Model Architecture

### Three-Component Ensemble

1. **Optikal Isolation Forest** (Anomaly Detection)
   - Detects statistical outliers in agent behavior
   - Unsupervised learning on behavioral metrics
   - Fast inference (~2ms on CPU)

2. **Optikal LSTM** (Sequential Analysis)
   - Analyzes temporal behavior sequences
   - Detects patterns invisible to point-in-time analysis
   - GPU-accelerated inference (~5ms)

3. **Optikal Ensemble** (Combined Scoring)
   - Weighted fusion of both models
   - Multi-factor risk assessment
   - Final threat score with confidence

## Features

**18 Security-Relevant Features**:
- Rate-based: API calls/min, errors/min, file ops/min
- Resource: CPU usage, memory usage, resource intensity
- Behavioral: Error rate, query diversity, success rate
- Temporal: Night activity, weekend patterns, cyclical time
- Composite: Activity score, suspicion score

## Training Data

**Synthetic Threat Scenarios**:
- Normal Behavior (Baseline)
- Credential Abuse (excessive API usage)
- Data Exfiltration (high network activity)
- Resource Abuse (CPU/memory spikes)
- Behavioral Drift (gradual deviation from normal)

## Quick Start

### 1. Install Dependencies

```bash
cd ml_models/optikal
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
python data_generator.py
```

Generates `optikal_training_data.csv` with 7,000 samples (5K normal, 2K threats).

### 3. Train Model

```bash
python optikal_trainer.py
```

Trains both Isolation Forest and LSTM, saves to `optikal_models/` directory.

### 4. Export to ONNX (for Triton)

```bash
python export_onnx.py
```

Exports models to ONNX format for GPU deployment with NVIDIA Triton.

## Model Performance

**Target Metrics**:
- Accuracy: >95%
- Precision: >90% (low false positives)
- Recall: >95% (catch most threats)
- Inference Latency: <10ms (GPU)
- Throughput: >10K inferences/sec

## Deployment

### Triton Inference Server

Models are exported to ONNX and deployed via NVIDIA Triton for GPU-accelerated inference:

```
models/
├── optikal_isolation_forest/
│   ├── 1/
│   │   └── model.onnx
│   └── config.pbtxt
└── optikal_lstm/
    ├── 1/
    │   └── model.onnx
    └── config.pbtxt
```

### Integration with Morpheus

Optikal is integrated into the Morpheus Threat Detection Pipeline:

```
Kafka (agent metrics) → Morpheus Pipeline → Optikal Model → Threat Alert
```

## Files

- `__init__.py` - Package initialization
- `data_generator.py` - Synthetic training data generation
- `feature_engineering.py` - Feature extraction and scaling
- `optikal_trainer.py` - Model training pipeline
- `export_onnx.py` - ONNX export for Triton deployment
- `requirements.txt` - Python dependencies
- `STATUS.md` - Development status

## Usage Example

```python
from optikal_trainer import OptikalIsolationForest, OptikalLSTM
from feature_engineering import OptikalFeatureEngineer
import numpy as np

# Load trained models
if_model = OptikalIsolationForest()
# ... load from file ...

# Engineer features
engineer = OptikalFeatureEngineer()
features = engineer.extract_features(agent_data)
X = engineer.transform(features)

# Predict
threat_score = if_model.predict_anomaly_score(X)
if threat_score > 0.7:
    print(f"⚠️ High threat detected: {threat_score:.2f}")
```

## Development Status

- ✅ Synthetic data generation
- ✅ Feature engineering (18 features)
- ✅ Isolation Forest implementation
- ✅ LSTM implementation
- ✅ Ensemble fusion
- ⏳ ONNX export
- ⏳ Triton deployment
- ⏳ Production validation

## Performance Optimization

**GPU Acceleration**:
- Triton Inference Server
- ONNX Runtime with CUDA
- Batch inference (up to 1000 samples)
- Dynamic batching for optimal throughput

**Model Optimization**:
- ONNX quantization (INT8)
- Model pruning
- TensorRT optimization

## Further Improvements

1. **Active Learning**: Refine model with production threat data
2. **Feature Selection**: AutoML for optimal feature set
3. **Explainability**: SHAP values for threat attribution
4. **Drift Detection**: Monitor model performance degradation
5. **A/B Testing**: Compare model versions in production

## License

Proprietary - ARGUS Platform

## Contact

ARGUS Security Team  
For questions or issues with the Optikal model.
