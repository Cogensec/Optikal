# Optikal Deployment Guide

## Quick Deployment to GitHub

### 1. Initialize Git Repository

```bash
cd optikal-repo
git init
git add .
git commit -m "Initial commit: Optikal v1.0.0"
```

### 2. Connect to GitHub

```bash
# Add remote repository
git remote add origin https://github.com/Cogensec/Optikal.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 3. Create Release (Optional)

```bash
git tag -a v1.0.0 -m "Optikal v1.0.0 - Initial Release"
git push origin v1.0.0
```

## Installation from GitHub

Once pushed, users can install directly from GitHub:

```bash
# Install latest from main
pip install git+https://github.com/Cogensec/Optikal.git

# Install specific version
pip install git+https://github.com/Cogensec/Optikal.git@v1.0.0

# Install with extras
pip install "optikal[all] @ git+https://github.com/Cogensec/Optikal.git"
```

## PyPI Distribution (Future)

To distribute via PyPI:

```bash
# Build package
python setup.py sdist bdist_wheel

# Upload to PyPI
pip install twine
twine upload dist/*
```

Then users caninstall via:

```bash
pip install optikal
```

## Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY optikal/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY optikal/ ./optikal/
COPY setup.py .

# Install package
RUN pip install -e .

CMD ["python", "-m", "optikal.train_quick"]
```

Build and run:

```bash
docker build -t cogensec/optikal:1.0.0 .
docker run cogensec/optikal:1.0.0
```

## Kubernetes Deployment

Create `k8s/optikal-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: optikal-training
spec:
  replicas: 1
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
        image: cogensec/optikal:1.0.0
        resources:
          limits:
            nvidia.com/gpu: 1
```

Deploy:

```bash
kubectl apply -f k8s/optikal-deployment.yaml
```

## CI/CD with GitHub Actions

Create `.github/workflows/test.yml`:

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e .[dev]
      - run: pytest tests/
```

## Model Versioning

Use Git LFS for large model files:

```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "*.pkl"
git lfs track "*.h5"
git lfs track "*.onnx"

# Add .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

## Monitoring in Production

Use MLflow for experiment tracking:

```python
import mlflow

mlflow.start_run()
mlflow.log_params({"contamination": 0.3, "n_estimators": 100})
mlflow.log_metrics({"accuracy": 0.96, "f1_score": 0.95})
mlflow.sklearn.log_model(model, "model")
mlflow.end_run()
```

## Security Best Practices

1. **API Keys**: Never commit API keys - use environment variables
2. **Model Signing**: Sign models before deployment
3. **RBAC**: Implement role-based access control
4. **Audit Logs**: Log all model predictions
5. **Rate Limiting**: Prevent abuse of inference endpoints
