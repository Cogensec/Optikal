"""
Quick Start Training Script for Optikal (Isolation Forest Only)

Trains just the Isolation Forest model without requiring TensorFlow.
This provides a faster path to get a working threat detection model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("OPTIKAL QUICK START - ISOLATION FOREST TRAINING")
print("=" * 70)

# Step 1: Generate synthetic data
print("\n[1/4] Generating synthetic training data...")
from data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(seed=42)
dataset = generator.generate_training_dataset(
    n_normal=2000,  # Smaller dataset for quick training
    n_threats_per_type=200
)

print(f"✓ Generated {len(dataset)} samples")
print(f"  - Normal: {(dataset['is_threat'] == 0).sum()}")
print(f"  - Threats: {(dataset['is_threat'] == 1).sum()}")

# Step 2: Feature engineering
print("\n[2/4] Engineering features...")
from feature_engineering import OptikalFeatureEngineer, prepare_training_data

# Save dataset
dataset.to_csv("optikal_quick_data.csv", index=False)

# Load and split
train_df, val_df, test_df = prepare_training_data("optikal_quick_data.csv", test_size=0.2, val_size=0.1)

# Extract features
engineer = OptikalFeatureEngineer()
train_features = engineer.extract_features(train_df)
test_features = engineer.extract_features(test_df)

# Scale
X_train, feature_names = engineer.fit_transform(train_features)
X_test, _ = engineer.transform(test_features)

y_train = train_features['is_threat'].values
y_test = test_features['is_threat'].values

print(f"✓ Extracted {len(feature_names)} features")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")

# Step 3: Train Isolation Forest
print("\n[3/4] Training Isolation Forest...")
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

model = IsolationForest(
    contamination=0.3,  # Expect 30% anomalies in our data
    random_state=42,
    n_estimators=100,
    max_samples=256,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train)
print("✓ Training complete")

# Step 4: Evaluate
print("\n[4/4] Evaluating model...")

# Predictions
scores_raw = model.decision_function(X_test)
# Normalize to [0, 1] (higher = more anomalous)
scores = 1 / (1 + np.exp(scores_raw))

# Try different thresholds
thresholds = [0.4, 0.5, 0.6, 0.7]
best_threshold = 0.5
best_f1 = 0

print("\nThreshold optimization:")
for threshold in thresholds:
    y_pred = (scores >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"  Threshold {threshold:.1f}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Final evaluation with best threshold
y_pred = (scores >= best_threshold).astype(int)

print(f"\n✓ Best threshold: {best_threshold}")
print("\n=== FINAL PERFORMANCE ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, scores):.4f}")

# Save model
print("\n[5/4] Saving model...")
output_dir = Path("optikal_models")
output_dir.mkdir(exist_ok=True)

joblib.dump(model, output_dir / "optikal_isolation_forest.pkl")
engineer.save_scaler(str(output_dir / "optikal_scaler.pkl"))

# Save metadata
import json
metadata = {
    "model_name": "Optikal Isolation Forest",
    "version": "1.0.0",
    "features": feature_names,
    "n_features": len(feature_names),
    "best_threshold": best_threshold,
    "performance": {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, scores))
    },
    "training_samples": len(train_df),
    "test_samples": len(test_df)
}

with open(output_dir / "optikal_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Model saved to: {output_dir}/")
print("  - optikal_isolation_forest.pkl")
print("  - optikal_scaler.pkl")
print("  - optikal_metadata.json")

print("\n" + "=" * 70)
print("✓ OPTIKAL QUICK START COMPLETE!")
print("=" * 70)
print("\nNext steps:")
print("1. Test inference: python test_inference.py")
print("2. For LSTM model: install tensorflow and run optikal_trainer.py")
print("3. Deploy to Triton: python export_onnx.py")
