"""
Quick Start Training Script for Optikal (Isolation Forest Only)

Trains just the Isolation Forest model without requiring TensorFlow.
This provides a faster path to get a working threat detection model.

Usage:
    python train_quick.py          # Run directly
    optikal-train                  # Via installed entry point
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

# Allow imports from the optikal package when run directly
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


def main():
    """Entry point for quick Isolation Forest training."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 70)
    logger.info("OPTIKAL QUICK START — ISOLATION FOREST TRAINING")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data
    # ------------------------------------------------------------------
    logger.info("[1/4] Generating synthetic training data...")
    from data_generator import SyntheticDataGenerator

    generator = SyntheticDataGenerator(seed=42)
    dataset = generator.generate_training_dataset(
        n_normal=2000,          # Smaller dataset for quick training
        n_threats_per_type=200,
    )

    n_normal  = int((dataset['is_threat'] == 0).sum())
    n_threats = int((dataset['is_threat'] == 1).sum())
    logger.info("Generated %d samples — normal: %d, threats: %d",
                len(dataset), n_normal, n_threats)

    # ------------------------------------------------------------------
    # Step 2: Feature engineering
    # ------------------------------------------------------------------
    logger.info("[2/4] Engineering features...")
    from feature_engineering import OptikalFeatureEngineer, prepare_training_data

    tmp_csv = "optikal_quick_data.csv"
    dataset.to_csv(tmp_csv, index=False)

    train_df, val_df, test_df = prepare_training_data(
        tmp_csv, test_size=0.2, val_size=0.1
    )

    engineer = OptikalFeatureEngineer()
    train_features = engineer.extract_features(train_df)
    test_features  = engineer.extract_features(test_df)

    X_train, feature_names = engineer.fit_transform(train_features)
    X_test, _              = engineer.transform(test_features)

    y_train = train_features['is_threat'].values
    y_test  = test_features['is_threat'].values

    logger.info("Extracted %d features — train: %d, test: %d",
                len(feature_names), X_train.shape[0], X_test.shape[0])

    # ------------------------------------------------------------------
    # Step 3: Train Isolation Forest
    # ------------------------------------------------------------------
    logger.info("[3/4] Training Isolation Forest...")

    model = IsolationForest(
        contamination=0.3,   # Expect ~30% anomalies in this synthetic dataset
        random_state=42,
        n_estimators=100,
        max_samples=256,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_train)
    logger.info("Training complete")

    # ------------------------------------------------------------------
    # Step 4: Evaluate with threshold search
    # ------------------------------------------------------------------
    logger.info("[4/4] Evaluating model...")

    scores_raw = model.decision_function(X_test)
    scores     = 1.0 / (1.0 + np.exp(scores_raw))  # Sigmoid normalisation

    thresholds = [0.4, 0.5, 0.6, 0.7]
    best_threshold = 0.5
    best_f1 = 0.0

    logger.info("Threshold optimisation:")
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        logger.info("  threshold=%.1f  F1=%.4f  Precision=%.4f  Recall=%.4f",
                    thr, f1, prec, rec)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr

    y_pred = (scores >= best_threshold).astype(int)
    logger.info("Best threshold: %.1f", best_threshold)
    logger.info("=== FINAL PERFORMANCE ===")
    logger.info("  Accuracy:  %.4f", accuracy_score(y_test, y_pred))
    logger.info("  Precision: %.4f", precision_score(y_test, y_pred, zero_division=0))
    logger.info("  Recall:    %.4f", recall_score(y_test, y_pred, zero_division=0))
    logger.info("  F1 Score:  %.4f", f1_score(y_test, y_pred, zero_division=0))
    logger.info("  ROC-AUC:   %.4f", roc_auc_score(y_test, scores))

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    output_dir = Path("optikal_models")
    output_dir.mkdir(exist_ok=True)

    joblib.dump(model, output_dir / "optikal_isolation_forest.pkl")
    engineer.save_scaler(str(output_dir / "optikal_scaler.pkl"))

    metadata = {
        "model_name":      "Optikal Isolation Forest",
        "version":         "1.0.0",
        "features":        feature_names,
        "n_features":      len(feature_names),
        "best_threshold":  best_threshold,
        "performance": {
            "accuracy":  float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score":  float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc":   float(roc_auc_score(y_test, scores)),
        },
        "training_samples": len(train_df),
        "test_samples":     len(test_df),
    }

    with open(output_dir / "optikal_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model saved to %s/", output_dir)
    logger.info("  optikal_isolation_forest.pkl")
    logger.info("  optikal_scaler.pkl")
    logger.info("  optikal_metadata.json")

    logger.info("=" * 70)
    logger.info("OPTIKAL QUICK START COMPLETE!")
    logger.info("=" * 70)
    logger.info("Next steps:")
    logger.info("  Inference:    python examples/inference_example.py")
    logger.info("  LSTM model:   pip install tensorflow && python optikal/optikal_trainer.py")
    logger.info("  Triton export: python optikal/export_onnx.py")


if __name__ == "__main__":
    main()
