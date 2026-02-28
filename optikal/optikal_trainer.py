"""
Optikal Model Trainer

Trains the Optikal threat detection ensemble model:
  1. OptikalIsolationForest  — Unsupervised anomaly detection
  2. OptikalLSTM             — Sequential behavioural analysis
  3. OptikalEnsemble         — Combined threat scoring
  4. ThreatClassifier        — Multi-class threat type identification

Exports models to ONNX for Triton deployment.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)
from sklearn.preprocessing import LabelEncoder
import joblib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM model will be skipped.")

# ONNX export
try:
    import tf2onnx
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX tools not available. Model export will be skipped.")

from data_generator import SyntheticDataGenerator
from feature_engineering import OptikalFeatureEngineer, prepare_training_data


# =============================================================================
# Isolation Forest component
# =============================================================================

class OptikalIsolationForest:
    """
    Optikal Isolation Forest — Anomaly Detection Component.

    Provides unsupervised detection of behavioural outliers via scikit-learn's
    IsolationForest with sigmoid-normalised anomaly scores.
    """

    def __init__(self, contamination: float = 0.15, random_state: int = 42):
        """
        Args:
            contamination:  Expected proportion of anomalies in training data
            random_state:   Random seed for reproducibility
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples=256,
            n_jobs=-1,
        )
        self.contamination = contamination
        self.fitted = False

    def train(self, X_train: np.ndarray):
        """
        Fit the Isolation Forest on scaled training features.

        Args:
            X_train: Scaled feature array [n_samples, n_features]
        """
        logger.info(
            "Training OptikalIsolationForest — samples: %d, features: %d, "
            "contamination: %.2f",
            X_train.shape[0], X_train.shape[1], self.contamination,
        )
        self.model.fit(X_train)
        self.fitted = True
        logger.info("Isolation Forest training complete")

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Return sigmoid-normalised anomaly scores (higher = more anomalous).

        Args:
            X: Scaled feature array [n_samples, n_features]

        Returns:
            Anomaly scores in [0, 1]
        """
        if not self.fitted:
            raise ValueError("Model not trained yet")
        raw = self.model.decision_function(X)
        return 1.0 / (1.0 + np.exp(raw))  # Sigmoid: low decision → high score

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Return binary threat predictions.

        Args:
            X:         Scaled features
            threshold: Classification threshold

        Returns:
            Integer array (1 = anomaly/threat, 0 = normal)
        """
        return (self.predict_anomaly_score(X) >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Evaluate and log model performance metrics, return raw scores."""
        scores = self.predict_anomaly_score(X)
        y_pred = (scores >= threshold).astype(int)

        logger.info("=== Isolation Forest Evaluation ===")
        logger.info("  Accuracy:  %.4f", accuracy_score(y_true, y_pred))
        logger.info("  Precision: %.4f", precision_score(y_true, y_pred, zero_division=0))
        logger.info("  Recall:    %.4f", recall_score(y_true, y_pred, zero_division=0))
        logger.info("  F1 Score:  %.4f", f1_score(y_true, y_pred, zero_division=0))
        logger.info("  ROC-AUC:   %.4f", roc_auc_score(y_true, scores))

        return scores

    def save(self, filepath: str):
        """Serialize model to disk."""
        joblib.dump(self.model, filepath)
        logger.info("Isolation Forest saved to: %s", filepath)


# =============================================================================
# LSTM component
# =============================================================================

class OptikalLSTM:
    """
    Optikal LSTM — Sequential Behavioural Analysis Component.

    Analyses ordered time-series windows of agent metrics to detect
    sequential patterns indicative of threat activity.
    """

    def __init__(self, sequence_length: int = 10, n_features: int = 18):
        """
        Args:
            sequence_length: Number of timesteps per input sequence
            n_features:      Number of features per timestep
        """
        if not KERAS_AVAILABLE:
            raise ImportError("TensorFlow/Keras required for LSTM model")

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model: Optional[Any] = None  # keras.Model when TF available
        self.fitted = False

    def build_model(self) -> Any:  # -> keras.Model
        """Construct and compile the two-layer LSTM architecture."""
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.n_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1, activation='sigmoid'),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')],
        )
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
    ):
        """
        Fit the LSTM on pre-scaled sequential data.

        Args:
            X_train:    Training sequences [n_samples, sequence_length, n_features]
            y_train:    Training labels
            X_val:      Validation sequences
            y_val:      Validation labels
            epochs:     Maximum training epochs (early stopping may terminate sooner)
            batch_size: Mini-batch size
        """
        logger.info(
            "Training OptikalLSTM — sequences: %d, seq_length: %d, features: %d, "
            "epochs: %d, batch_size: %d",
            X_train.shape[0], X_train.shape[1], X_train.shape[2],
            epochs, batch_size,
        )

        self.model = self.build_model()

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1,
        )

        self.fitted = True
        logger.info("LSTM training complete")
        return history

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Return threat scores for input sequences.

        Args:
            X: Input sequences [n_samples, sequence_length, n_features]

        Returns:
            Scores in [0, 1]
        """
        if not self.fitted:
            raise ValueError("Model not trained yet")
        return self.model.predict(X, verbose=0).flatten()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_score(X) >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Evaluate and log model performance, return raw scores."""
        scores = self.predict_score(X)
        y_pred = (scores >= threshold).astype(int)

        logger.info("=== LSTM Evaluation ===")
        logger.info("  Accuracy:  %.4f", accuracy_score(y_true, y_pred))
        logger.info("  Precision: %.4f", precision_score(y_true, y_pred, zero_division=0))
        logger.info("  Recall:    %.4f", recall_score(y_true, y_pred, zero_division=0))
        logger.info("  F1 Score:  %.4f", f1_score(y_true, y_pred, zero_division=0))
        logger.info("  ROC-AUC:   %.4f", roc_auc_score(y_true, scores))

        return scores

    def save(self, filepath: str):
        """Save model in H5 format."""
        self.model.save(filepath)
        logger.info("LSTM model saved to: %s", filepath)


# =============================================================================
# Multi-class threat classifier
# =============================================================================

class ThreatClassifier:
    """
    Post-hoc multi-class classifier that identifies threat type.

    Trained on threat samples only, using the same 18-dimensional feature
    vector as the Isolation Forest. When the ensemble flags a threat, the
    ThreatClassifier answers *which kind* of threat it is, enabling
    differentiated operator responses (e.g. block outbound for DATA_EXFILTRATION
    vs. throttle compute for RESOURCE_ABUSE).
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        self.label_encoder = LabelEncoder()
        self.fitted = False
        self.classes_: Optional[np.ndarray] = None

    def train(self, X: np.ndarray, threat_types: np.ndarray):
        """
        Fit the classifier on threat samples.

        Args:
            X:            Feature array [n_threat_samples, n_features]
            threat_types: String array of threat type labels (e.g. "credential_abuse")
        """
        y_encoded = self.label_encoder.fit_transform(threat_types)
        self.classes_ = self.label_encoder.classes_
        self.model.fit(X, y_encoded)
        self.fitted = True
        logger.info(
            "ThreatClassifier trained — samples: %d, classes: %s",
            len(X), list(self.classes_),
        )

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Return probability for each threat type.

        Args:
            X: Feature array [n_samples, n_features]

        Returns:
            Dict mapping threat type name → probability array [n_samples]
        """
        if not self.fitted:
            raise ValueError("ThreatClassifier not trained yet")
        proba = self.model.predict_proba(X)
        return {cls: proba[:, i] for i, cls in enumerate(self.classes_)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return most probable threat type label per sample."""
        if not self.fitted:
            raise ValueError("ThreatClassifier not trained yet")
        return self.label_encoder.inverse_transform(self.model.predict(X))

    def save(self, filepath: str):
        """Serialize model and encoder to disk."""
        joblib.dump({'model': self.model, 'encoder': self.label_encoder}, filepath)
        logger.info("ThreatClassifier saved to: %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> 'ThreatClassifier':
        """Load a serialized ThreatClassifier from disk."""
        obj = cls()
        saved = joblib.load(filepath)
        obj.model = saved['model']
        obj.label_encoder = saved['encoder']
        obj.classes_ = obj.label_encoder.classes_
        obj.fitted = True
        logger.info("ThreatClassifier loaded from: %s", filepath)
        return obj


# =============================================================================
# Ensemble
# =============================================================================

class OptikalEnsemble:
    """
    Optikal Ensemble — Combined Threat Scoring.

    Fuses Isolation Forest and LSTM scores via weighted combination and,
    optionally, classifies the threat type via ThreatClassifier.
    """

    def __init__(
        self,
        isolation_forest: OptikalIsolationForest,
        lstm: OptikalLSTM,
        threat_classifier: Optional[ThreatClassifier] = None,
        if_weight: float = 0.4,
        lstm_weight: float = 0.6,
    ):
        """
        Args:
            isolation_forest:   Trained Isolation Forest component
            lstm:               Trained LSTM component
            threat_classifier:  Optional trained ThreatClassifier
            if_weight:          Relative weight for IF scores
            lstm_weight:        Relative weight for LSTM scores
        """
        self.isolation_forest = isolation_forest
        self.lstm = lstm
        self.threat_classifier = threat_classifier

        # Normalise weights
        total = if_weight + lstm_weight
        self.if_weight = if_weight / total
        self.lstm_weight = lstm_weight / total

    def predict_score(
        self,
        X_tabular: np.ndarray,
        X_sequential: np.ndarray,
    ) -> np.ndarray:
        """
        Compute weighted ensemble threat scores.

        Args:
            X_tabular:    Scaled tabular features for IF   [n_samples, n_features]
            X_sequential: Scaled sequences for LSTM        [n_samples, seq_len, n_features]

        Returns:
            Ensemble scores in [0, 1]
        """
        if_scores   = self.isolation_forest.predict_anomaly_score(X_tabular)
        lstm_scores = self.lstm.predict_score(X_sequential)
        return self.if_weight * if_scores + self.lstm_weight * lstm_scores

    def predict(
        self,
        X_tabular: np.ndarray,
        X_sequential: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate threat predictions with detailed per-model scoring.

        When a ThreatClassifier is available, also returns per-threat-type
        probabilities for each flagged sample.

        Args:
            X_tabular:    Scaled tabular features [n_samples, n_features]
            X_sequential: Scaled sequences        [n_samples, seq_len, n_features]
            threshold:    Classification threshold

        Returns:
            Tuple of (binary_predictions [n_samples], detailed_scores dict)
        """
        if_scores       = self.isolation_forest.predict_anomaly_score(X_tabular)
        lstm_scores     = self.lstm.predict_score(X_sequential)
        ensemble_scores = self.if_weight * if_scores + self.lstm_weight * lstm_scores
        predictions     = (ensemble_scores >= threshold).astype(int)

        detailed_scores: Dict = {
            'ensemble':        ensemble_scores,
            'isolation_forest': if_scores,
            'lstm':            lstm_scores,
        }

        if self.threat_classifier is not None and self.threat_classifier.fitted:
            detailed_scores['threat_type_probs'] = (
                self.threat_classifier.predict_proba(X_tabular)
            )
            detailed_scores['threat_type'] = (
                self.threat_classifier.predict(X_tabular)
            )

        return predictions, detailed_scores

    def evaluate(
        self,
        X_tabular: np.ndarray,
        X_sequential: np.ndarray,
        y_true: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Evaluate ensemble performance and log metrics, return raw scores."""
        ensemble_scores = self.predict_score(X_tabular, X_sequential)
        y_pred = (ensemble_scores >= threshold).astype(int)

        logger.info("=== Optikal Ensemble Evaluation ===")
        logger.info("  Accuracy:  %.4f", accuracy_score(y_true, y_pred))
        logger.info("  Precision: %.4f", precision_score(y_true, y_pred, zero_division=0))
        logger.info("  Recall:    %.4f", recall_score(y_true, y_pred, zero_division=0))
        logger.info("  F1 Score:  %.4f", f1_score(y_true, y_pred, zero_division=0))
        logger.info("  ROC-AUC:   %.4f", roc_auc_score(y_true, ensemble_scores))
        logger.info("  Weights: IF=%.2f, LSTM=%.2f", self.if_weight, self.lstm_weight)

        return ensemble_scores


# =============================================================================
# Main training pipeline
# =============================================================================

def train_optikal_model(
    data_path: str = "optikal_training_data.csv",
    output_dir: str = "optikal_models",
):
    """
    Main training pipeline for the Optikal threat detection model.

    Steps:
      1. Load and temporally split data
      2. Feature engineering + scaling
      3. Train Isolation Forest
      4. Train LSTM (if TensorFlow available)
      5. Train ThreatClassifier (multi-class)
      6. Evaluate ensemble
      7. Save all model artefacts and metadata

    Args:
        data_path:  Path to training data CSV
        output_dir: Directory to write trained model files
    """
    logger.info("=" * 70)
    logger.info("OPTIKAL THREAT DETECTION MODEL — TRAINING PIPELINE")
    logger.info("=" * 70)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    logger.info("[1/7] Loading data...")
    train_df, val_df, test_df = prepare_training_data(data_path)

    # -------------------------------------------------------------------------
    # Step 2: Feature engineering + scaling
    # -------------------------------------------------------------------------
    logger.info("[2/7] Engineering features...")
    engineer = OptikalFeatureEngineer()

    train_features = engineer.extract_features(train_df)
    val_features   = engineer.extract_features(val_df)
    test_features  = engineer.extract_features(test_df)

    X_train, feature_names = engineer.fit_transform(train_features)
    X_val,  _              = engineer.transform(val_features)
    X_test, _              = engineer.transform(test_features)

    y_train = train_features['is_threat'].values
    y_val   = val_features['is_threat'].values
    y_test  = test_features['is_threat'].values

    engineer.save_scaler(f"{output_dir}/optikal_scaler.pkl")

    # -------------------------------------------------------------------------
    # Step 3: Isolation Forest
    # -------------------------------------------------------------------------
    logger.info("[3/7] Training Isolation Forest...")
    if_model = OptikalIsolationForest(contamination=0.2)
    if_model.train(X_train)
    if_model.save(f"{output_dir}/optikal_isolation_forest.pkl")
    if_scores_test = if_model.evaluate(X_test, y_test)

    # -------------------------------------------------------------------------
    # Step 4: LSTM  (requires pre-scaled arrays — fixes P0-2 data leakage)
    # -------------------------------------------------------------------------
    lstm_model: Optional[OptikalLSTM] = None

    if KERAS_AVAILABLE:
        logger.info("[4/7] Training LSTM...")

        # Use create_sequences_from_array() to avoid unscaled-data leakage
        train_seq, train_seq_labels = engineer.create_sequences_from_array(
            X_train, y_train, sequence_length=10
        )
        val_seq,   val_seq_labels   = engineer.create_sequences_from_array(
            X_val, y_val, sequence_length=10
        )
        test_seq,  test_seq_labels  = engineer.create_sequences_from_array(
            X_test, y_test, sequence_length=10
        )

        lstm_model = OptikalLSTM(sequence_length=10, n_features=len(feature_names))
        lstm_model.train(
            train_seq, train_seq_labels,
            val_seq,   val_seq_labels,
            epochs=20,
            batch_size=32,
        )
        lstm_model.save(f"{output_dir}/optikal_lstm.h5")
        lstm_scores_test = lstm_model.evaluate(test_seq, test_seq_labels)
    else:
        logger.info("[4/7] Skipping LSTM (TensorFlow not available)")

    # -------------------------------------------------------------------------
    # Step 5: Multi-class threat classifier
    # -------------------------------------------------------------------------
    logger.info("[5/7] Training ThreatClassifier...")
    threat_clf = ThreatClassifier()

    # Train only on threat samples
    train_threat_mask = y_train == 1
    if train_threat_mask.sum() > 0:
        threat_clf.train(
            X_train[train_threat_mask],
            train_features['threat_type'].values[train_threat_mask],
        )
        threat_clf.save(f"{output_dir}/optikal_threat_classifier.pkl")

        # Quick evaluation on test threats
        test_threat_mask = y_test == 1
        if test_threat_mask.sum() > 0:
            y_type_pred = threat_clf.predict(X_test[test_threat_mask])
            y_type_true = test_features['threat_type'].values[test_threat_mask]
            logger.info("ThreatClassifier report:\n%s",
                        classification_report(y_type_true, y_type_pred, zero_division=0))
    else:
        logger.warning("No threat samples in training data — skipping ThreatClassifier")

    # -------------------------------------------------------------------------
    # Step 6: Ensemble evaluation
    # -------------------------------------------------------------------------
    if lstm_model is not None:
        logger.info("[6/7] Evaluating Ensemble...")
        threat_clf_for_ensemble = threat_clf if threat_clf.fitted else None
        ensemble = OptikalEnsemble(if_model, lstm_model, threat_clf_for_ensemble)

        # Align tabular and sequential test sets (sequences are shorter by seq_len-1)
        n_seq = len(test_seq)
        X_test_aligned = X_test[-n_seq:]
        y_test_aligned = y_test[-n_seq:]

        ensemble.evaluate(X_test_aligned, test_seq, y_test_aligned)
    else:
        logger.info("[6/7] Skipping Ensemble evaluation (LSTM not available)")

    # -------------------------------------------------------------------------
    # Step 7: Save metadata
    # -------------------------------------------------------------------------
    logger.info("[7/7] Saving model metadata...")
    metadata = {
        "model_name":        "Optikal",
        "version":           "1.0.0",
        "features":          feature_names,
        "n_features":        len(feature_names),
        "sequence_length":   10,
        "contamination":     0.2,
        "training_samples":  len(train_df),
        "val_samples":       len(val_df),
        "test_samples":      len(test_df),
        "threat_types":      list(train_df['threat_type'].unique()),
        "lstm_available":    lstm_model is not None,
    }

    with open(f"{output_dir}/optikal_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("Metadata saved")
    logger.info("=" * 70)
    logger.info("OPTIKAL TRAINING COMPLETE — models written to: %s/", output_dir)
    logger.info("  optikal_isolation_forest.pkl")
    logger.info("  optikal_threat_classifier.pkl")
    if lstm_model is not None:
        logger.info("  optikal_lstm.h5")
    logger.info("  optikal_scaler.pkl")
    logger.info("  optikal_metadata.json")
    logger.info("=" * 70)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="[%(levelname)s] %(message)s")

    import os
    if not os.path.exists("optikal_training_data.csv"):
        logger.info("Generating synthetic training data...")
        generator = SyntheticDataGenerator(seed=42)
        dataset = generator.generate_training_dataset(
            n_normal=5000,
            n_threats_per_type=500,
        )
        generator.save_dataset(dataset, "optikal_training_data.csv")

    train_optikal_model(
        data_path="optikal_training_data.csv",
        output_dir="optikal_models",
    )
