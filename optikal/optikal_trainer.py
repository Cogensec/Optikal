"""
Optikal Model Trainer

Trains the Optikal threat detection ensemble model:
  1. OptikalIsolationForest  — Unsupervised anomaly detection
  2. OptikalLSTM             — Sequential behavioural analysis (with attention)
  3. OptikalGBM              — Supervised gradient boosting (optional third member)
  4. ThreatClassifier        — Multi-class threat type identification
  5. OptikalMetaLearner      — Learned ensemble fusion (LogisticRegression meta-model)
  6. OptikalEnsemble         — Combined threat scoring with uncertainty estimates

New in this version (P2/P3 enhancements):
  - LSTM Bahdanau attention mechanism
  - Monte Carlo Dropout for LSTM uncertainty quantification
  - OptikalGBM third ensemble member (LightGBM or sklearn GBM fallback)
  - OptikalMetaLearner replaces static weighting when trained
  - MLflow experiment tracking (optional)
  - Optuna-driven hyperparameter search (optional)
"""

from __future__ import annotations

import logging
import warnings
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    logger.warning("TensorFlow not available — LSTM model will be skipped.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.info("LightGBM not available — GBM will use sklearn GradientBoosting.")

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.info("MLflow not available — experiment tracking disabled.")

try:
    import tf2onnx  # noqa: F401
    import onnx     # noqa: F401
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

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
        """Fit the Isolation Forest on scaled training features."""
        logger.info(
            "Training OptikalIsolationForest — samples: %d, features: %d, "
            "contamination: %.2f",
            X_train.shape[0], X_train.shape[1], self.contamination,
        )
        self.model.fit(X_train)
        self.fitted = True
        logger.info("Isolation Forest training complete")

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Return sigmoid-normalised anomaly scores in [0, 1] (higher = more anomalous)."""
        if not self.fitted:
            raise ValueError("Model not trained yet")
        raw = self.model.decision_function(X)
        return 1.0 / (1.0 + np.exp(raw))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary threat predictions (1 = anomaly, 0 = normal)."""
        return (self.predict_anomaly_score(X) >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Evaluate and log model performance metrics; return raw scores."""
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
# LSTM component (with Bahdanau attention + MC Dropout)
# =============================================================================

class OptikalLSTM:
    """
    Optikal LSTM — Sequential Behavioural Analysis Component.

    Uses a Bahdanau-style attention mechanism over the LSTM output to focus
    on the most diagnostic timesteps. Monte Carlo Dropout is used at inference
    time to produce calibrated uncertainty estimates.
    """

    def __init__(self, sequence_length: int = 10, n_features: int = 18):
        if not KERAS_AVAILABLE:
            raise ImportError("TensorFlow/Keras required for LSTM model")
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model: Optional[Any] = None
        self._attention_model: Optional[Any] = None  # exposes attention weights
        self.fitted = False

    def build_model(self) -> Any:
        """
        Construct and compile the attention-based LSTM architecture.

        Architecture:
          Input → LSTM(128, return_seq) → Dropout(0.2)
               → Dense(1, tanh) attention scores → Softmax (time axis)
               → Weighted context vector (reduce_sum over time)
               → Dense(64, relu) → Dropout(0.2) → Dense(32, relu)
               → Dense(1, sigmoid)
        """
        inputs   = layers.Input(shape=(self.sequence_length, self.n_features))

        # Encode the sequence
        lstm_out = layers.LSTM(128, return_sequences=True)(inputs)
        lstm_out = layers.Dropout(0.2)(lstm_out)

        # Bahdanau attention: compute a scalar score per timestep, then softmax
        attn_scores  = layers.Dense(1, activation='tanh')(lstm_out)     # [B, T, 1]
        attn_weights = layers.Softmax(axis=1)(attn_scores)              # [B, T, 1]

        # Context vector: weighted sum over the time dimension
        weighted = layers.Multiply()([lstm_out, attn_weights])          # [B, T, 128]
        context  = layers.Lambda(
            lambda x: tf.keras.backend.sum(x, axis=1)
        )(weighted)                                                      # [B, 128]

        # Classification head
        x = layers.Dense(64, activation='relu')(context)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        # Main model (for training)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')],
        )

        # Attention-exposing model (shares weights; used for interpretability)
        self._attention_model = keras.Model(
            inputs=inputs,
            outputs=[outputs, attn_weights],
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
        """Fit the LSTM on pre-scaled sequential data."""
        logger.info(
            "Training OptikalLSTM (attention) — sequences: %d, seq_len: %d, "
            "features: %d, epochs: %d",
            X_train.shape[0], X_train.shape[1], X_train.shape[2], epochs,
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
        """Return threat scores in [0, 1] for input sequences."""
        if not self.fitted:
            raise ValueError("Model not trained yet")
        return self.model.predict(X, verbose=0).flatten()

    def predict_score_with_uncertainty(
        self,
        X: np.ndarray,
        n_passes: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout inference: run N forward passes with dropout active.

        Returns mean and std of scores across passes, providing a calibrated
        uncertainty estimate. High std → the model is unsure.

        Args:
            X:        Input sequences [n_samples, seq_len, n_features]
            n_passes: Number of stochastic forward passes (default 50)

        Returns:
            Tuple of:
              - mean_scores [n_samples]  — central estimate
              - std_scores  [n_samples]  — uncertainty (higher = less confident)
        """
        if not self.fitted:
            raise ValueError("Model not trained yet")
        # training=True keeps Dropout active during inference
        scores_all = np.stack([
            self.model(X, training=True).numpy().flatten()
            for _ in range(n_passes)
        ])  # [n_passes, n_samples]
        return scores_all.mean(axis=0), scores_all.std(axis=0)

    def predict_with_attention(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return threat scores AND per-timestep attention weights.

        Args:
            X: Input sequences [n_samples, seq_len, n_features]

        Returns:
            Tuple of:
              - scores          [n_samples]
              - attention_weights [n_samples, seq_len, 1]
        """
        if not self.fitted or self._attention_model is None:
            raise ValueError("Model not trained yet")
        scores, attn = self._attention_model.predict(X, verbose=0)
        return scores.flatten(), attn

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_score(X) >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Evaluate and log model performance; return raw scores."""
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

    Trained on threat samples only; tells operators *what kind* of threat was
    detected, enabling differentiated responses.
    """

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.label_encoder = LabelEncoder()
        self.fitted = False
        self.classes_: Optional[np.ndarray] = None

    def train(self, X: np.ndarray, threat_types: np.ndarray):
        """Fit on threat samples only."""
        y_enc = self.label_encoder.fit_transform(threat_types)
        self.classes_ = self.label_encoder.classes_
        self.model.fit(X, y_enc)
        self.fitted = True
        logger.info(
            "ThreatClassifier trained — samples: %d, classes: %s",
            len(X), list(self.classes_),
        )

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return probability for each threat type as a dict."""
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
    def load(cls, filepath: str) -> "ThreatClassifier":
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
# GBM third ensemble member (P3-5)
# =============================================================================

class OptikalGBM:
    """
    Optikal GBM — Supervised Gradient Boosting Component.

    Unlike the Isolation Forest (unsupervised), the GBM fully leverages the
    labeled ``is_threat`` column and learns non-linear feature interactions.
    Uses LightGBM when available; falls back to sklearn GradientBoostingClassifier.
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.fitted = False
        self._backend: str = "lightgbm" if LIGHTGBM_AVAILABLE else "sklearn"

        if LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=0.05,
                num_leaves=31,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=0.05,
                max_depth=5,
                random_state=random_state,
            )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """
        Fit the GBM on labeled training features.

        Args:
            X_train: Scaled training features [n_samples, n_features]
            y_train: Binary training labels
            X_val:   Optional validation set (used for early stopping with LightGBM)
            y_val:   Validation labels
        """
        logger.info(
            "Training OptikalGBM (%s) — samples: %d, features: %d",
            self._backend, X_train.shape[0], X_train.shape[1],
        )

        if LIGHTGBM_AVAILABLE and X_val is not None:
            callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
        else:
            self.model.fit(X_train, y_train)

        self.fitted = True
        logger.info("GBM training complete (backend: %s)", self._backend)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Return threat probability scores in [0, 1]."""
        if not self.fitted:
            raise ValueError("GBM not trained yet")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_score(X) >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Evaluate and log performance; return raw scores."""
        scores = self.predict_score(X)
        y_pred = (scores >= threshold).astype(int)
        logger.info("=== GBM Evaluation (%s) ===", self._backend)
        logger.info("  Accuracy:  %.4f", accuracy_score(y_true, y_pred))
        logger.info("  Precision: %.4f", precision_score(y_true, y_pred, zero_division=0))
        logger.info("  Recall:    %.4f", recall_score(y_true, y_pred, zero_division=0))
        logger.info("  F1 Score:  %.4f", f1_score(y_true, y_pred, zero_division=0))
        logger.info("  ROC-AUC:   %.4f", roc_auc_score(y_true, scores))
        return scores

    def save(self, filepath: str):
        """Serialize model to disk."""
        joblib.dump({'model': self.model, 'backend': self._backend}, filepath)
        logger.info("GBM model saved to: %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "OptikalGBM":
        """Load a serialized GBM from disk."""
        obj = cls()
        saved = joblib.load(filepath)
        obj.model = saved['model']
        obj._backend = saved['backend']
        obj.fitted = True
        logger.info("GBM loaded from: %s (backend: %s)", filepath, obj._backend)
        return obj


# =============================================================================
# Meta-learner — learned ensemble fusion (P3-4)
# =============================================================================

class OptikalMetaLearner:
    """
    Learned ensemble fusion via LogisticRegression meta-model.

    Trained on the stacked out-of-validation predictions of the base models
    (IF + LSTM, or IF + LSTM + GBM) to discover optimal weighting. At
    inference time, replaces the static IF/LSTM weight combination.

    When not fitted, OptikalEnsemble falls back to static weights.
    """

    def __init__(self):
        self.model = LogisticRegression(C=1.0, random_state=42, max_iter=500)
        self.fitted = False
        self.n_base_models: int = 2

    def fit(
        self,
        base_scores: List[np.ndarray],
        y: np.ndarray,
    ):
        """
        Train the meta-learner on stacked base model predictions.

        Args:
            base_scores: List of score arrays from each base model, each [n_samples]
            y:           True binary labels [n_samples]
        """
        X_meta = np.column_stack(base_scores)
        self.model.fit(X_meta, y)
        self.n_base_models = len(base_scores)
        self.fitted = True

        coefs = self.model.coef_[0]
        names = [f"model_{i}" for i in range(len(coefs))]
        logger.info(
            "MetaLearner fitted — learned weights: %s",
            dict(zip(names, [f"{c:.4f}" for c in coefs])),
        )

    def predict_score(self, base_scores: List[np.ndarray]) -> np.ndarray:
        """
        Return meta-model threat scores from stacked base model predictions.

        Args:
            base_scores: Score arrays from each base model, each [n_samples]

        Returns:
            Meta-model threat probability [n_samples] in [0, 1]
        """
        if not self.fitted:
            raise ValueError("MetaLearner not fitted yet")
        X_meta = np.column_stack(base_scores)
        return self.model.predict_proba(X_meta)[:, 1]

    def save(self, filepath: str):
        """Serialize to disk."""
        joblib.dump(self, filepath)
        logger.info("MetaLearner saved to: %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "OptikalMetaLearner":
        """Load from disk."""
        obj = joblib.load(filepath)
        logger.info("MetaLearner loaded from: %s", filepath)
        return obj


# =============================================================================
# Ensemble (supports IF + LSTM + optional GBM, static or meta-learned weights)
# =============================================================================

class OptikalEnsemble:
    """
    Optikal Ensemble — Combined Threat Scoring.

    Fuses Isolation Forest, LSTM, and (optionally) GBM scores. When a fitted
    ``OptikalMetaLearner`` is provided, it replaces static weight blending.
    LSTM uncertainty (MC Dropout) is surfaced in the ``predict()`` output.
    """

    def __init__(
        self,
        isolation_forest: OptikalIsolationForest,
        lstm: Optional[OptikalLSTM] = None,
        gbm: Optional[OptikalGBM] = None,
        threat_classifier: Optional[ThreatClassifier] = None,
        meta_learner: Optional[OptikalMetaLearner] = None,
        if_weight: float = 0.4,
        lstm_weight: float = 0.6,
        gbm_weight: float = 0.0,
    ):
        """
        Args:
            isolation_forest:  Trained IF component
            lstm:              Optional trained LSTM component
            gbm:               Optional trained GBM component
            threat_classifier: Optional ThreatClassifier
            meta_learner:      Optional trained MetaLearner (overrides static weights)
            if_weight:         Static IF weight (normalised internally)
            lstm_weight:       Static LSTM weight
            gbm_weight:        Static GBM weight (used when GBM provided and no meta-learner)
        """
        self.isolation_forest  = isolation_forest
        self.lstm              = lstm
        self.gbm               = gbm
        self.threat_classifier = threat_classifier
        self.meta_learner      = meta_learner

        # Compute normalised static weights
        active_weights = [if_weight]
        if lstm is not None:
            active_weights.append(lstm_weight)
        if gbm is not None:
            active_weights.append(gbm_weight if gbm_weight > 0 else 1.0)
        total = sum(active_weights) or 1.0
        self.if_weight   = if_weight   / total
        self.lstm_weight = lstm_weight / total if lstm is not None else 0.0
        self.gbm_weight  = active_weights[-1] / total if gbm is not None else 0.0

    def _get_base_scores(
        self,
        X_tabular: np.ndarray,
        X_sequential: Optional[np.ndarray],
    ) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
        """Collect scores from all available base models."""
        base_scores: List[np.ndarray] = []
        named: Dict[str, np.ndarray] = {}

        if_s = self.isolation_forest.predict_anomaly_score(X_tabular)
        base_scores.append(if_s)
        named["isolation_forest"] = if_s

        if self.lstm is not None and X_sequential is not None:
            lstm_s = self.lstm.predict_score(X_sequential)
            base_scores.append(lstm_s)
            named["lstm"] = lstm_s

        if self.gbm is not None:
            gbm_s = self.gbm.predict_score(X_tabular)
            base_scores.append(gbm_s)
            named["gbm"] = gbm_s

        return base_scores, named

    def predict_score(
        self,
        X_tabular: np.ndarray,
        X_sequential: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute ensemble threat scores.

        Uses the meta-learner when fitted; otherwise applies static weights.
        """
        base_scores, _ = self._get_base_scores(X_tabular, X_sequential)

        if self.meta_learner and self.meta_learner.fitted:
            return self.meta_learner.predict_score(base_scores)

        # Static weighted combination
        score = self.if_weight * base_scores[0]
        idx = 1
        if self.lstm is not None and X_sequential is not None:
            score = score + self.lstm_weight * base_scores[idx]
            idx += 1
        if self.gbm is not None:
            score = score + self.gbm_weight * base_scores[idx]

        return score

    def predict(
        self,
        X_tabular: np.ndarray,
        X_sequential: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        mc_passes: int = 0,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate threat predictions with detailed per-model scoring.

        Args:
            X_tabular:    Scaled tabular features [n_samples, n_features]
            X_sequential: Scaled sequences [n_samples, seq_len, n_features] or None
            threshold:    Classification threshold
            mc_passes:    Number of MC Dropout passes for LSTM uncertainty
                          (0 = deterministic; recommended: 50)

        Returns:
            Tuple of (binary_predictions [n_samples], detailed_scores dict)
        """
        base_scores, named = self._get_base_scores(X_tabular, X_sequential)

        if self.meta_learner and self.meta_learner.fitted:
            ensemble_scores = self.meta_learner.predict_score(base_scores)
        else:
            ensemble_scores = self.predict_score(X_tabular, X_sequential)

        predictions = (ensemble_scores >= threshold).astype(int)

        detailed: Dict = {"ensemble": ensemble_scores, **named}

        # MC Dropout uncertainty
        if mc_passes > 0 and self.lstm is not None and X_sequential is not None:
            mean_lstm, std_lstm = self.lstm.predict_score_with_uncertainty(
                X_sequential, n_passes=mc_passes
            )
            detailed["lstm_mean"]        = mean_lstm
            detailed["lstm_uncertainty"] = std_lstm
            detailed["confidence"]       = 1.0 - std_lstm

        # Multi-class threat type
        if self.threat_classifier is not None and self.threat_classifier.fitted:
            detailed["threat_type_probs"] = (
                self.threat_classifier.predict_proba(X_tabular)
            )
            detailed["threat_type"] = self.threat_classifier.predict(X_tabular)

        return predictions, detailed

    def evaluate(
        self,
        X_tabular: np.ndarray,
        y_true: np.ndarray,
        X_sequential: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Evaluate ensemble performance and log metrics; return raw scores."""
        ensemble_scores = self.predict_score(X_tabular, X_sequential)
        y_pred = (ensemble_scores >= threshold).astype(int)

        logger.info("=== Optikal Ensemble Evaluation ===")
        logger.info("  Accuracy:  %.4f", accuracy_score(y_true, y_pred))
        logger.info("  Precision: %.4f", precision_score(y_true, y_pred, zero_division=0))
        logger.info("  Recall:    %.4f", recall_score(y_true, y_pred, zero_division=0))
        logger.info("  F1 Score:  %.4f", f1_score(y_true, y_pred, zero_division=0))
        logger.info("  ROC-AUC:   %.4f", roc_auc_score(y_true, ensemble_scores))

        fusion = "meta-learner" if (self.meta_learner and self.meta_learner.fitted) else "static"
        logger.info("  Fusion: %s  IF=%.2f  LSTM=%.2f  GBM=%.2f",
                    fusion, self.if_weight, self.lstm_weight, self.gbm_weight)

        return ensemble_scores


# =============================================================================
# Main training pipeline
# =============================================================================

def train_optikal_model(
    data_path: str = "optikal_training_data.csv",
    output_dir: str = "optikal_models",
    tune: bool = False,
    use_mlflow: bool = False,
    n_hp_trials: int = 50,
):
    """
    Main training pipeline for the Optikal threat detection model.

    Steps:
      1.  Load and temporally split data
      2.  Feature engineering + scaling
      3.  (optional) Optuna hyperparameter search for Isolation Forest
      4.  Train Isolation Forest
      5.  Train LSTM (if TensorFlow available)
      6.  Train GBM
      7.  Train ThreatClassifier (multi-class)
      8.  Fit MetaLearner on validation predictions
      9.  Evaluate ensemble
      10. Save all model artefacts + metadata
      (optional) MLflow experiment tracking throughout

    Args:
        data_path:    Path to training data CSV
        output_dir:   Directory to write trained model files
        tune:         When True, run Optuna hyperparameter search before training
        use_mlflow:   When True, log params/metrics/models to MLflow
        n_hp_trials:  Number of Optuna trials (only used when tune=True)
    """
    logger.info("=" * 70)
    logger.info("OPTIKAL THREAT DETECTION MODEL — TRAINING PIPELINE")
    logger.info("=" * 70)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Start MLflow run if requested
    _mlflow_run = None
    if use_mlflow and MLFLOW_AVAILABLE:
        _mlflow_run = mlflow.start_run(run_name="optikal_training")
        logger.info("MLflow run started: %s", _mlflow_run.info.run_id)

    try:
        _run_training_pipeline(
            data_path, output_dir, tune, use_mlflow, n_hp_trials
        )
    finally:
        if _mlflow_run and MLFLOW_AVAILABLE:
            mlflow.end_run()
            logger.info("MLflow run ended")


def _run_training_pipeline(
    data_path: str,
    output_dir: str,
    tune: bool,
    use_mlflow: bool,
    n_hp_trials: int,
):
    """Inner implementation so MLflow run wrapping stays clean."""

    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    logger.info("[1/10] Loading data...")
    train_df, val_df, test_df = prepare_training_data(data_path)

    # -------------------------------------------------------------------------
    # Step 2: Feature engineering + scaling
    # -------------------------------------------------------------------------
    logger.info("[2/10] Engineering features...")
    engineer = OptikalFeatureEngineer()

    train_features = engineer.extract_features(train_df)
    val_features   = engineer.extract_features(val_df)
    test_features  = engineer.extract_features(test_df)

    X_train, feature_names = engineer.fit_transform(train_features)
    X_val,   _             = engineer.transform(val_features)
    X_test,  _             = engineer.transform(test_features)

    y_train = train_features['is_threat'].values
    y_val   = val_features['is_threat'].values
    y_test  = test_features['is_threat'].values

    engineer.save_scaler(f"{output_dir}/optikal_scaler.pkl")

    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.log_param("n_features",        len(feature_names))
        mlflow.log_param("training_samples",  len(train_df))
        mlflow.log_param("val_samples",       len(val_df))
        mlflow.log_param("test_samples",      len(test_df))

    # -------------------------------------------------------------------------
    # Step 3: Hyperparameter search (optional)
    # -------------------------------------------------------------------------
    contamination = 0.15
    n_estimators  = 100
    if_threshold  = 0.5

    if tune:
        logger.info("[3/10] Running Optuna hyperparameter search (%d trials)...", n_hp_trials)
        try:
            from hyperparameter_search import run_hyperparameter_search
            hp_results = run_hyperparameter_search(
                X_train, y_train, X_val, y_val, n_trials=n_hp_trials
            )
            best = hp_results["best_params"]
            contamination = best["contamination"]
            n_estimators  = best["n_estimators"]
            if_threshold  = best["threshold"]
            logger.info(
                "Tuned params — contamination: %.4f, n_estimators: %d, threshold: %.4f",
                contamination, n_estimators, if_threshold,
            )
            if use_mlflow and MLFLOW_AVAILABLE:
                mlflow.log_params(best)
                mlflow.log_metric("hp_best_f1", hp_results["best_value"])
        except ImportError:
            logger.warning("hyperparameter_search module not found — skipping")
    else:
        logger.info("[3/10] Skipping hyperparameter search (tune=False)")

    # -------------------------------------------------------------------------
    # Step 4: Isolation Forest
    # -------------------------------------------------------------------------
    logger.info("[4/10] Training Isolation Forest...")
    if_model = OptikalIsolationForest(contamination=contamination)
    if_model.model.n_estimators = n_estimators
    if_model.train(X_train)
    if_model.save(f"{output_dir}/optikal_isolation_forest.pkl")
    if_scores_test  = if_model.evaluate(X_test, y_test, threshold=if_threshold)
    if_scores_val   = if_model.predict_anomaly_score(X_val)
    if_scores_train = if_model.predict_anomaly_score(X_train)

    if use_mlflow and MLFLOW_AVAILABLE:
        y_pred_if = (if_scores_test >= if_threshold).astype(int)
        mlflow.log_metrics({
            "if_f1":      f1_score(y_test, y_pred_if, zero_division=0),
            "if_roc_auc": roc_auc_score(y_test, if_scores_test),
        })
        mlflow.log_param("contamination", contamination)

    # -------------------------------------------------------------------------
    # Step 5: LSTM
    # -------------------------------------------------------------------------
    lstm_model: Optional[OptikalLSTM] = None
    lstm_scores_val  = np.zeros(len(X_val))
    lstm_scores_test = np.zeros(len(X_test))

    if KERAS_AVAILABLE:
        logger.info("[5/10] Training LSTM (attention)...")
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
            epochs=20, batch_size=32,
        )
        lstm_model.save(f"{output_dir}/optikal_lstm.h5")
        lstm_scores_test_raw = lstm_model.evaluate(test_seq, test_seq_labels)

        # Align for meta-learner: last n_seq samples
        lstm_scores_val  = lstm_model.predict_score(val_seq)
        lstm_scores_test = lstm_scores_test_raw

        if use_mlflow and MLFLOW_AVAILABLE:
            y_pred_lstm = (lstm_scores_test >= 0.5).astype(int)
            mlflow.log_metrics({
                "lstm_f1":      f1_score(test_seq_labels, y_pred_lstm, zero_division=0),
                "lstm_roc_auc": roc_auc_score(test_seq_labels, lstm_scores_test),
            })
    else:
        logger.info("[5/10] Skipping LSTM (TensorFlow not available)")

    # -------------------------------------------------------------------------
    # Step 6: GBM
    # -------------------------------------------------------------------------
    logger.info("[6/10] Training GBM...")
    gbm_model = OptikalGBM()
    gbm_model.train(X_train, y_train, X_val, y_val)
    gbm_model.save(f"{output_dir}/optikal_gbm.pkl")
    gbm_scores_test = gbm_model.evaluate(X_test, y_test)
    gbm_scores_val  = gbm_model.predict_score(X_val)

    if use_mlflow and MLFLOW_AVAILABLE:
        y_pred_gbm = (gbm_scores_test >= 0.5).astype(int)
        mlflow.log_metrics({
            "gbm_f1":      f1_score(y_test, y_pred_gbm, zero_division=0),
            "gbm_roc_auc": roc_auc_score(y_test, gbm_scores_test),
        })

    # -------------------------------------------------------------------------
    # Step 7: Multi-class threat classifier
    # -------------------------------------------------------------------------
    logger.info("[7/10] Training ThreatClassifier...")
    threat_clf = ThreatClassifier()
    train_threat_mask = y_train == 1

    if train_threat_mask.sum() > 0:
        threat_clf.train(
            X_train[train_threat_mask],
            train_features['threat_type'].values[train_threat_mask],
        )
        threat_clf.save(f"{output_dir}/optikal_threat_classifier.pkl")

        test_threat_mask = y_test == 1
        if test_threat_mask.sum() > 0:
            y_type_pred = threat_clf.predict(X_test[test_threat_mask])
            y_type_true = test_features['threat_type'].values[test_threat_mask]
            logger.info("ThreatClassifier report:\n%s",
                        classification_report(y_type_true, y_type_pred, zero_division=0))
    else:
        logger.warning("No threat samples in training — skipping ThreatClassifier")

    # -------------------------------------------------------------------------
    # Step 8: MetaLearner — train on validation predictions
    # -------------------------------------------------------------------------
    logger.info("[8/10] Training MetaLearner...")
    meta_learner = OptikalMetaLearner()

    # Assemble validation base scores (align to shortest if LSTM was used)
    if lstm_model is not None:
        n_val_seq = len(lstm_scores_val)
        val_if_aligned  = if_scores_val[-n_val_seq:]
        val_y_aligned   = y_val[-n_val_seq:]
        val_gbm_aligned = gbm_scores_val[-n_val_seq:]
        base_val_scores = [val_if_aligned, lstm_scores_val, val_gbm_aligned]
    else:
        base_val_scores = [if_scores_val, gbm_scores_val]
        val_y_aligned   = y_val

    meta_learner.fit(base_val_scores, val_y_aligned)
    meta_learner.save(f"{output_dir}/optikal_meta_learner.pkl")

    # -------------------------------------------------------------------------
    # Step 9: Ensemble evaluation
    # -------------------------------------------------------------------------
    logger.info("[9/10] Evaluating Ensemble...")
    ensemble = OptikalEnsemble(
        isolation_forest=if_model,
        lstm=lstm_model,
        gbm=gbm_model,
        threat_classifier=threat_clf if threat_clf.fitted else None,
        meta_learner=meta_learner,
    )

    if lstm_model is not None:
        n_seq = len(test_seq)
        X_test_aligned = X_test[-n_seq:]
        y_test_aligned = test_seq_labels
        ens_scores = ensemble.evaluate(X_test_aligned, y_test_aligned, test_seq)
    else:
        ens_scores = ensemble.evaluate(X_test, y_test)

    if use_mlflow and MLFLOW_AVAILABLE:
        y_pred_ens = (ens_scores >= 0.5).astype(int)
        y_aligned  = y_test_aligned if lstm_model is not None else y_test
        mlflow.log_metrics({
            "ensemble_f1":      f1_score(y_aligned, y_pred_ens, zero_division=0),
            "ensemble_roc_auc": roc_auc_score(y_aligned, ens_scores),
        })
        mlflow.sklearn.log_model(if_model.model, "isolation_forest")
        mlflow.sklearn.log_model(gbm_model.model, "gbm")

    # -------------------------------------------------------------------------
    # Step 10: Save metadata
    # -------------------------------------------------------------------------
    logger.info("[10/10] Saving model metadata...")
    metadata = {
        "model_name":       "Optikal",
        "version":          "2.0.0",
        "features":         feature_names,
        "n_features":       len(feature_names),
        "sequence_length":  10,
        "contamination":    contamination,
        "training_samples": len(train_df),
        "val_samples":      len(val_df),
        "test_samples":     len(test_df),
        "threat_types":     list(train_df['threat_type'].unique()),
        "lstm_available":   lstm_model is not None,
        "gbm_backend":      gbm_model._backend,
        "meta_learner":     True,
        "attention_lstm":   True,
    }
    with open(f"{output_dir}/optikal_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 70)
    logger.info("OPTIKAL TRAINING COMPLETE — models in: %s/", output_dir)
    logger.info("  optikal_isolation_forest.pkl")
    logger.info("  optikal_gbm.pkl")
    logger.info("  optikal_threat_classifier.pkl")
    logger.info("  optikal_meta_learner.pkl")
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
