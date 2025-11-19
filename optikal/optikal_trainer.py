"""
Optikal Model Trainer

Trains the Optikal threat detection ensemble model:
1. Optikal Isolation Forest - Anomaly detection
2. Optikal LSTM - Sequential behavioral analysis  
3. Optikal Ensemble - Combined threat scoring

Exports models to ONNX for Triton deployment.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f_1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("WARNING: TensorFlow not available. LSTM model will be skipped.")

# ONNX export
try:
    import tf2onnx
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("WARNING: ONNX tools not available. Model export will be skipped.")

from data_generator import SyntheticDataGenerator
from feature_engineering import OptiMalFeatureEngineer, prepare_training_data


class OptiMalIsolationForest:
    """
    Optikal Isolation Forest - Anomaly Detection Component
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize Isolation Forest
        
        Args:
            contamination: Expected proportion of anomalies
            random_state: Random seed
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples=256,
            n_jobs=-1
        )
        self.contamination = contamination
        self.fitted = False
    
    def train(self, X_train: np.ndarray):
        """Train Isolation Forest"""
        print("\n=== Training Optikal Isolation Forest ===")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Contamination: {self.contamination}")
        
        self.model.fit(X_train)
        self.fitted = True
        
        print("✓ Isolation Forest training complete")
    
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores (higher = more anomalous)
        
        Returns:
            Anomaly scores in range [0, 1]
        """
        if not self.fitted:
            raise ValueError("Model not trained yet")
        
        # Get decision function (negative for anomalies)
        scores = self.model.decision_function(X)
        
        # Normalize to [0, 1] range (higher = more anomalous)
        scores_normalized = 1 / (1 + np.exp(scores))  # Sigmoid transformation
        
        return scores_normalized
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels
        
        Args:
            X: Features
            threshold: Threshold for classification
            
        Returns:
            Binary predictions (1 = anomaly/threat)
        """
        scores = self.predict_anomaly_score(X)
        return (scores >= threshold).astype(int)
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
        """Evaluate model performance"""
        scores = self.predict_anomaly_score(X)
        y_pred = (scores >= threshold).astype(int)
        
        print("\n=== Isolation Forest Evaluation ===")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_true, scores):.4f}")
        
        return scores
    
    def save(self, filepath: str):
        """Save model to file"""
        joblib.dump(self.model, filepath)
        print(f"✓ Isolation Forest saved to: {filepath}")


class OptiMalLSTM:
    """
    Optikal LSTM - Sequential Behavioral Analysis Component
    """
    
    def __init__(self, sequence_length: int = 10, n_features: int = 18):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Number of timesteps in sequence
            n_features: Number of features per timestep
        """
        if not KERAS_AVAILABLE:
            raise ImportError("TensorFlow/Keras required for LSTM model")
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.fitted = False
    
    def build_model(self) -> keras.Model:
        """Build LSTM architecture"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.sequence_length, self.n_features)),
            
            # LSTM layers
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32
    ):
        """
        Train LSTM model
        
        Args:
            X_train: Training sequences [n_samples, sequence_length, n_features]
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
        """
        print("\n=== Training Optikal LSTM ===")
        print(f"Training sequences: {X_train.shape[0]}")
        print(f"Sequence length: {X_train.shape[1]}")
        print(f"Features: {X_train.shape[2]}")
        print(f"Epochs: {epochs}")
        
        # Build model
        self.model = self.build_model()
        
        print(f"\nModel architecture:")
        self.model.summary()
       
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        self.fitted = True
        print("\n✓ LSTM training complete")
        
        return history
    
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Predict threat scores
        
        Args:
            X: Input sequences
            
        Returns:
            Threat scores in range [0, 1]
        """
        if not self.fitted:
            raise ValueError("Model not trained yet")
        
        scores = self.model.predict(X, verbose=0)
        return scores.flatten()
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels"""
        scores = self.predict_score(X)
        return (scores >= threshold).astype(int)
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
        """Evaluate model performance"""
        scores = self.predict_score(X)
        y_pred = (scores >= threshold).astype(int)
        
        print("\n=== LSTM Evaluation ===")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_true, scores):.4f}")
        
        return scores
    
    def save(self, filepath: str):
        """Save model to H5 format"""
        self.model.save(filepath)
        print(f"✓ LSTM model saved to: {filepath}")


class OptiMalEnsemble:
    """
    Optikal Ensemble - Combined Threat Scoring
    """
    
    def __init__(
        self,
        isolation_forest: OptiMalIsolationForest,
        lstm: OptiMalLSTM,
        if_weight: float = 0.4,
        lstm_weight: float = 0.6
    ):
        """
        Initialize ensemble
        
        Args:
            isolation_forest: Trained Isolation Forest model
            lstm: Trained LSTM model
            if_weight: Weight for Isolation Forest score
            lstm_weight: Weight for LSTM score
        """
        self.isolation_forest = isolation_forest
        self.lstm = lstm
        self.if_weight = if_weight
        self.lstm_weight = lstm_weight
        
        # Normalize weights
        total_weight = if_weight + lstm_weight
        self.if_weight /= total_weight
        self.lstm_weight /= total_weight
    
    def predict_score(
        self,
        X_tabular: np.ndarray,
        X_sequential: np.ndarray
    ) -> np.ndarray:
        """
        Predict ensemble threat scores
        
        Args:
            X_tabular: Tabular features for Isolation Forest
            X_sequential: Sequential features for LSTM
            
        Returns:
            Combined threat scores [0, 1]
        """
        # Get individual predictions
        if_scores = self.isolation_forest.predict_anomaly_score(X_tabular)
        lstm_scores = self.lstm.predict_score(X_sequential)
        
        # Weighted combination
        ensemble_scores = (
            self.if_weight * if_scores +
            self.lstm_weight * lstm_scores
        )
        
        return ensemble_scores
    
    def predict(
        self,
        X_tabular: np.ndarray,
        X_sequential: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Predict with detailed scoring
        
        Returns:
            Tuple of (predictions, detailed_scores)
        """
        ensemble_scores = self.predict_score(X_tabular, X_sequential)
        predictions = (ensemble_scores >= threshold).astype(int)
        
        detailed_scores = {
            'ensemble': ensemble_scores,
            'isolation_forest': self.isolation_forest.predict_anomaly_score(X_tabular),
            'lstm': self.lstm.predict_score(X_sequential)
        }
        
        return predictions, detailed_scores
    
    def evaluate(
        self,
        X_tabular: np.ndarray,
        X_sequential: np.ndarray,
        y_true: np.ndarray,
        threshold: float = 0.5
    ):
        """Evaluate ensemble performance"""
        ensemble_scores = self.predict_score(X_tabular, X_sequential)
        y_pred = (ensemble_scores >= threshold).astype(int)
        
        print("\n=== Optikal Ensemble Evaluation ===")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall: {recall_score(y_true, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_true, ensemble_scores):.4f}")
        
        print(f"\nWeights: IF={self.if_weight:.2f}, LSTM={self.lstm_weight:.2f}")
        
        return ensemble_scores


def train_optikal_model(
    data_path: str = "optikal_training_data.csv",
    output_dir: str = "optikal_models"
):
    """
    Main training pipeline for Optikal model
    
    Args:
        data_path: Path to training data CSV
        output_dir: Directory to save trained models
    """
    print("=" * 70)
    print("OPTIKAL THREAT DETECTION MODEL - TRAINING PIPELINE")
    print("=" * 70)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and prepare data
    print("\n[1/6] Loading data...")
    train_df, val_df, test_df = prepare_training_data(data_path)
    
    # Step 2: Feature engineering
    print("\n[2/6] Engineering features...")
    engineer = OptiMalFeatureEngineer()
    
    train_features = engineer.extract_features(train_df)
    val_features = engineer.extract_features(val_df)
    test_features = engineer.extract_features(test_df)
    
    X_train, feature_names = engineer.fit_transform(train_features)
    X_val, _ = engineer.transform(val_features)
    X_test, _ = engineer.transform(test_features)
    
    y_train = train_features['is_threat'].values
    y_val = val_features['is_threat'].values
    y_test = test_features['is_threat'].values
    
    engineer.save_scaler(f"{output_dir}/optikal_scaler.pkl")
    
    # Step 3: Train Isolation Forest
    print("\n[3/6] Training Isolation Forest...")
    if_model = OptiIsolationForest(contamination=0.2)
    if_model.train(X_train)
    if_model.save(f"{output_dir}/optikal_isolation_forest.pkl")
    
    # Evaluate
    if_scores_test = if_model.evaluate(X_test, y_test)
    
    # Step 4: Train LSTM (if available)
    lstm_model = None
    if KERAS_AVAILABLE:
        print("\n[4/6] Training LSTM...")
        
        # Create sequences
        train_seq, train_seq_labels = engineer.create_sequences(
            train_features, sequence_length=10
        )
        val_seq, val_seq_labels = engineer.create_sequences(
            val_features, sequence_length=10
        )
        test_seq, test_seq_labels = engineer.create_sequences(
            test_features, sequence_length=10
        )
        
        lstm_model = OptiMalLSTM(sequence_length=10, n_features=len(feature_names))
        lstm_model.train(
            train_seq, train_seq_labels,
            val_seq, val_seq_labels,
            epochs=20,
            batch_size=32
        )
        lstm_model.save(f"{output_dir}/optikal_lstm.h5")
        
        # Evaluate
        lstm_scores_test = lstm_model.evaluate(test_seq, test_seq_labels)
    else:
        print("\n[4/6] Skipping LSTM (TensorFlow not available)")
    
    # Step 5: Ensemble evaluation
    if lstm_model:
        print("\n[5/6] Evaluating Ensemble...")
        ensemble = OptiMalEnsemble(if_model, lstm_model)
        
        # Note: Using only samples that have sequences
        n_seq = len(test_seq)
        X_test_seq = X_test[-n_seq:]
        y_test_seq = y_test[-n_seq:]
        
        ensemble_scores = ensemble.evaluate(
            X_test_seq, test_seq, y_test_seq
        )
    else:
        print("\n[5/6] Skipping Ensemble (LSTM not available)")
    
    # Step 6: Save metadata
    print("\n[6/6] Saving model metadata...")
    metadata = {
        "model_name": "Optikal",
        "version": "1.0.0",
        "features": feature_names,
        "n_features": len(feature_names),
        "sequence_length": 10,
        "contamination": 0.2,
        "training_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "threat_types": list(train_df['threat_type'].unique())
    }
    
    with open(f"{output_dir}/optikal_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved")
    
    print("\n" + "=" * 70)
    print("✓ OPTIKAL TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModels saved to: {output_dir}/")
    print("  - optikal_isolation_forest.pkl")
    if KERAS_AVAILABLE:
        print("  - optikal_lstm.h5")
    print("  - optikal_scaler.pkl")
    print("  - optikal_metadata.json")


if __name__ == "__main__":
    # Generate training data if it doesn't exist
    import os
    if not os.path.exists("optikal_training_data.csv"):
        print("Generating synthetic training data...\n")
        generator = SyntheticDataGenerator(seed=42)
        dataset = generator.generate_training_dataset(
            n_normal=5000,
            n_threats_per_type=500
        )
        generator.save_dataset(dataset, "optikal_training_data.csv")
        print()
    
    # Train Optikal model
    train_optikal_model(
        data_path="optikal_training_data.csv",
        output_dir="optikal_models"
    )
