"""
Feature Engineering for Optikal

Extracts security-relevant features from agent behavioral data
for threat detection model training.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib


class OptiMalFeatureEngineer:
    """
    Feature engineering pipeline for Optikal threat detection
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        self.scaler = StandardScaler()
        self.fitted = False
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from raw agent behavioral data
        
        Args:
            df: Raw behavioral data
            
        Returns:
            DataFrame with engineered features
        """
        features_df = df.copy()
        
        # === Rate-based Features ===
        # Normalized per-minute rates
        features_df['api_calls_per_min'] = features_df['api_calls_count']
        features_df['errors_per_min'] = features_df['error_count']
        features_df['file_ops_per_min'] = features_df['file_operations_count']
        features_df['network_calls_per_min'] = features_df['network_calls_count']
        
        # === Resource Features ===
        features_df['cpu_usage'] = features_df['cpu_usage_percent'] / 100.0
        features_df['memory_usage_gb'] = features_df['memory_usage_mb'] / 1024.0
        
        # Resource intensity ratio
        features_df['resource_intensity'] = (
            features_df['cpu_usage'] * features_df['memory_usage_gb']
        )
        
        # === Behavioral Features ===
        features_df['error_rate'] = features_df['error_count'] / (features_df['api_calls_count'] + 1)
        features_df['query_diversity_score'] = features_df['query_diversity']
        features_df['success_rate_score'] = features_df['success_rate']
        
        # === Temporal Features ===
        features_df['is_night'] = ((features_df['hour_of_day'] >= 0) & 
                                   (features_df['hour_of_day'] <= 6)).astype(int)
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for hour
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour_of_day'] / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour_of_day'] / 24)
        
        # === Response Time Features ===
        features_df['response_time_sec'] = features_df['response_time_ms'] / 1000.0
        features_df['response_time_log'] = np.log1p(features_df['response_time_ms'])
        
        # === Composite Anomaly Indicators ===
        # High activity score
        features_df['activity_score'] = (
            features_df['api_calls_count'] / 100.0 +
            features_df['network_calls_count'] / 50.0 +
            features_df['file_operations_count'] / 10.0
        ) / 3.0
        
        # Suspicious behavior score (hand-crafted)
        features_df['suspicion_score'] = (
            (1 - features_df['query_diversity']) * 0.3 +  # Low diversity is suspicious
            (features_df['error_rate']) * 0.3 +            # High errors suspicious
            (features_df['is_night']) * 0.2 +              # Night activity suspicious
            (1 - features_df['success_rate']) * 0.2        # Low success suspicious
        )
        
        return features_df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns for model training"""
        return [
            # Rate features
            'api_calls_per_min',
            'errors_per_min',
            'file_ops_per_min',
            'network_calls_per_min',
            
            # Resource features
            'cpu_usage',
            'memory_usage_gb',
            'resource_intensity',
            
            # Behavioral features
            'error_rate',
            'query_diversity_score',
            'success_rate_score',
            
            # Temporal features
            'is_night',
            'is_weekend',
            'hour_sin',
            'hour_cos',
            
            # Response time features
            'response_time_sec',
            'response_time_log',
            
            # Composite features
            'activity_score',
            'suspicion_score'
        ]
    
    def fit_scaler(self, df: pd.DataFrame):
        """
        Fit scaler on training data
        
        Args:
            df: Training DataFrame with features
        """
        feature_cols = self.get_feature_columns()
        X = df[feature_cols]
        
        self.scaler.fit(X)
        self.fitted = True
        
        print(f"✓ Scaler fitted on {len(feature_cols)} features")
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Transform features with scaling
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (scaled features array, feature names)
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        feature_cols = self.get_feature_columns()
        X = df[feature_cols]
        
        X_scaled = self.scaler.transform(X)
        
        return X_scaled, feature_cols
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Fit scaler and transform in one step
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (scaled features array, feature names)
        """
        self.fit_scaler(df)
        return self.transform(df)
    
    def save_scaler(self, filepath: str):
        """Save fitted scaler to file"""
        if not self.fitted:
            raise ValueError("Scaler not fitted yet")
        
        joblib.dump(self.scaler, filepath)
        print(f"✓ Scaler saved to: {filepath}")
    
    def load_scaler(self, filepath: str):
        """Load fitted scaler from file"""
        self.scaler = joblib.load(filepath)
        self.fitted = True
        print(f"✓ Scaler loaded from: {filepath}")
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 10,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time-series sequences for LSTM training
        
        Args:
            df: DataFrame with features (must be sorted by timestamp)
            sequence_length: Number of timesteps in each sequence
            stride: Step size between sequences
            
        Returns:
            Tuple of (sequences array [n_sequences, sequence_length, n_features],
                     labels array [n_sequences])
        """
        feature_cols = self.get_feature_columns()
        
        # Get features and labels
        X = df[feature_cols].values
        y = df['is_threat'].values
        
        sequences = []
        labels = []
        
        # Create sliding windows
        for i in range(0, len(X) - sequence_length + 1, stride):
            seq = X[i:i + sequence_length]
            label = y[i + sequence_length - 1]  # Label of last timestep
            
            sequences.append(seq)
            labels.append(label)
        
        return np.array(sequences), np.array(labels)


def prepare_training_data(
    csv_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split data into train/val/test sets
    
    Args:
        csv_path: Path to training data CSV
        test_size: Fraction for test set
        val_size: Fraction for validation set
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Sort by timestamp for temporal consistency
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Split data
    n = len(df)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_test - n_val
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train+n_val]
    test_df = df.iloc[n_train+n_val:]
    
    print(f"✓ Data split:")
    print(f"  - Train: {len(train_df)} samples")
    print(f"  - Val: {len(val_df)} samples")
    print(f"  - Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage
    print("=== Optikal Feature Engineering ===\n")
    
    # Load synthetic data
    train_df, val_df, test_df = prepare_training_data("optikal_training_data.csv")
    
    # Initialize feature engineer
    engineer = OptiMalFeatureEngineer()
    
    # Extract features
    print("\nExtracting features...")
    train_features = engineer.extract_features(train_df)
    val_features = engineer.extract_features(val_df)
    test_features = engineer.extract_features(test_df)
    
    print(f"✓ Extracted {len(engineer.get_feature_columns())} features")
    
    # Fit and transform
    print("\nScaling features...")
    X_train, feature_names = engineer.fit_transform(train_features)
    X_val, _ = engineer.transform(val_features)
    X_test, _ = engineer.transform(test_features)
    
    print(f"✓ Training data shape: {X_train.shape}")
    print(f"✓ Validation data shape: {X_val.shape}")
    print(f"✓ Test data shape: {X_test.shape}")
    
    # Save scaler
    engineer.save_scaler("optikal_scaler.pkl")
    
    print("\n✓ Feature engineering complete!")
