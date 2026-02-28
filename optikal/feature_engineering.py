"""
Feature Engineering for Optikal

Extracts security-relevant features from agent behavioral data
for threat detection model training.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

# Required raw input columns for feature extraction
REQUIRED_INPUT_COLUMNS = [
    "cpu_usage_percent",
    "memory_usage_mb",
    "network_calls_count",
    "file_operations_count",
    "api_calls_count",
    "error_count",
    "hour_of_day",
    "day_of_week",
    "query_diversity",
    "response_time_ms",
    "success_rate",
]

# Value bounds for each input column (None means no upper/lower bound)
COLUMN_BOUNDS = {
    "cpu_usage_percent":     (0, 100),
    "memory_usage_mb":       (0, None),
    "network_calls_count":   (0, None),
    "file_operations_count": (0, None),
    "api_calls_count":       (0, None),
    "error_count":           (0, None),
    "hour_of_day":           (0, 23),
    "day_of_week":           (0, 6),
    "query_diversity":       (0, 1),
    "response_time_ms":      (0, None),
    "success_rate":          (0, 1),
}


class OptikalFeatureEngineer:
    """
    Feature engineering pipeline for Optikal threat detection.
    """

    def __init__(self):
        """Initialize feature engineer."""
        self.scaler = StandardScaler()
        self.fitted = False

    def validate_input(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate a raw input DataFrame before feature extraction.

        Checks that all required columns are present, are numeric, and fall
        within their expected value ranges. Returns a list of errors rather
        than raising immediately so callers can decide whether to log-and-skip
        or hard-fail.

        Args:
            df: Raw behavioral input DataFrame

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors: List[str] = []

        # 1. Check required columns are present
        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")

        # 2. Check dtype and value bounds for each present column
        for col, (lo, hi) in COLUMN_BOUNDS.items():
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' must be numeric, got {df[col].dtype}")
                continue
            if lo is not None and (df[col] < lo).any():
                n_bad = int((df[col] < lo).sum())
                errors.append(
                    f"Column '{col}' has {n_bad} value(s) below minimum {lo}"
                )
            if hi is not None and (df[col] > hi).any():
                n_bad = int((df[col] > hi).sum())
                errors.append(
                    f"Column '{col}' has {n_bad} value(s) above maximum {hi}"
                )

        return len(errors) == 0, errors

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from raw agent behavioral data.

        Validates input first; logs warnings for any issues but continues
        to allow callers to handle partial data gracefully.

        Args:
            df: Raw behavioral data

        Returns:
            DataFrame with engineered features appended
        """
        valid, errors = self.validate_input(df)
        if not valid:
            logger.warning("Input validation failed (%d issue(s)): %s", len(errors), errors)

        features_df = df.copy()

        # === Rate-based Features ===
        features_df['api_calls_per_min'] = features_df['api_calls_count']
        features_df['errors_per_min'] = features_df['error_count']
        features_df['file_ops_per_min'] = features_df['file_operations_count']
        features_df['network_calls_per_min'] = features_df['network_calls_count']

        # === Resource Features ===
        features_df['cpu_usage'] = features_df['cpu_usage_percent'] / 100.0
        features_df['memory_usage_gb'] = features_df['memory_usage_mb'] / 1024.0
        features_df['resource_intensity'] = (
            features_df['cpu_usage'] * features_df['memory_usage_gb']
        )

        # === Behavioral Features ===
        features_df['error_rate'] = (
            features_df['error_count'] / (features_df['api_calls_count'] + 1)
        )
        features_df['query_diversity_score'] = features_df['query_diversity']
        features_df['success_rate_score'] = features_df['success_rate']

        # === Temporal Features ===
        features_df['is_night'] = (
            (features_df['hour_of_day'] >= 0) & (features_df['hour_of_day'] <= 6)
        ).astype(int)
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
        features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour_of_day'] / 24)
        features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour_of_day'] / 24)

        # === Response Time Features ===
        features_df['response_time_sec'] = features_df['response_time_ms'] / 1000.0
        features_df['response_time_log'] = np.log1p(features_df['response_time_ms'])

        # === Composite Anomaly Indicators ===
        features_df['activity_score'] = (
            features_df['api_calls_count'] / 100.0
            + features_df['network_calls_count'] / 50.0
            + features_df['file_operations_count'] / 10.0
        ) / 3.0

        # Suspicious behavior score (hand-crafted signal)
        features_df['suspicion_score'] = (
            (1 - features_df['query_diversity']) * 0.3   # Low diversity is suspicious
            + features_df['error_rate'] * 0.3             # High errors suspicious
            + features_df['is_night'] * 0.2               # Night activity suspicious
            + (1 - features_df['success_rate']) * 0.2     # Low success suspicious
        )

        return features_df

    def get_feature_columns(self) -> List[str]:
        """Return the ordered list of feature column names used by the model."""
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
            'suspicion_score',
        ]

    def fit_scaler(self, df: pd.DataFrame):
        """
        Fit the StandardScaler on training data.

        Args:
            df: Training DataFrame with engineered features
        """
        feature_cols = self.get_feature_columns()
        self.scaler.fit(df[feature_cols])
        self.fitted = True
        logger.info("Scaler fitted on %d features", len(feature_cols))

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Scale features using the fitted scaler.

        Args:
            df: DataFrame with engineered features

        Returns:
            Tuple of (scaled_array [n_samples, n_features], feature_names)
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        feature_cols = self.get_feature_columns()
        X_scaled = self.scaler.transform(df[feature_cols])
        return X_scaled, feature_cols

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Fit the scaler and transform features in a single step.

        Args:
            df: Training DataFrame with engineered features

        Returns:
            Tuple of (scaled_array [n_samples, n_features], feature_names)
        """
        self.fit_scaler(df)
        return self.transform(df)

    def save_scaler(self, filepath: str):
        """Serialize the fitted scaler to disk."""
        if not self.fitted:
            raise ValueError("Scaler not fitted yet")
        joblib.dump(self.scaler, filepath)
        logger.info("Scaler saved to: %s", filepath)

    def load_scaler(self, filepath: str):
        """Load a previously serialized scaler from disk."""
        self.scaler = joblib.load(filepath)
        self.fitted = True
        logger.info("Scaler loaded from: %s", filepath)

    def create_sequences_from_array(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        sequence_length: int = 10,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time-series sequences for LSTM training from pre-scaled arrays.

        This is the preferred method over create_sequences() because it operates
        on already-scaled numpy arrays, avoiding training/inference distribution
        mismatch caused by feeding unscaled values to the LSTM.

        Args:
            X_scaled:        Scaled feature array [n_samples, n_features]
            y:               Label array [n_samples]
            sequence_length: Number of timesteps per sequence
            stride:          Step size between consecutive sequences

        Returns:
            Tuple of:
              - sequences [n_sequences, sequence_length, n_features]
              - labels    [n_sequences]  (label of the last timestep in each window)
        """
        sequences: List[np.ndarray] = []
        labels: List[int] = []

        for i in range(0, len(X_scaled) - sequence_length + 1, stride):
            sequences.append(X_scaled[i:i + sequence_length])
            labels.append(y[i + sequence_length - 1])

        return np.array(sequences), np.array(labels)

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 10,
        stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time-series sequences from a raw (unscaled) DataFrame.

        Warning: This method reads unscaled feature values directly from the
        DataFrame. Prefer create_sequences_from_array() when you have already-
        scaled data to avoid a training/inference distribution mismatch.

        Args:
            df:              DataFrame with feature columns (sorted by timestamp)
            sequence_length: Number of timesteps per sequence
            stride:          Step size between sequences

        Returns:
            Tuple of (sequences [n_sequences, seq_len, n_features], labels [n_sequences])
        """
        feature_cols = self.get_feature_columns()
        X = df[feature_cols].values
        y = df['is_threat'].values

        sequences: List[np.ndarray] = []
        labels: List[int] = []

        for i in range(0, len(X) - sequence_length + 1, stride):
            sequences.append(X[i:i + sequence_length])
            labels.append(y[i + sequence_length - 1])

        return np.array(sequences), np.array(labels)


def prepare_training_data(
    csv_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a CSV dataset and split it into train / validation / test sets.

    Data is sorted by timestamp before splitting to preserve temporal order.

    Args:
        csv_path:  Path to training data CSV
        test_size: Fraction of data reserved for the test set
        val_size:  Fraction of data reserved for the validation set

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Sort by timestamp for temporal consistency
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    n = len(df)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_test - n_val

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    logger.info(
        "Data split — train: %d, val: %d, test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    return train_df, val_df, test_df


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="[%(levelname)s] %(message)s")

    train_df, val_df, test_df = prepare_training_data("optikal_training_data.csv")

    engineer = OptikalFeatureEngineer()

    train_features = engineer.extract_features(train_df)
    val_features = engineer.extract_features(val_df)
    test_features = engineer.extract_features(test_df)

    logger.info("Extracted %d features", len(engineer.get_feature_columns()))

    X_train, feature_names = engineer.fit_transform(train_features)
    X_val, _ = engineer.transform(val_features)
    X_test, _ = engineer.transform(test_features)

    logger.info("Train shape: %s, Val shape: %s, Test shape: %s",
                X_train.shape, X_val.shape, X_test.shape)

    engineer.save_scaler("optikal_scaler.pkl")
    logger.info("Feature engineering complete!")
