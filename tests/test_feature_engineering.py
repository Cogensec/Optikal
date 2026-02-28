"""
Tests for optikal.feature_engineering
"""

import numpy as np
import pandas as pd
import pytest

from optikal.feature_engineering import (
    OptikalFeatureEngineer,
    REQUIRED_INPUT_COLUMNS,
    prepare_training_data,
)
from optikal.data_generator import SyntheticDataGenerator

EXPECTED_N_FEATURES = 18


@pytest.fixture(scope="module")
def raw_df():
    gen = SyntheticDataGenerator(seed=7)
    return gen.generate_training_dataset(n_normal=200, n_threats_per_type=30)


@pytest.fixture(scope="module")
def features_df(raw_df):
    eng = OptikalFeatureEngineer()
    return eng.extract_features(raw_df)


@pytest.fixture(scope="module")
def fitted_engineer(features_df):
    eng = OptikalFeatureEngineer()
    eng.fit_scaler(features_df)
    return eng


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------

class TestValidateInput:
    def test_valid_data_passes(self, raw_df):
        eng = OptikalFeatureEngineer()
        ok, errors = eng.validate_input(raw_df)
        assert ok, f"Valid data failed validation: {errors}"
        assert errors == []

    def test_missing_column_detected(self, raw_df):
        bad = raw_df.drop(columns=["cpu_usage_percent"])
        ok, errors = eng_for_validation().validate_input(bad)
        assert not ok
        assert any("cpu_usage_percent" in e for e in errors)

    def test_out_of_range_cpu_detected(self, raw_df):
        bad = raw_df.copy()
        bad.loc[0, "cpu_usage_percent"] = 150  # > 100
        ok, errors = eng_for_validation().validate_input(bad)
        assert not ok
        assert any("cpu_usage_percent" in e for e in errors)

    def test_out_of_range_hour_detected(self, raw_df):
        bad = raw_df.copy()
        bad.loc[0, "hour_of_day"] = 25
        ok, errors = eng_for_validation().validate_input(bad)
        assert not ok
        assert any("hour_of_day" in e for e in errors)

    def test_non_numeric_column_detected(self, raw_df):
        bad = raw_df.copy()
        bad["cpu_usage_percent"] = "high"
        ok, errors = eng_for_validation().validate_input(bad)
        assert not ok


def eng_for_validation():
    return OptikalFeatureEngineer()


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_produces_expected_feature_columns(self, raw_df):
        eng = OptikalFeatureEngineer()
        features = eng.extract_features(raw_df)
        expected = eng.get_feature_columns()
        for col in expected:
            assert col in features.columns, f"Missing feature column: {col}"

    def test_feature_count(self):
        eng = OptikalFeatureEngineer()
        assert len(eng.get_feature_columns()) == EXPECTED_N_FEATURES

    def test_cpu_usage_normalised(self, raw_df):
        eng = OptikalFeatureEngineer()
        features = eng.extract_features(raw_df)
        assert (features['cpu_usage'] >= 0).all()
        assert (features['cpu_usage'] <= 1).all()

    def test_suspicion_score_range(self, features_df):
        # Components are [0,1]-bounded, so suspicion_score should be [0, ~1]
        assert (features_df['suspicion_score'] >= 0).all()
        assert (features_df['suspicion_score'] <= 1.1).all()  # small float tolerance

    def test_hour_cyclical_encoding(self, features_df):
        assert features_df['hour_sin'].between(-1, 1).all()
        assert features_df['hour_cos'].between(-1, 1).all()

    def test_error_rate_non_negative(self, features_df):
        assert (features_df['error_rate'] >= 0).all()

    def test_preserves_row_count(self, raw_df, features_df):
        assert len(features_df) == len(raw_df)


# ---------------------------------------------------------------------------
# Scaler: fit, transform, fit_transform
# ---------------------------------------------------------------------------

class TestScaler:
    def test_fit_transform_shape(self, features_df):
        eng = OptikalFeatureEngineer()
        X, names = eng.fit_transform(features_df)
        assert X.shape == (len(features_df), EXPECTED_N_FEATURES)
        assert len(names) == EXPECTED_N_FEATURES

    def test_transform_before_fit_raises(self, features_df):
        eng = OptikalFeatureEngineer()
        with pytest.raises(ValueError, match="Scaler not fitted"):
            eng.transform(features_df)

    def test_fitted_flag(self, features_df):
        eng = OptikalFeatureEngineer()
        assert not eng.fitted
        eng.fit_scaler(features_df)
        assert eng.fitted

    def test_scaled_output_zero_mean(self, features_df):
        """StandardScaler should produce approximately zero-mean columns."""
        eng = OptikalFeatureEngineer()
        X, _ = eng.fit_transform(features_df)
        # Mean should be near 0 for each feature (tolerance for small datasets)
        assert np.abs(X.mean(axis=0)).max() < 0.1

    def test_save_load_scaler(self, fitted_engineer, features_df, tmp_path):
        path = str(tmp_path / "scaler.pkl")
        fitted_engineer.save_scaler(path)

        eng2 = OptikalFeatureEngineer()
        eng2.load_scaler(path)
        assert eng2.fitted

        X1, _ = fitted_engineer.transform(features_df)
        X2, _ = eng2.transform(features_df)
        np.testing.assert_array_almost_equal(X1, X2)

    def test_save_unfitted_raises(self, tmp_path):
        eng = OptikalFeatureEngineer()
        with pytest.raises(ValueError):
            eng.save_scaler(str(tmp_path / "scaler.pkl"))


# ---------------------------------------------------------------------------
# create_sequences_from_array  (P0-2 fix)
# ---------------------------------------------------------------------------

class TestCreateSequencesFromArray:
    def test_output_shape(self, X_scaled, y_labels):
        eng = OptikalFeatureEngineer()
        seq_len = 10
        X_seq, y_seq = eng.create_sequences_from_array(X_scaled, y_labels, seq_len)

        expected_n_sequences = len(X_scaled) - seq_len + 1
        assert X_seq.shape == (expected_n_sequences, seq_len, X_scaled.shape[1])
        assert y_seq.shape == (expected_n_sequences,)

    def test_label_is_last_timestep(self, X_scaled, y_labels):
        eng = OptikalFeatureEngineer()
        seq_len = 5
        _, y_seq = eng.create_sequences_from_array(X_scaled, y_labels, seq_len)
        # Label of first sequence should equal y_labels[seq_len - 1]
        assert y_seq[0] == y_labels[seq_len - 1]

    def test_stride(self, X_scaled, y_labels):
        eng = OptikalFeatureEngineer()
        seq_len, stride = 5, 2
        X_seq, _ = eng.create_sequences_from_array(X_scaled, y_labels, seq_len, stride)
        # Number of sequences with stride
        expected = len(range(0, len(X_scaled) - seq_len + 1, stride))
        assert len(X_seq) == expected

    def test_minimum_samples_for_one_sequence(self):
        eng = OptikalFeatureEngineer()
        seq_len = 10
        X = np.random.rand(seq_len, 18)
        y = np.zeros(seq_len, dtype=int)
        X_seq, y_seq = eng.create_sequences_from_array(X, y, seq_len)
        assert len(X_seq) == 1

    def test_fewer_samples_than_seq_len_returns_empty(self):
        eng = OptikalFeatureEngineer()
        X = np.random.rand(5, 18)
        y = np.zeros(5, dtype=int)
        X_seq, y_seq = eng.create_sequences_from_array(X, y, sequence_length=10)
        assert len(X_seq) == 0
        assert len(y_seq) == 0


# ---------------------------------------------------------------------------
# prepare_training_data
# ---------------------------------------------------------------------------

class TestPrepareTrainingData:
    def test_split_sizes(self, tmp_path, raw_df):
        csv = str(tmp_path / "data.csv")
        raw_df.to_csv(csv, index=False)

        train, val, test = prepare_training_data(csv, test_size=0.2, val_size=0.1)
        total = len(train) + len(val) + len(test)
        assert total == len(raw_df)
        assert len(test) == pytest.approx(len(raw_df) * 0.2, abs=2)
        assert len(val)  == pytest.approx(len(raw_df) * 0.1, abs=2)

    def test_temporal_ordering(self, tmp_path, raw_df):
        """Rows should be sorted by timestamp after splitting."""
        csv = str(tmp_path / "data.csv")
        raw_df.to_csv(csv, index=False)
        train, val, test = prepare_training_data(csv)
        # Timestamps within each split should be non-decreasing
        for split in [train, val, test]:
            ts = pd.to_datetime(split['timestamp'])
            assert (ts.diff().dropna() >= pd.Timedelta(0)).all()
