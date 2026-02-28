"""
Tests for optikal.optikal_trainer

Covers the Isolation Forest, ThreatClassifier, and ensemble without
requiring TensorFlow (LSTM tests are marked slow/integration).
"""

import numpy as np
import pytest

from optikal.optikal_trainer import (
    OptikalIsolationForest,
    ThreatClassifier,
    OptikalEnsemble,
)
from optikal.data_generator import SyntheticDataGenerator, ThreatScenario
from optikal.feature_engineering import OptikalFeatureEngineer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def training_data():
    gen = SyntheticDataGenerator(seed=13)
    df = gen.generate_training_dataset(n_normal=300, n_threats_per_type=60)
    eng = OptikalFeatureEngineer()
    features = eng.extract_features(df)
    X, feature_names = eng.fit_transform(features)
    y = features['is_threat'].values
    threat_types = features['threat_type'].values
    return X, y, threat_types, feature_names


@pytest.fixture(scope="module")
def trained_if(training_data):
    X, y, _, _ = training_data
    model = OptikalIsolationForest(contamination=0.25, random_state=42)
    model.train(X)
    return model


@pytest.fixture(scope="module")
def trained_classifier(training_data):
    X, y, threat_types, _ = training_data
    clf = ThreatClassifier()
    threat_mask = y == 1
    clf.train(X[threat_mask], threat_types[threat_mask])
    return clf


# ---------------------------------------------------------------------------
# OptikalIsolationForest
# ---------------------------------------------------------------------------

class TestOptikalIsolationForest:
    def test_fitted_flag_after_train(self, trained_if):
        assert trained_if.fitted is True

    def test_unfitted_raises(self, training_data):
        X, _, _, _ = training_data
        model = OptikalIsolationForest()
        with pytest.raises(ValueError, match="not trained"):
            model.predict_anomaly_score(X[:5])

    def test_scores_in_unit_interval(self, trained_if, training_data):
        X, _, _, _ = training_data
        scores = trained_if.predict_anomaly_score(X)
        assert scores.min() >= 0.0, "Scores below 0"
        assert scores.max() <= 1.0, "Scores above 1"

    def test_scores_shape(self, trained_if, training_data):
        X, _, _, _ = training_data
        scores = trained_if.predict_anomaly_score(X)
        assert scores.shape == (len(X),)

    def test_predictions_binary(self, trained_if, training_data):
        X, _, _, _ = training_data
        preds = trained_if.predict(X)
        assert set(preds).issubset({0, 1})

    def test_save_load(self, trained_if, training_data, tmp_path):
        path = str(tmp_path / "if_model.pkl")
        trained_if.save(path)

        loaded = OptikalIsolationForest()
        import joblib
        loaded.model = joblib.load(path)
        loaded.fitted = True

        X, _, _, _ = training_data
        orig_scores   = trained_if.predict_anomaly_score(X)
        loaded_scores = loaded.predict_anomaly_score(X)
        np.testing.assert_array_almost_equal(orig_scores, loaded_scores)

    def test_higher_scores_for_threats(self, trained_if, training_data):
        """On average, threats should score higher than normal samples."""
        X, y, _, _ = training_data
        scores = trained_if.predict_anomaly_score(X)
        mean_threat = scores[y == 1].mean()
        mean_normal = scores[y == 0].mean()
        assert mean_threat > mean_normal, (
            f"Expected threats (mean={mean_threat:.3f}) to score higher "
            f"than normal (mean={mean_normal:.3f})"
        )


# ---------------------------------------------------------------------------
# ThreatClassifier  (P1-5)
# ---------------------------------------------------------------------------

class TestThreatClassifier:
    def test_fitted_flag(self, trained_classifier):
        assert trained_classifier.fitted is True

    def test_classes_cover_all_threat_types(self, trained_classifier):
        expected = {
            ThreatScenario.CREDENTIAL_ABUSE,
            ThreatScenario.DATA_EXFILTRATION,
            ThreatScenario.RESOURCE_ABUSE,
            ThreatScenario.BEHAVIORAL_DRIFT,
            ThreatScenario.PROMPT_INJECTION,
            ThreatScenario.PRIVILEGE_ESCALATION,
        }
        actual = set(trained_classifier.classes_)
        assert expected.issubset(actual), (
            f"Missing threat classes: {expected - actual}"
        )

    def test_predict_proba_returns_all_classes(self, trained_classifier, training_data):
        X, y, _, _ = training_data
        proba = trained_classifier.predict_proba(X[:5])
        assert isinstance(proba, dict)
        for cls in trained_classifier.classes_:
            assert cls in proba
            assert proba[cls].shape == (5,)

    def test_proba_sums_to_one(self, trained_classifier, training_data):
        X, y, _, _ = training_data
        proba = trained_classifier.predict_proba(X[:10])
        total = np.stack(list(proba.values()), axis=0).sum(axis=0)
        np.testing.assert_allclose(total, np.ones(10), atol=1e-6)

    def test_predict_returns_known_class(self, trained_classifier, training_data):
        X, y, _, _ = training_data
        preds = trained_classifier.predict(X[:10])
        for p in preds:
            assert p in trained_classifier.classes_

    def test_unfitted_predict_raises(self, training_data):
        X, _, _, _ = training_data
        clf = ThreatClassifier()
        with pytest.raises(ValueError, match="not trained"):
            clf.predict(X[:3])

    def test_unfitted_predict_proba_raises(self, training_data):
        X, _, _, _ = training_data
        clf = ThreatClassifier()
        with pytest.raises(ValueError, match="not trained"):
            clf.predict_proba(X[:3])

    def test_save_load(self, trained_classifier, training_data, tmp_path):
        path = str(tmp_path / "threat_clf.pkl")
        trained_classifier.save(path)
        loaded = ThreatClassifier.load(path)
        assert loaded.fitted
        assert set(loaded.classes_) == set(trained_classifier.classes_)

        X, _, _, _ = training_data
        np.testing.assert_array_equal(
            trained_classifier.predict(X[:5]),
            loaded.predict(X[:5]),
        )


# ---------------------------------------------------------------------------
# OptikalEnsemble (IF-only mode — no LSTM required)
# ---------------------------------------------------------------------------

class TestOptikalEnsembleIFOnly:
    """
    Tests the ensemble in IF-only mode by passing the same IF scores for
    the LSTM slot via a simple mock that satisfies the interface.
    """

    class _MockLSTM:
        """Minimal LSTM stub that returns zeros."""
        fitted = True

        def predict_score(self, X):
            return np.zeros(len(X))

    def _make_ensemble(self, trained_if, trained_classifier=None):
        return OptikalEnsemble(
            isolation_forest=trained_if,
            lstm=self._MockLSTM(),
            threat_classifier=trained_classifier,
            if_weight=0.4,
            lstm_weight=0.6,
        )

    def test_weights_normalised(self, trained_if):
        ens = self._make_ensemble(trained_if)
        assert abs(ens.if_weight + ens.lstm_weight - 1.0) < 1e-9

    def test_predict_score_shape(self, trained_if, training_data):
        X, y, _, _ = training_data
        ens = self._make_ensemble(trained_if)
        # Mock LSTM returns zeros, so we pass X for both tabular and sequential
        scores = ens.predict_score(X, X)
        assert scores.shape == (len(X),)

    def test_predict_returns_binary(self, trained_if, training_data):
        X, y, _, _ = training_data
        ens = self._make_ensemble(trained_if)
        preds, _ = ens.predict(X, X)
        assert set(preds).issubset({0, 1})

    def test_detailed_scores_keys(self, trained_if, training_data):
        X, y, _, _ = training_data
        ens = self._make_ensemble(trained_if)
        _, details = ens.predict(X, X)
        assert 'ensemble' in details
        assert 'isolation_forest' in details
        assert 'lstm' in details

    def test_threat_type_in_details_when_classifier_present(
        self, trained_if, trained_classifier, training_data
    ):
        X, y, _, _ = training_data
        ens = self._make_ensemble(trained_if, trained_classifier)
        _, details = ens.predict(X, X)
        assert 'threat_type' in details
        assert 'threat_type_probs' in details
