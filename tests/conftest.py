"""
Shared pytest fixtures for Optikal tests.

All fixtures are scoped to the session or module to avoid regenerating
data for every individual test.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the optikal package importable when tests run from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "optikal"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from optikal.data_generator import SyntheticDataGenerator
from optikal.feature_engineering import OptikalFeatureEngineer


# ---------------------------------------------------------------------------
# Raw data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def generator():
    """A SyntheticDataGenerator with a fixed seed."""
    return SyntheticDataGenerator(seed=0)


@pytest.fixture(scope="session")
def normal_df(generator):
    """200 normal behaviour samples."""
    return generator.generate_normal_behavior(n_samples=200)


@pytest.fixture(scope="session")
def credential_df(generator):
    """50 credential abuse samples."""
    return generator.generate_credential_abuse(n_samples=50)


@pytest.fixture(scope="session")
def injection_df(generator):
    """50 prompt injection samples."""
    return generator.generate_prompt_injection(n_samples=50)


@pytest.fixture(scope="session")
def escalation_df(generator):
    """50 privilege escalation samples."""
    return generator.generate_privilege_escalation(n_samples=50)


@pytest.fixture(scope="session")
def mixed_dataset(generator):
    """Small but complete dataset spanning all threat types."""
    return generator.generate_training_dataset(
        n_normal=200,
        n_threats_per_type=40,
    )


# ---------------------------------------------------------------------------
# Feature engineering fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def engineer(mixed_dataset):
    """A fitted OptikalFeatureEngineer."""
    eng = OptikalFeatureEngineer()
    features = eng.extract_features(mixed_dataset)
    eng.fit_scaler(features)
    return eng


@pytest.fixture(scope="session")
def train_features(engineer, mixed_dataset):
    """Extracted (but unscaled) feature DataFrame for the full mixed dataset."""
    return engineer.extract_features(mixed_dataset)


@pytest.fixture(scope="session")
def X_scaled(engineer, train_features):
    """Scaled feature array for the full mixed dataset."""
    X, _ = engineer.transform(train_features)
    return X


@pytest.fixture(scope="session")
def y_labels(train_features):
    """Binary threat labels aligned to X_scaled."""
    return train_features['is_threat'].values
