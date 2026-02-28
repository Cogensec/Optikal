"""
Optikal Configuration System

Centralises all hyperparameters and runtime settings into typed dataclasses.
This replaces hardcoded values scattered across trainer, quick-train, and
feature engineering files, making every training run reproducible.

Usage::

    from optikal.config import OptikalConfig

    # Use defaults
    cfg = OptikalConfig()

    # Load from YAML
    cfg = OptikalConfig.from_yaml("config.yaml")

    # Save a run's configuration
    cfg.to_yaml("run_2026_02_28.yaml")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Sub-configurations
# =============================================================================

@dataclass
class DataConfig:
    """Settings for synthetic data generation."""

    n_normal: int = 5000
    """Number of normal behaviour samples."""

    n_threats_per_type: int = 500
    """Number of samples to generate for each threat scenario."""

    random_seed: int = 42
    """Random seed for reproducibility."""

    test_size: float = 0.20
    """Fraction of data reserved for the test set."""

    val_size: float = 0.10
    """Fraction of data reserved for the validation set."""


@dataclass
class IsolationForestConfig:
    """Hyperparameters for the Isolation Forest component."""

    contamination: float = 0.15
    """Expected proportion of anomalies in training data (0 < contamination < 0.5)."""

    n_estimators: int = 100
    """Number of base estimators (trees)."""

    max_samples: int = 256
    """Number of samples drawn to train each base estimator."""

    random_state: int = 42

    threshold: float = 0.5
    """Decision threshold applied to sigmoid-normalised anomaly scores."""


@dataclass
class LSTMConfig:
    """Architecture and training settings for the LSTM component."""

    sequence_length: int = 10
    """Number of timesteps in each input window."""

    lstm_units: List[int] = field(default_factory=lambda: [64, 32])
    """Units in each LSTM layer (one entry per layer)."""

    dense_units: List[int] = field(default_factory=lambda: [16])
    """Units in fully-connected layers after the LSTM stack."""

    dropout_rate: float = 0.2
    """Dropout applied after each LSTM layer."""

    dense_dropout_rate: float = 0.1
    """Dropout applied after dense layers."""

    learning_rate: float = 0.001
    """Adam optimiser learning rate."""

    epochs: int = 20
    """Maximum training epochs (early stopping may terminate sooner)."""

    batch_size: int = 32

    early_stopping_patience: int = 3
    """Epochs without improvement before training is stopped."""

    threshold: float = 0.5
    """Sigmoid output threshold for binary classification."""


@dataclass
class EnsembleConfig:
    """Settings for the weighted ensemble fusion."""

    if_weight: float = 0.4
    """Relative weight for the Isolation Forest score."""

    lstm_weight: float = 0.6
    """Relative weight for the LSTM score."""

    threshold: float = 0.5
    """Ensemble score threshold for flagging a threat."""


@dataclass
class ThreatClassifierConfig:
    """Settings for the post-hoc multi-class ThreatClassifier."""

    n_estimators: int = 100
    random_state: int = 42
    enabled: bool = True
    """Set to False to skip training the ThreatClassifier."""


@dataclass
class ExplainabilityConfig:
    """Settings for SHAP-based feature attribution."""

    enabled: bool = True
    """Set to False to skip fitting the explainer (faster training)."""

    n_background_samples: int = 100
    """Number of background samples used to estimate SHAP baseline."""

    top_k_features: int = 5
    """Number of top features to return in explain() output."""


@dataclass
class OptikalConfig:
    """
    Top-level Optikal configuration.

    Aggregates all component configs and runtime settings. Pass an instance
    to training functions to ensure full reproducibility of every run.

    Example::

        cfg = OptikalConfig(
            data=DataConfig(n_normal=10_000, n_threats_per_type=1_000),
            isolation_forest=IsolationForestConfig(contamination=0.10),
        )
        cfg.to_yaml("my_run.yaml")
    """

    data:               DataConfig            = field(default_factory=DataConfig)
    isolation_forest:   IsolationForestConfig = field(default_factory=IsolationForestConfig)
    lstm:               LSTMConfig            = field(default_factory=LSTMConfig)
    ensemble:           EnsembleConfig        = field(default_factory=EnsembleConfig)
    threat_classifier:  ThreatClassifierConfig = field(default_factory=ThreatClassifierConfig)
    explainability:     ExplainabilityConfig  = field(default_factory=ExplainabilityConfig)

    output_dir: str = "optikal_models"
    """Directory where all trained model artefacts are written."""

    log_format: str = "plain"
    """Logging format: 'plain' or 'json'."""

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to a plain nested dict (JSON-serialisable)."""
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Serialise configuration to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Configuration written to: %s", path)

    @classmethod
    def from_dict(cls, d: dict) -> "OptikalConfig":
        """Reconstruct from a nested dict (e.g. loaded from JSON/YAML)."""
        return cls(
            data=DataConfig(**d.get("data", {})),
            isolation_forest=IsolationForestConfig(**d.get("isolation_forest", {})),
            lstm=LSTMConfig(**d.get("lstm", {})),
            ensemble=EnsembleConfig(**d.get("ensemble", {})),
            threat_classifier=ThreatClassifierConfig(**d.get("threat_classifier", {})),
            explainability=ExplainabilityConfig(**d.get("explainability", {})),
            output_dir=d.get("output_dir", "optikal_models"),
            log_format=d.get("log_format", "plain"),
        )

    @classmethod
    def from_json(cls, path: str) -> "OptikalConfig":
        """Load configuration from a JSON file."""
        d = json.loads(Path(path).read_text())
        cfg = cls.from_dict(d)
        logger.info("Configuration loaded from: %s", path)
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "OptikalConfig":
        """
        Load configuration from a YAML file.

        Requires PyYAML (``pip install pyyaml``).
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML config support: pip install pyyaml"
            ) from exc

        d = yaml.safe_load(Path(path).read_text())
        cfg = cls.from_dict(d or {})
        logger.info("Configuration loaded from: %s", path)
        return cfg

    def to_yaml(self, path: str) -> None:
        """
        Serialise configuration to a YAML file.

        Requires PyYAML (``pip install pyyaml``).
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML config support: pip install pyyaml"
            ) from exc

        Path(path).write_text(yaml.dump(self.to_dict(), default_flow_style=False))
        logger.info("Configuration written to: %s", path)
