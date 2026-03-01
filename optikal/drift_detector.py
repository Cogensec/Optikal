"""
Optikal Feature Drift Detector

Monitors for distribution shift between training data and live inference inputs
using Population Stability Index (PSI) per feature. PSI >= 0.2 signals significant
drift that warrants model retraining consideration.

Usage::

    detector = FeatureDriftDetector(feature_names, window_size=500)
    detector.fit_reference(X_train)

    # During inference — auto-checks when buffer fills
    detector.update(X_batch)

    # Or check on demand
    psi_scores = detector.check_drift(X_window)
    print(detector.summary(psi_scores))
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Industry-standard PSI thresholds
PSI_NEGLIGIBLE   = 0.1   # PSI < 0.10: no meaningful change
PSI_MODERATE     = 0.1   # 0.10 <= PSI < 0.20: moderate shift, monitor closely
PSI_SIGNIFICANT  = 0.2   # PSI >= 0.20: significant drift, consider retraining


def _compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-8,
) -> float:
    """
    Population Stability Index (PSI) between two distributions.

    PSI = Σ (p_current − p_reference) × ln(p_current / p_reference)

    Args:
        reference: Reference (training) distribution values
        current:   Current (inference) distribution values
        n_bins:    Number of equal-width histogram bins
        eps:       Stability constant to avoid log(0)

    Returns:
        PSI value — 0.0 means identical distributions.
    """
    lo = min(reference.min(), current.min())
    hi = max(reference.max(), current.max()) + eps
    bins = np.linspace(lo, hi, n_bins + 1)

    ref_pct = np.histogram(reference, bins=bins)[0] / (len(reference) + eps) + eps
    cur_pct = np.histogram(current,   bins=bins)[0] / (len(current)   + eps) + eps

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class FeatureDriftDetector:
    """
    Monitors feature distribution drift between training reference data and
    live inference inputs.

    Maintains a rolling buffer of recent inference inputs. When the buffer
    reaches ``window_size`` samples, PSI is computed per feature against the
    training reference and any drifted features are logged as warnings.
    """

    def __init__(
        self,
        feature_names: List[str],
        window_size: int = 500,
        n_bins: int = 10,
        psi_threshold: float = PSI_SIGNIFICANT,
    ):
        """
        Args:
            feature_names:  Ordered list matching column order of X arrays
            window_size:    Buffer size that triggers an automatic drift check
            n_bins:         Histogram bins for PSI computation
            psi_threshold:  PSI above which a feature is flagged as drifted
        """
        self.feature_names  = feature_names
        self.window_size    = window_size
        self.n_bins         = n_bins
        self.psi_threshold  = psi_threshold

        self.reference_: Optional[np.ndarray] = None
        self._buffer: List[np.ndarray] = []
        self.fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_reference(self, X_train: np.ndarray):
        """
        Store the training distribution as the reference for all future checks.

        Args:
            X_train: Scaled training features [n_samples, n_features]
        """
        if X_train.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X_train has {X_train.shape[1]} features but "
                f"{len(self.feature_names)} feature names were provided."
            )
        self.reference_ = X_train.copy()
        self.fitted = True
        logger.info(
            "DriftDetector fitted — %d reference samples, %d features",
            X_train.shape[0], X_train.shape[1],
        )

    # ------------------------------------------------------------------
    # Drift checking
    # ------------------------------------------------------------------

    def check_drift(self, X_window: np.ndarray) -> Dict[str, float]:
        """
        Compute PSI per feature between the reference and a recent window.

        Args:
            X_window: Recent inference inputs [n_samples, n_features]

        Returns:
            Dict mapping feature_name → PSI value.
            Features with PSI >= psi_threshold are logged as warnings.
        """
        if not self.fitted:
            raise ValueError("Call fit_reference() before check_drift().")

        psi_scores: Dict[str, float] = {}
        drifted: List[str] = []

        for i, name in enumerate(self.feature_names):
            psi = _compute_psi(
                self.reference_[:, i],
                X_window[:, i],
                self.n_bins,
            )
            psi_scores[name] = psi
            if psi >= self.psi_threshold:
                drifted.append(name)

        if drifted:
            logger.warning(
                "Feature drift detected — %d feature(s) above threshold "
                "(PSI >= %.2f): %s",
                len(drifted), self.psi_threshold, drifted,
            )
            for name in drifted:
                logger.warning("  %-32s PSI=%.4f", name, psi_scores[name])
        else:
            max_psi = max(psi_scores.values()) if psi_scores else 0.0
            logger.info("No significant drift detected (max PSI=%.4f)", max_psi)

        return psi_scores

    def update(self, X_batch: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Add a batch of inference inputs to the rolling buffer.

        Automatically calls check_drift() when the buffer reaches
        ``window_size`` samples and clears the buffer afterwards.

        Args:
            X_batch: Inference input batch [n_samples, n_features]

        Returns:
            PSI score dict if a check was triggered; None otherwise.
        """
        self._buffer.append(X_batch)
        buffered = sum(len(b) for b in self._buffer)

        if buffered >= self.window_size:
            X_window = np.vstack(self._buffer)
            self._buffer = []
            logger.info(
                "Drift check triggered — %d buffered samples", len(X_window)
            )
            return self.check_drift(X_window)

        return None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self, psi_scores: Dict[str, float]) -> str:
        """Return a human-readable PSI report string."""
        lines = ["Feature Drift Report (PSI):"]
        for name, psi in sorted(psi_scores.items(), key=lambda x: -x[1]):
            if psi >= self.psi_threshold:
                flag = "DRIFT"
            elif psi >= PSI_NEGLIGIBLE:
                flag = "WATCH"
            else:
                flag = "OK"
            lines.append(f"  {name:<32} PSI={psi:.4f}  [{flag}]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        """Serialize the fitted detector to disk."""
        if not self.fitted:
            raise ValueError("Detector not fitted — call fit_reference() first.")
        joblib.dump(self, filepath)
        logger.info("DriftDetector saved to: %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "FeatureDriftDetector":
        """Load a previously serialized detector from disk."""
        obj = joblib.load(filepath)
        logger.info("DriftDetector loaded from: %s", filepath)
        return obj
