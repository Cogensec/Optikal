"""
Optikal Active Learning Pipeline

Buffers high-uncertainty inference samples for human labeling. When enough
labeled samples accumulate, the configured retrain callback is invoked so
the model can learn from real (non-synthetic) agent behavior.

Uncertainty is derived from the Isolation Forest anomaly score:

    uncertainty = 1 - 2 * |score - 0.5|

A sample at score=0.5 is maximally uncertain (uncertainty=1.0); one at
score=0.0 or 1.0 is fully certain (uncertainty=0.0).

Usage::

    buffer = ActiveLearningBuffer(
        uncertainty_threshold=0.3,
        min_labeled=200,
        retrain_callback=my_retrain_fn,
    )

    # During inference
    buffer.add_uncertain(X_batch, scores, sample_ids)

    # After a human analyst labels samples
    buffer.add_label("sample_123", label=1, threat_type="credential_abuse")

    # Check readiness and stats
    print(buffer.stats)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LabeledSample:
    """A human-labeled sample retained for retraining."""

    sample_id:   str
    features:    List[float]    # Scaled feature vector
    label:       int            # 0 = normal, 1 = threat
    threat_type: Optional[str]  # e.g. "credential_abuse" (None for normal)
    labeled_at:  float          # Unix timestamp


class ActiveLearningBuffer:
    """
    Buffers high-uncertainty inference inputs and triggers retraining once
    ``min_labeled`` human-labeled samples have been provided.

    The buffer evicts the least-uncertain sample when it reaches
    ``buffer_capacity`` to keep memory bounded.
    """

    def __init__(
        self,
        uncertainty_threshold: float = 0.3,
        min_labeled: int = 200,
        buffer_capacity: int = 10_000,
        retrain_callback: Optional[Callable] = None,
        persist_dir: Optional[str] = None,
    ):
        """
        Args:
            uncertainty_threshold: Samples with uncertainty >= this value are buffered.
            min_labeled:           Labeled samples required to trigger retraining.
            buffer_capacity:       Maximum unlabeled samples kept in memory.
            retrain_callback:      Called with ``(X, y, threat_types)`` when enough
                                   labeled samples are ready. Signature:
                                   ``callback(X: np.ndarray, y: np.ndarray,
                                   threat_types: np.ndarray) -> None``
            persist_dir:           If set, the buffer is saved here after each retrain.
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.min_labeled           = min_labeled
        self.buffer_capacity       = buffer_capacity
        self.retrain_callback      = retrain_callback
        self.persist_dir           = Path(persist_dir) if persist_dir else None

        # Unlabeled pool: sample_id → {features, score, uncertainty, added_at}
        self._unlabeled: Dict[str, dict] = {}
        # Labeled samples waiting to trigger retraining
        self._labeled: List[LabeledSample] = []
        self._retrain_count = 0

    # ------------------------------------------------------------------
    # Uncertainty calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _uncertainty(scores: np.ndarray) -> np.ndarray:
        """Convert IF anomaly scores to uncertainty values in [0, 1]."""
        return 1.0 - 2.0 * np.abs(scores - 0.5)

    # ------------------------------------------------------------------
    # Adding samples
    # ------------------------------------------------------------------

    def add_uncertain(
        self,
        X: np.ndarray,
        scores: np.ndarray,
        sample_ids: Optional[List[str]] = None,
    ) -> int:
        """
        Buffer samples whose IF score falls near the decision boundary.

        Args:
            X:          Scaled feature array [n_samples, n_features]
            scores:     IF anomaly scores [n_samples] in [0, 1]
            sample_ids: Optional per-sample IDs (auto-generated if None)

        Returns:
            Number of samples added to the buffer.
        """
        uncertainties = self._uncertainty(scores)
        mask = uncertainties >= self.uncertainty_threshold

        if not mask.any():
            return 0

        if sample_ids is None:
            t = time.time()
            sample_ids = [f"s_{t}_{i}" for i in range(len(X))]

        n_added = 0
        selected_X   = X[mask]
        selected_sc  = scores[mask]
        selected_unc = uncertainties[mask]
        selected_ids = [sid for sid, m in zip(sample_ids, mask) if m]

        for x, sc, unc, sid in zip(selected_X, selected_sc, selected_unc, selected_ids):
            if len(self._unlabeled) >= self.buffer_capacity:
                # Evict the sample closest to certainty (lowest uncertainty)
                evict_id = min(
                    self._unlabeled,
                    key=lambda k: self._unlabeled[k]["uncertainty"],
                )
                del self._unlabeled[evict_id]

            self._unlabeled[sid] = {
                "features":    x.tolist(),
                "score":       float(sc),
                "uncertainty": float(unc),
                "added_at":    time.time(),
            }
            n_added += 1

        logger.debug(
            "ActiveLearningBuffer: +%d samples buffered (total: %d)",
            n_added, len(self._unlabeled),
        )
        return n_added

    def add_label(
        self,
        sample_id: str,
        label: int,
        threat_type: Optional[str] = None,
    ) -> bool:
        """
        Record a human label for a buffered sample.

        Automatically triggers retraining if ``min_labeled`` is reached.

        Args:
            sample_id:   ID of the buffered sample to label
            label:       0 = normal, 1 = threat
            threat_type: Optional threat type string for ThreatClassifier

        Returns:
            True if the sample was found and labeled; False if not found.
        """
        if sample_id not in self._unlabeled:
            logger.warning(
                "Sample '%s' not found in the active learning buffer", sample_id
            )
            return False

        entry = self._unlabeled.pop(sample_id)
        self._labeled.append(LabeledSample(
            sample_id=sample_id,
            features=entry["features"],
            label=label,
            threat_type=threat_type,
            labeled_at=time.time(),
        ))

        logger.debug(
            "Label recorded — id: %s  label: %d  type: %s  (total labeled: %d)",
            sample_id, label, threat_type, len(self._labeled),
        )

        if self.should_retrain():
            logger.info(
                "Retraining threshold reached (%d labeled samples)",
                len(self._labeled),
            )
            self.trigger_retrain()

        return True

    # ------------------------------------------------------------------
    # Retraining
    # ------------------------------------------------------------------

    def should_retrain(self) -> bool:
        """Return True when ``min_labeled`` samples are ready."""
        return len(self._labeled) >= self.min_labeled

    def trigger_retrain(self):
        """
        Invoke the retrain callback with all accumulated labeled samples.

        The callback receives:
          - ``X``            np.ndarray [n_labeled, n_features]
          - ``y``            np.ndarray [n_labeled] — 0/1 labels
          - ``threat_types`` np.ndarray [n_labeled] — str labels

        The labeled buffer is cleared after a successful call.
        """
        if not self.retrain_callback:
            logger.warning(
                "No retrain_callback configured — set one to enable auto-retraining"
            )
            return

        X = np.array([s.features for s in self._labeled])
        y = np.array([s.label    for s in self._labeled])
        threat_types = np.array([
            s.threat_type if s.threat_type else "unknown"
            for s in self._labeled
        ])

        logger.info(
            "Triggering retrain — %d samples (threats: %d, normal: %d)",
            len(X), int(y.sum()), int((y == 0).sum()),
        )

        self.retrain_callback(X, y, threat_types)
        self._retrain_count += 1
        self._labeled = []  # Reset labeled pool for next round

        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.save(self.persist_dir / "active_learning_buffer.pkl")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict:
        """Return current buffer statistics."""
        return {
            "unlabeled_count":  len(self._unlabeled),
            "labeled_count":    len(self._labeled),
            "retrain_count":    self._retrain_count,
            "buffer_capacity":  self.buffer_capacity,
            "min_labeled":      self.min_labeled,
            "ready_to_retrain": self.should_retrain(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath):
        """Serialize buffer state to disk."""
        joblib.dump(self, filepath)
        logger.info("ActiveLearningBuffer saved to: %s", filepath)

    @classmethod
    def load(cls, filepath) -> "ActiveLearningBuffer":
        """Load a previously saved buffer from disk."""
        obj = joblib.load(filepath)
        logger.info("ActiveLearningBuffer loaded from: %s", filepath)
        return obj
