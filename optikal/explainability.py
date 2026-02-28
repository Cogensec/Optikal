"""
Optikal Explainability Module

Provides SHAP-based feature attribution for the Isolation Forest component,
so that operators receive actionable context alongside a threat alert
("cpu_usage drove 43% of this score") rather than a raw number.

The README showed SHAP usage in its examples but no implementation existed.
This module implements it.

Usage::

    from optikal.explainability import OptikalExplainer

    explainer = OptikalExplainer(feature_names)
    explainer.fit(if_model, X_train_background)
    explainer.save("optikal_models/optikal_explainer.pkl")

    result = explainer.explain(X_sample)
    print(result["top_features"])   # [("cpu_usage", 0.42), ("error_rate", 0.31), ...]

Requirements::

    pip install shap>=0.42.0
    # or: pip install optikal[explain]
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import joblib

logger = logging.getLogger(__name__)


def _require_shap():
    """Import shap lazily and raise a helpful error if missing."""
    try:
        import shap
        return shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for explainability support.\n"
            "Install it with:  pip install shap>=0.42.0\n"
            "or:               pip install optikal[explain]"
        ) from exc


class OptikalExplainer:
    """
    SHAP-based feature attribution for the Optikal Isolation Forest.

    Uses shap.TreeExplainer for efficient exact SHAP values (no sampling
    approximation) because IsolationForest is an ensemble of decision trees.

    Example::

        explainer = OptikalExplainer(feature_names=engineer.get_feature_columns())
        explainer.fit(if_model.model, X_train[:200])   # background sample
        result = explainer.explain(X_single_sample)
        # result["top_features"] → [("feature_name", shap_value), ...]
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        top_k: int = 5,
    ):
        """
        Args:
            feature_names: Ordered list of feature names (length = n_features).
                           If None, features will be labelled f_0, f_1, …
            top_k:         Number of top features to surface in explain() output.
        """
        self.feature_names = feature_names
        self.top_k = top_k
        self._explainer = None
        self.fitted = False

    def fit(
        self,
        if_model,                    # sklearn IsolationForest instance
        X_background: np.ndarray,
        n_background: int = 100,
    ) -> "OptikalExplainer":
        """
        Fit the SHAP TreeExplainer on a background dataset.

        Args:
            if_model:     Trained sklearn IsolationForest (the .model attribute
                          of OptikalIsolationForest, not the wrapper).
            X_background: Scaled feature array used as the SHAP background
                          distribution.  A representative sample of training
                          data works well.
            n_background: Maximum number of background samples to use (sampled
                          without replacement if X_background is larger).

        Returns:
            self (for chaining)
        """
        shap = _require_shap()

        if len(X_background) > n_background:
            idx = np.random.choice(len(X_background), n_background, replace=False)
            background = X_background[idx]
        else:
            background = X_background

        logger.info(
            "Fitting SHAP TreeExplainer on %d background samples, %d features",
            len(background),
            background.shape[1],
        )

        self._explainer = shap.TreeExplainer(if_model, background)
        self.fitted = True
        logger.info("SHAP explainer fitted")
        return self

    def explain(self, X: np.ndarray) -> Dict:
        """
        Compute SHAP values for one or more samples and return a summary.

        Args:
            X: Scaled feature array [n_samples, n_features]

        Returns:
            Dict containing:
              "shap_values"  — raw SHAP array [n_samples, n_features]
              "feature_names"— ordered feature names
              "mean_abs_shap"— mean |SHAP| per feature across samples
              "top_features" — list of (feature_name, mean_abs_shap) for top-k features,
                               sorted descending
        """
        if not self.fitted:
            raise ValueError(
                "Explainer not fitted. Call fit() before explain()."
            )

        shap_values = self._explainer.shap_values(X)

        # IsolationForest TreeExplainer returns a list [normal_class, anomaly_class]
        # We want the anomaly-class values (index 1) when available.
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]

        # Ensure 2D
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        n_features = shap_values.shape[1]
        names = (
            self.feature_names
            if self.feature_names and len(self.feature_names) == n_features
            else [f"f_{i}" for i in range(n_features)]
        )

        mean_abs = np.abs(shap_values).mean(axis=0)
        ranked_idx = np.argsort(mean_abs)[::-1]
        top_features: List[Tuple[str, float]] = [
            (names[i], float(mean_abs[i]))
            for i in ranked_idx[: self.top_k]
        ]

        return {
            "shap_values":   shap_values,
            "feature_names": names,
            "mean_abs_shap": mean_abs,
            "top_features":  top_features,
        }

    def explain_single(self, x: np.ndarray) -> Dict:
        """
        Convenience wrapper for explaining a single sample.

        Args:
            x: 1-D feature array [n_features] or 2-D [1, n_features]

        Returns:
            Same structure as explain(), with scalar SHAP values per feature.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        result = self.explain(x)
        # Flatten to 1-D for single-sample convenience
        result["shap_values"] = result["shap_values"].flatten()
        return result

    def save(self, filepath: str) -> None:
        """Serialise the fitted explainer to disk."""
        if not self.fitted:
            raise ValueError("Explainer not fitted yet")
        joblib.dump(
            {
                "explainer":     self._explainer,
                "feature_names": self.feature_names,
                "top_k":         self.top_k,
            },
            filepath,
        )
        logger.info("Explainer saved to: %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "OptikalExplainer":
        """Load a serialised explainer from disk."""
        saved = joblib.load(filepath)
        obj = cls(
            feature_names=saved["feature_names"],
            top_k=saved["top_k"],
        )
        obj._explainer = saved["explainer"]
        obj.fitted = True
        logger.info("Explainer loaded from: %s", filepath)
        return obj
