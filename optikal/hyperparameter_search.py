"""
Optikal Hyperparameter Optimization

Uses Optuna Bayesian optimization to find the best Isolation Forest
hyperparameters and classification threshold, maximizing F1 on the
validation set.

Requires: pip install optikal[tuning]

Usage::

    results = run_hyperparameter_search(
        X_train, y_train, X_val, y_val, n_trials=50
    )
    print("Best params:", results["best_params"])
    print("Best F1:    ", results["best_value"])
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _require_optuna():
    """Lazy import with a helpful installation hint."""
    try:
        import optuna
        return optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for hyperparameter search. "
            "Install it with: pip install optikal[tuning]"
        )


def _objective(
    trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """
    Optuna objective: train an IsolationForest with trial hyperparameters
    and return validation F1 score.

    Search space:
      - contamination:  [0.05, 0.35]
      - n_estimators:   {50, 100, 150, 200, 250, 300}
      - threshold:      [0.35, 0.70]

    Args:
        trial:   Optuna Trial object
        X_train: Scaled training features
        y_train: Binary training labels
        X_val:   Scaled validation features
        y_val:   Binary validation labels

    Returns:
        F1 score on the validation set (higher = better).
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import f1_score

    contamination = trial.suggest_float("contamination", 0.05, 0.35)
    n_estimators  = trial.suggest_int("n_estimators", 50, 300, step=50)
    threshold     = trial.suggest_float("threshold", 0.35, 0.70)

    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=256,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    raw   = model.decision_function(X_val)
    scores = 1.0 / (1.0 + np.exp(raw))          # sigmoid normalisation
    y_pred = (scores >= threshold).astype(int)

    return float(f1_score(y_val, y_pred, zero_division=0))


def run_hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    study_name: str = "optikal_hp_search",
    storage: Optional[str] = None,
    show_progress: bool = True,
) -> Dict:
    """
    Run Bayesian hyperparameter search for the Optikal Isolation Forest.

    Args:
        X_train:       Scaled training features [n_samples, n_features]
        y_train:       Binary training labels [n_samples]
        X_val:         Scaled validation features
        y_val:         Binary validation labels
        n_trials:      Number of Optuna trials (default 50)
        study_name:    Name for the Optuna study
        storage:       Optional Optuna storage URL, e.g. "sqlite:///optikal_hp.db".
                       Pass None for an in-memory study (results are not persisted).
        show_progress: Show tqdm progress bar during optimization

    Returns:
        Dict with keys:
          - ``best_params`` — dict of best hyperparameter values
          - ``best_value``  — best F1 score achieved
          - ``study``       — the Optuna Study object (for further analysis)
    """
    optuna = _require_optuna()

    logger.info(
        "Starting hyperparameter search — %d trials, study: '%s'",
        n_trials, study_name,
    )

    # Suppress per-trial INFO logs to keep output clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: _objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=show_progress,
    )

    best = study.best_params
    logger.info("Hyperparameter search complete")
    logger.info("  Best F1:            %.4f", study.best_value)
    logger.info("  Best contamination: %.4f", best.get("contamination"))
    logger.info("  Best n_estimators:  %d",   best.get("n_estimators"))
    logger.info("  Best threshold:     %.4f", best.get("threshold"))

    return {
        "best_params": best,
        "best_value":  study.best_value,
        "study":       study,
    }
