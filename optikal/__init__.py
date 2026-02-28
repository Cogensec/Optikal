"""
Optikal - AI-Powered Threat Detection Model for ARGUS Platform

GPU-accelerated ML model for detecting security threats in AI agent behaviour.

Model Architecture:
  - OptikalIsolationForest  — Anomaly detection for behavioural outliers
  - OptikalLSTM             — Sequential behavioural pattern analysis
  - OptikalEnsemble         — Combined threat scoring with confidence weighting
  - ThreatClassifier        — Multi-class threat type identification
  - OptikalExplainer        — SHAP-based feature attribution

Created for NVIDIA Morpheus integration with the ARGUS platform.
"""

import json
import logging

__version__ = "1.0.0"
__author__  = "Cogensec"


class _JsonFormatter(logging.Formatter):
    """Formatter that outputs properly escaped JSON log records."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(log_obj)


def configure_logging(level: int = logging.INFO, fmt: str = "plain") -> None:
    """
    Configure package-level logging for Optikal.

    Call this once at application startup. If not called, the standard Python
    logging no-op handler applies (no output by default).

    Args:
        level: Logging level, e.g. logging.DEBUG, logging.INFO, logging.WARNING.
        fmt:   Output format:
                 "plain" — human-readable: ``[2026-02-28 12:00:00] INFO [optikal.trainer] msg``
                 "json"  — structured JSON for log aggregators (ELK, Splunk, etc.):
                           ``{"timestamp": "...", "level": "INFO", "module": "...", "message": "..."}``
    """
    if fmt == "json":
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    pkg_logger = logging.getLogger("optikal")
    pkg_logger.setLevel(level)
    # Remove existing handlers to allow reconfiguration
    for existing_handler in pkg_logger.handlers[:]:
        pkg_logger.removeHandler(existing_handler)
    pkg_logger.addHandler(handler)
    pkg_logger.propagate = False
