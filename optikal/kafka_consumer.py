"""
Optikal Kafka Consumer

Connects to ARGUS Kafka topics, streams real-time AI agent events through
the Optikal threat detection pipeline, and publishes threat alerts back to
a configurable output topic.

ARGUS input topics (default):
  - argus.agent.metrics     — Agent behavioral metrics (ASR)
  - argus.io.validation     — I/O validation results (ASG)
  - argus.policy.decisions  — Authorization decisions (PDE)

Output topic:
  - argus.threat.alerts     — Optikal threat scores + type classifications

Requires: pip install optikal[streaming]

Usage::

    from optikal.kafka_consumer import KafkaConfig, OptikalKafkaConsumer

    consumer = OptikalKafkaConsumer(
        engineer=fitted_engineer,
        if_model=fitted_if_model,
        config=KafkaConfig(bootstrap_servers="kafka:9092"),
        threat_classifier=fitted_classifier,
    )
    consumer.run()           # blocking; Ctrl-C to stop
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _require_kafka():
    """Lazy import with a helpful installation hint."""
    try:
        from confluent_kafka import Consumer, Producer, KafkaError  # noqa: F401
        return Consumer, Producer, KafkaError
    except ImportError:
        raise ImportError(
            "confluent-kafka is required for stream integration. "
            "Install it with: pip install optikal[streaming]"
        )


@dataclass
class KafkaConfig:
    """Kafka connection and topic configuration."""

    bootstrap_servers: str = "localhost:9092"
    consumer_group:    str = "optikal-consumer"
    input_topics:      List[str] = field(default_factory=lambda: [
        "argus.agent.metrics",
        "argus.io.validation",
        "argus.policy.decisions",
    ])
    output_topic:      str   = "argus.threat.alerts"
    poll_timeout_s:    float = 1.0
    auto_offset_reset: str   = "latest"


# ---------------------------------------------------------------------------
# Field mapping from ARGUS event aliases → Optikal required column names
# ---------------------------------------------------------------------------

_ARGUS_FIELD_MAP: Dict[str, str] = {
    "cpu_usage":        "cpu_usage_percent",
    "memory_usage":     "memory_usage_mb",
    "network_calls":    "network_calls_count",
    "file_operations":  "file_operations_count",
    "api_calls":        "api_calls_count",
    "errors":           "error_count",
    "hour":             "hour_of_day",
    "weekday":          "day_of_week",
    # Direct pass-throughs (no alias needed, listed for documentation)
    "query_diversity":  "query_diversity",
    "response_time_ms": "response_time_ms",
    "success_rate":     "success_rate",
}

_REQUIRED_COLUMNS = [
    "cpu_usage_percent", "memory_usage_mb", "network_calls_count",
    "file_operations_count", "api_calls_count", "error_count",
    "hour_of_day", "day_of_week", "query_diversity",
    "response_time_ms", "success_rate",
]


def _parse_argus_event(event: dict) -> Optional[pd.DataFrame]:
    """
    Map an ARGUS Kafka event dict to a single-row Optikal input DataFrame.

    Handles both direct Optikal column names (pass-through) and ARGUS
    abbreviated aliases. Returns None if required fields cannot be resolved.
    """
    row: Dict = {}

    for optikal_col in _REQUIRED_COLUMNS:
        if optikal_col in event:
            row[optikal_col] = event[optikal_col]
        else:
            for alias, mapped in _ARGUS_FIELD_MAP.items():
                if mapped == optikal_col and alias in event:
                    row[optikal_col] = event[alias]
                    break

    missing = [c for c in _REQUIRED_COLUMNS if c not in row]
    if missing:
        logger.debug("Skipping event — missing fields: %s", missing)
        return None

    return pd.DataFrame([row])


class OptikalKafkaConsumer:
    """
    Real-time Kafka consumer that runs the Optikal inference pipeline on
    ARGUS agent metric events and publishes structured threat alerts.

    The consumer runs an Isolation Forest (and optionally a ThreatClassifier)
    on each incoming event. Threat alerts are published to the output topic
    (or routed to a custom ``on_alert`` callback).
    """

    def __init__(
        self,
        engineer,
        if_model,
        config: Optional[KafkaConfig] = None,
        threat_classifier=None,
        on_alert: Optional[Callable[[dict], None]] = None,
    ):
        """
        Args:
            engineer:          Fitted OptikalFeatureEngineer
            if_model:          Fitted OptikalIsolationForest
            config:            Kafka connection config (defaults to localhost)
            threat_classifier: Optional fitted ThreatClassifier
            on_alert:          Optional callback invoked for every threat alert.
                               When None, alerts are published to ``config.output_topic``.
        """
        self.engineer          = engineer
        self.if_model          = if_model
        self.threat_classifier = threat_classifier
        self.config            = config or KafkaConfig()
        self.on_alert          = on_alert
        self._running          = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_alert(
        self,
        event: dict,
        score: float,
        is_threat: bool,
        threat_type: Optional[str],
    ) -> dict:
        """Build a structured threat alert payload."""
        return {
            "agent_id":    event.get("agent_id", "unknown"),
            "timestamp":   event.get("timestamp", time.time()),
            "threat_score": round(float(score), 4),
            "is_threat":   is_threat,
            "threat_type": threat_type,
            "model":       "optikal-isolation-forest",
            "version":     "1.0.0",
        }

    def _process_event(self, event: dict, producer) -> Optional[dict]:
        """
        Run inference on a single ARGUS event.

        Returns:
            Alert dict when a threat is detected; None otherwise.
        """
        row_df = _parse_argus_event(event)
        if row_df is None:
            return None

        try:
            features_df = self.engineer.extract_features(row_df)
            X_scaled, _ = self.engineer.transform(features_df)
        except Exception as exc:
            logger.warning("Feature extraction failed: %s", exc)
            return None

        score     = float(self.if_model.predict_anomaly_score(X_scaled)[0])
        is_threat = score >= 0.5

        threat_type: Optional[str] = None
        if is_threat and self.threat_classifier and self.threat_classifier.fitted:
            threat_type = str(self.threat_classifier.predict(X_scaled)[0])

        alert = self._build_alert(event, score, is_threat, threat_type)

        if is_threat:
            logger.info(
                "Threat detected — agent: %s  score: %.4f  type: %s",
                alert["agent_id"], score, threat_type or "unknown",
            )

        # Route alert
        if self.on_alert:
            self.on_alert(alert)
        elif producer is not None and is_threat:
            producer.produce(
                self.config.output_topic,
                key=str(alert["agent_id"]),
                value=json.dumps(alert),
            )
            producer.poll(0)

        return alert if is_threat else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, max_messages: Optional[int] = None):
        """
        Start the consumer loop (blocking).

        Args:
            max_messages: Stop after processing this many messages.
                          Pass None to run indefinitely (stop with Ctrl-C
                          or by calling ``stop()`` from another thread).
        """
        Consumer, Producer, KafkaError = _require_kafka()

        consumer = Consumer({
            "bootstrap.servers": self.config.bootstrap_servers,
            "group.id":          self.config.consumer_group,
            "auto.offset.reset": self.config.auto_offset_reset,
        })
        producer = Producer({"bootstrap.servers": self.config.bootstrap_servers})

        consumer.subscribe(self.config.input_topics)
        self._running = True

        logger.info(
            "OptikalKafkaConsumer started — topics: %s → %s",
            self.config.input_topics, self.config.output_topic,
        )

        n_processed = n_threats = 0
        try:
            while self._running:
                msg = consumer.poll(timeout=self.config.poll_timeout_s)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() != KafkaError._PARTITION_EOF:
                        logger.error("Kafka error: %s", msg.error())
                    continue

                try:
                    event = json.loads(msg.value().decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    logger.warning("Failed to decode message: %s", exc)
                    continue

                alert = self._process_event(event, producer)
                n_processed += 1
                if alert:
                    n_threats += 1

                if max_messages is not None and n_processed >= max_messages:
                    break

        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        finally:
            consumer.close()
            producer.flush()
            self._running = False
            logger.info(
                "Consumer stopped — messages: %d, threats: %d",
                n_processed, n_threats,
            )

    def stop(self):
        """Signal the ``run()`` loop to exit after the current message."""
        self._running = False
